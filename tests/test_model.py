"""Testes do motor Bayesian BWM. Executar com: python -m pytest -q"""

import numpy as np
import pandas as pd
import pytest

from bwm import analysis as A
from bwm import data as D
from bwm import ranking as RK
from bwm import report as R
from bwm.model import (_Posterior, bayesian_bwm, classic_bwm_linear, classic_cr, input_based_cr)


@pytest.fixture(scope="module")
def example_result():
    ex = D.example_case()
    res = bayesian_bwm(ex["bo"].values, ex["ow"].values, list(ex["bo"].columns), list(ex["bo"].index),
                       draws=800, burn_in=500, chains=2, seed=1)
    return ex, res


def test_gradient_matches_finite_differences():
    ab = np.array([[1, 2, 4, 8, 3], [2, 1, 4, 7, 3]], float)
    aw = np.array([[8, 4, 2, 1, 3], [4, 7, 2, 1, 3]], float)
    post = _Posterior(ab, aw, 0.01, 0.01)
    theta = np.random.default_rng(0).normal(size=post.dim) * 0.5
    _, g = post.logp_grad(theta)
    num = np.zeros_like(theta)
    for i in range(len(theta)):
        e = np.zeros_like(theta)
        e[i] = 1e-6
        num[i] = (post.logp_grad(theta + e)[0] - post.logp_grad(theta - e)[0]) / 2e-6
    assert np.max(np.abs(num - g)) < 1e-4


def test_weights_are_on_simplex(example_result):
    _, res = example_result
    assert np.allclose(res.w_star.sum(axis=1), 1)
    assert (res.w_star > 0).all()


def test_consistent_single_dm_recovers_weights():
    true = np.array([0.4, 0.3, 0.2, 0.1])
    ab = true[0] / true
    aw = true / true[-1]
    res = bayesian_bwm(ab[None], aw[None], draws=800, burn_in=500, chains=2, seed=3)
    assert np.argmax(res.w_star.mean(axis=0)) == 0
    assert np.argmin(res.w_star.mean(axis=0)) == 3


def test_credal_matrix_is_complementary(example_result):
    _, res = example_result
    P = res.credal_matrix()
    off = ~np.eye(len(P), dtype=bool)
    assert np.allclose((P + P.T)[off], 1, atol=1e-9)


def test_classic_bwm_perfectly_consistent():
    true = np.array([0.5, 0.25, 0.125, 0.125])
    w, xi = classic_bwm_linear(true[0] / true, true / true[-1])
    assert xi < 1e-8
    assert np.allclose(w, true, atol=1e-6)


def test_input_based_cr_zero_when_consistent():
    cr, thr = input_based_cr(np.array([1, 2, 4, 8]), np.array([8, 4, 2, 1]))
    assert cr == 0 and thr is not None


def test_reports_are_generated(example_result):
    ex, res = example_result
    types = dict(zip(ex["criteria"]["Critério"], ex["criteria"]["Tipo"]))
    meta = {"title": ex["title"], "industry": ex["industry"], "context": ex["context"],
            "criteria_df": ex["criteria"]}
    b = R.build_bundle(res, ex["bo"], ex["ow"], meta, ex["alt"], types)
    assert R.html_report(b, offline=False).startswith(b"<!doctype html>")
    assert R.docx_report(b)[:2] == b"PK"
    assert R.excel_report(b)[:2] == b"PK"
    assert len(b["text"]) >= 5


def test_validation_detects_missing_best():
    bo = pd.DataFrame([[2, 3, 4]], columns=list("ABC"), index=["X"])
    ow = pd.DataFrame([[4, 2, 1]], columns=list("ABC"), index=["X"])
    errors, _, _ = D.validate(bo, ow)
    assert any("melhor" in e for e in errors)


def test_alternatives_cost_criteria_inverted(example_result):
    ex, res = example_result
    types = dict(zip(ex["criteria"]["Critério"], ex["criteria"]["Tipo"]))
    ev = A.alternatives_analysis(res, ex["alt"], types)
    assert set(ev["methods"]) == set(RK.METHODS)
    _, N = RK.saw_scores(ev["X"], ev["is_benefit"], res.w_star[:1])
    gama = ev["names"].index("Fornecedor Gama (PL)")
    preco = res.criteria.index("Preço")
    assert N[gama, preco] == pytest.approx(1.0)  # o mais barato recebe 1 num critério de custo


def test_topsis_dominant_alternative_wins():
    X = np.array([[10, 1], [5, 5], [1, 10]], float)   # critério 1 benefício, critério 2 custo
    C, _ = RK.topsis_scores(X, [True, False], np.array([[0.5, 0.5]]))
    assert C[0, 0] == pytest.approx(1.0) and C[0, 2] == pytest.approx(0.0)


def test_vikor_q_in_unit_interval():
    X = np.array([[8, 3], [6, 2], [9, 5], [5, 1]], float)
    Q, S, R_ = RK.vikor_scores(X, [True, False], np.array([[0.6, 0.4], [0.3, 0.7]]))
    assert (Q >= 0).all() and (Q <= 1).all()
    assert Q.shape == (2, 4) and np.argmin(Q[0]) == 0


def test_sensitivity_extremes():
    X = np.array([[10, 1], [1, 10]], float)
    g, raw, sw = RK.sensitivity("SAW", X, [True, True], np.array([0.5, 0.5]), 0)
    assert raw[-1, 0] > raw[-1, 1] and raw[0, 1] > raw[0, 0] and len(sw) == 1


def test_classic_cr_zero_when_consistent():
    xi, cr = classic_cr(np.array([1, 2, 4, 8]), np.array([8, 4, 2, 1]))
    assert xi == 0 and cr == 0


@pytest.mark.parametrize("name", list(D.EXAMPLES))
def test_all_examples_are_valid(name):
    ex = D.example_case(name)
    errors, warnings, _ = D.validate(ex["bo"], ex["ow"])
    assert not errors and not warnings
    assert list(ex["alt"].columns) == ex["criteria"]["Critério"].tolist()


# --- Modelos, leitura de ficheiros e leitura por setor ----------------------

import io as _io
import json as _json

from bwm import ai as AI
from bwm import templates as T


class _Up(_io.BytesIO):
    def __init__(self, data, name):
        super().__init__(data)
        self.name = name

    def getvalue(self):
        return super().getvalue()


@pytest.mark.parametrize("name", list(D.EXAMPLES))
def test_excel_template_round_trip(name):
    ex = D.example_case(name)
    meta = {"title": ex["title"], "sector": ex["sector"], "context": ex["context"]}
    xl = T.excel_template(ex["criteria"], ex["dms"], ex["bo"], ex["ow"], ex["alt"], meta)
    out = D.read_uploaded(_Up(xl, "m.xlsx"))
    assert np.allclose(out["bo"].values, ex["bo"].values)
    assert np.allclose(out["ow"].values, ex["ow"].values)
    assert list(out["criteria"]["Unidade"]) == list(ex["criteria"]["Unidade"])
    assert out["meta"]["sector"] == ex["sector"]
    assert np.allclose(out["alt"].values.astype(float), ex["alt"].values)


def test_blank_template_reads_without_alternatives():
    xl = T.excel_template(D.industry_criteria_df("Energia"), D.default_dm_df(2))
    out = D.read_uploaded(_Up(xl, "b.xlsx"))
    assert out["bo"].shape == (2, 6) and "alt" not in out


def test_questionnaire_is_docx():
    ex = D.example_case()
    assert T.questionnaire_docx(ex["criteria"], {"title": "x"})[:2] == b"PK"


def test_ai_payload_is_json_and_report_includes_reading(example_result):
    ex, res = example_result
    types = dict(zip(ex["criteria"]["Critério"], ex["criteria"]["Tipo"]))
    meta = {"title": ex["title"], "sector": "Setor X", "context": ex["context"], "criteria_df": ex["criteria"],
            "units": dict(zip(ex["criteria"]["Critério"], ex["criteria"]["Unidade"])),
            "ai_text": "### Título\n- ponto **forte**"}
    b = R.build_bundle(res, ex["bo"], ex["ow"], meta, ex["alt"], types)
    _json.dumps(AI.build_payload(b), ensure_ascii=False)
    html_ = R.html_report(b, offline=False).decode()
    assert "Leitura para o setor" in html_ and "<b>forte</b>" in html_ and "Setor X" in html_
    assert R.docx_report(b)[:2] == b"PK"

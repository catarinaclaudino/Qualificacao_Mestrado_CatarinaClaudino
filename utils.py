"""
utils.py
========
Funções auxiliares compartilhadas por todas as etapas do pipeline.
Mantém cada script "00_..py" .. "10_..py" enxuto e evita duplicar lógica de
logging, descoberta de arquivos e identificação de colunas.
"""

import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy import stats

import config


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    try:
        config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(config.LOGS_DIR / "pipeline.log", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except Exception:
        pass
    return logger


# ---------------------------------------------------------------------------
# Descoberta de arquivos e colunas (Passos 1-3: "não assumir nomes de coluna")
# ---------------------------------------------------------------------------
def discover_data_files(data_dir):
    """Varre recursivamente `data_dir` e retorna todos os arquivos suportados."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return []
    files = [
        p for p in data_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower() in config.SUPPORTED_EXTENSIONS
        and not p.name.startswith("~$")  # arquivos temporários do Excel
    ]
    return sorted(files)


def find_column(columns, keywords):
    """
    Retorna o primeiro nome de coluna original cujo nome normalizado contenha
    alguma das `keywords` (também normalizadas). Retorna None se não achar.
    """
    normed = {c: str(c).strip().lower().replace(" ", "_").replace("-", "_") for c in columns}
    for kw in keywords:
        kw_n = kw.lower().replace(" ", "_").replace("-", "_")
        for orig, n in normed.items():
            if kw_n in n:
                return orig
    return None


# ---------------------------------------------------------------------------
# Persistência de artefatos intermediários
# ---------------------------------------------------------------------------
def save_pickle(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(obj, fh)


def load_pickle(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# Modelos de degradação (usados nos Passos 6-9)
# ---------------------------------------------------------------------------
def model_linear(n, b0, b1):
    return b0 + b1 * n


def model_poly(n, *betas):
    n = np.asarray(n, dtype=float)
    out = np.zeros_like(n)
    for i, b in enumerate(betas):
        out = out + b * n ** i
    return out


def model_exponential(n, b0, b1):
    n = np.asarray(n, dtype=float)
    return b0 * np.exp(-b1 * n)


def model_logarithmic(n, b0, b1):
    n = np.asarray(n, dtype=float)
    return b0 - b1 * np.log(np.clip(n, 1e-6, None))


def model_power_law(n, b0, b1, b2):
    n = np.asarray(n, dtype=float)
    return b0 - b1 * np.power(np.clip(n, 1e-6, None), b2)


MODEL_REGISTRY = {
    "Linear": {"func": model_linear, "n_params": 2, "p0": [100.0, -0.01]},
    "Polynomial_2": {"func": None, "n_params": 3, "p0": None},  # tratado via numpy.polyfit
    "Exponential": {"func": model_exponential, "n_params": 2, "p0": [100.0, 0.0005]},
    "Logarithmic": {"func": model_logarithmic, "n_params": 2, "p0": [100.0, 1.0]},
    "Power_Law": {"func": model_power_law, "n_params": 3, "p0": [100.0, 0.01, 1.0]},
}


def regression_metrics(y_true, y_pred, n_params):
    """R2, R2 ajustado, RMSE, MAE e desvio-padrão residual."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    resid = y_true - y_pred
    n = len(y_true)
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    if n - n_params - 1 > 0:
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_params - 1)
    else:
        adj_r2 = np.nan
    rmse = np.sqrt(ss_res / n)
    mae = np.mean(np.abs(resid))
    # desvio-padrão residual (usando graus de liberdade do modelo)
    dof = max(n - n_params, 1)
    resid_std = np.sqrt(ss_res / dof)
    return {
        "r2": r2,
        "adj_r2": adj_r2,
        "rmse": rmse,
        "mae": mae,
        "resid_std": resid_std,
        "n": n,
        "n_params": n_params,
        "rss": ss_res,
    }


def compute_bic(rss, n, k):
    """
    Bayesian Information Criterion (Schwarz, 1978): BIC = n*ln(RSS/n) + k*ln(n).

    Used as the primary quantitative model-selection criterion (Passo 7),
    computed EXCLUSIVELY on the fitting/training partition of the data.
    Lower BIC indicates a better parsimony-adjusted fit. `rss` must already
    be the residual sum of squares on that same partition.
    """
    if n <= 0 or rss <= 0:
        return np.nan
    return float(n * np.log(rss / n) + k * np.log(n))


def prediction_interval_margin(n_query, n_arr, resid_std, n_params, alpha):
    """
    Margem do intervalo de predição de 95% (ou 1-`alpha`) em qualquer ponto
    `n_query` (escalar ou array), usando t de Student e um fator de leverage
    que infla a margem conforme a distância ao "centro de massa" dos dados
    de treino/full-fit (`n_arr`) - aproximação padrão para modelos não
    lineares (Passo 8, README Step 8: "Construct 95% prediction intervals").

    Fatorado aqui (2026-09-07) para ser IDENTICO entre o Passo 8
    (`07_uncertainty_quantification.py`, banda em torno da curva de SoH,
    Table_08/Figura 11) e o Passo 10 (`09_failure_analysis_and_rul.py`,
    propagação para um range de N_F_hat/RUL, Table_10/Figuras 13-14) -
    pedido explícito da pesquisadora para que o range de RUL derive
    exatamente da mesma banda de incerteza já reportada no Passo 8, e não
    de um cálculo novo e potencialmente inconsistente.

    Retorna (margin, t_crit, dof).
    """
    n_arr = np.asarray(n_arr, dtype=float)
    n_query = np.asarray(n_query, dtype=float)
    n_obs = len(n_arr)
    dof = max(n_obs - n_params, 1)
    t_crit = stats.t.ppf(1 - alpha / 2, dof)
    mean_n = np.mean(n_arr)
    span_n = max(n_arr.max() - n_arr.min(), 1.0)
    leverage_factor = 1.0 + ((n_query - mean_n) / span_n) ** 2 / n_obs
    margin = t_crit * resid_std * np.sqrt(leverage_factor)
    return margin, float(t_crit), int(dof)


def check_monotonicity(y_grid, tolerance):
    """
    Physical-plausibility screen (Passo 7): checks whether a predicted SoH
    trajectory evaluated on a dense grid is non-increasing (dSoH/dN <= 0),
    within `tolerance` (absolute, in %SoH units) to avoid flagging
    floating-point noise as a meaningful increase.

    Returns (monotonicity_pass: bool, max_increase: float) where
    `max_increase` is the largest positive step found (0.0 or negative-most
    value if trajectory is non-increasing everywhere).
    """
    y_grid = np.asarray(y_grid, dtype=float)
    diffs = np.diff(y_grid)
    max_increase = float(np.max(diffs)) if len(diffs) else 0.0
    passes = bool(np.all(diffs <= tolerance))
    return passes, max_increase


# ---------------------------------------------------------------------------
# Pettitt's non-parametric change-point test
# ---------------------------------------------------------------------------
def pettitt_test(x):
    """
    Teste de Pettitt (1979) para detecção de ponto de mudança em uma série.

    Retorna dict com:
      K          - estatística máxima |U_t|
      tau_index  - índice (0-based) do ponto de mudança na série `x`
      p_value    - valor-p aproximado (Pettitt, 1979)
      U          - vetor completo da estatística U_t
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 3:
        return {"K": 0.0, "tau_index": None, "p_value": 1.0, "U": np.array([])}

    # sign(x_i - x_j) para todo par (i, j)
    diff = x[:, None] - x[None, :]
    sgn = np.sign(diff)

    # U_t = soma_{i<=t} soma_{j>t} sign(x_i - x_j),  t = 1..n-1
    cum_over_i = np.cumsum(sgn, axis=0)  # cum_over_i[t-1, j] = soma_{i=1}^{t} sign(x_i - x_j)
    U = np.array([cum_over_i[t - 1, t:].sum() for t in range(1, n)])

    K = np.max(np.abs(U))
    tau_index = int(np.argmax(np.abs(U)))  # posição 0-based dentro de x (mudança ocorre após este índice)
    p_value = 2.0 * np.exp((-6.0 * K ** 2) / (n ** 3 + n ** 2))
    p_value = float(min(max(p_value, 0.0), 1.0))
    return {"K": float(K), "tau_index": tau_index, "p_value": p_value, "U": U}


def check_persistence(residuals, tau_index, window):
    """
    Critério de persistência: a mudança detectada em `tau_index` deve
    persistir por `window` observações subsequentes, i.e., o sinal do desvio
    em relação à média pré-mudança deve se manter na maioria das observações
    seguintes.
    """
    residuals = np.asarray(residuals, dtype=float)
    n = len(residuals)
    if tau_index is None or tau_index + window >= n or tau_index < 1:
        return False, np.nan

    pre_mean = np.mean(residuals[:tau_index + 1])
    post_window = residuals[tau_index + 1: tau_index + 1 + window]
    shift_sign = np.sign(np.mean(post_window) - pre_mean)
    if shift_sign == 0:
        return False, 0.0
    agreement = np.mean(np.sign(post_window - pre_mean) == shift_sign)
    persists = agreement >= 0.6  # maioria (>=60%) das observações confirma a direção da mudança
    return bool(persists), float(agreement)

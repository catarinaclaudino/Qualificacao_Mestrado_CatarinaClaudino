"""
05_degradation_models.py
==========================
PASSO 6: Modelagem estatística da degradação.

Ajusta, para cada bateria, os 5 modelos candidatos descritos no README:
    - Linear:        SoH(N) = b0 + b1*N
    - Polinomial:     grau 2
    - Exponencial:    SoH(N) = b0 * exp(-b1*N)
    - Logarítmico:    SoH(N) = b0 - b1*ln(N)
    - Lei de potência: SoH(N) = b0 - b1*N^b2

Esta etapa apenas AJUSTA cada modelo a TODOS os dados (uso descritivo/
visual, figura 05) e calcula suas métricas de ajuste completo (R², R²
ajustado, RMSE, MAE, desvio-padrão residual, nº de parâmetros). A
SELEÇÃO do melhor modelo NAO ocorre aqui - ocorre no Passo 7
(06_residual_analysis_and_validation.py), exclusivamente com base nos
dados de treino/ajuste (ver hierarquia de seleção documentada lá).

Saída:
    tables/Table_05_Model_Comparison.csv
    processed_data/fitted_models.pkl      (parâmetros de cada modelo por bateria)
    processed_data/battery_data_info.pkl  (metadados por bateria)
    figures/05_candidate_models_comparison.png

Pode ser executado isoladamente (requer 03_soh_calculation.py):
    python 05_degradation_models.py
"""

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

import config
import utils

log = utils.get_logger("05_degradation_models")

MODEL_NAMES = ["Linear", "Polynomial_2",
               "Exponential", "Logarithmic", "Power_Law"]


def fit_one_model(name, n, y):
    """Ajusta um modelo e retorna (params, y_pred, n_params) ou None se falhar."""
    n = np.asarray(n, dtype=float)
    y = np.asarray(y, dtype=float)

    try:
        if name == "Linear":
            params = np.polyfit(n, y, 1)  # [b1, b0] (numpy: maior grau primeiro)
            y_pred = np.polyval(params, n)
            return {"type": "polyfit", "coeffs": params.tolist(), "degree": 1}, y_pred, 2

        if name.startswith("Polynomial_"):
            degree = int(name.split("_")[1])
            params = np.polyfit(n, y, degree)
            y_pred = np.polyval(params, n)
            return {"type": "polyfit", "coeffs": params.tolist(), "degree": degree}, y_pred, degree + 1

        if name == "Exponential":
            p0 = [max(y[0], 1.0), 1e-4]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, _ = curve_fit(utils.model_exponential, n, y, p0=p0, maxfev=20000)
            y_pred = utils.model_exponential(n, *popt)
            return {"type": "curve_fit", "func": "exponential", "params": popt.tolist()}, y_pred, 2

        if name == "Logarithmic":
            p0 = [y[0], 1.0]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, _ = curve_fit(utils.model_logarithmic, n, y, p0=p0, maxfev=20000)
            y_pred = utils.model_logarithmic(n, *popt)
            return {"type": "curve_fit", "func": "logarithmic", "params": popt.tolist()}, y_pred, 2

        if name == "Power_Law":
            p0 = [y[0], 0.01, 1.0]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, _ = curve_fit(utils.model_power_law, n, y, p0=p0, maxfev=20000,
                                     bounds=([-np.inf, -np.inf, 0.01], [np.inf, np.inf, 5]))
            y_pred = utils.model_power_law(n, *popt)
            return {"type": "curve_fit", "func": "power_law", "params": popt.tolist()}, y_pred, 3

    except Exception as e:  # noqa: BLE001
        log.warning(f"Falha ao ajustar modelo '{name}': {e}")
        return None, None, None

    return None, None, None


def predict_with_stored_model(model_info, n):
    n = np.asarray(n, dtype=float)
    if model_info["type"] == "polyfit":
        return np.polyval(model_info["coeffs"], n)
    func_map = {
        "exponential": utils.model_exponential,
        "logarithmic": utils.model_logarithmic,
        "power_law": utils.model_power_law,
    }
    func = func_map[model_info["func"]]
    return func(n, *model_info["params"])


def main():
    log.info("=" * 70)
    log.info("PASSO 6: MODELAGEM ESTATÍSTICA DA DEGRADAÇÃO")
    log.info("=" * 70)

    in_path = config.PROCESSED_DATA_DIR / "soh_data.csv"
    if not in_path.exists():
        log.error(f"Arquivo não encontrado: {in_path}. Rode antes: python 03_soh_calculation.py")
        raise SystemExit(1)

    df = pd.read_csv(in_path)

    all_fitted_models = {}
    battery_info = {}
    comparison_rows = []

    n_batteries = df["battery_id"].nunique()
    n_cols = min(n_batteries, 2)
    n_rows = int(np.ceil(n_batteries / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows), squeeze=False)
    axes_flat = axes.flatten()

    for idx, (battery_id, bdf) in enumerate(df.groupby("battery_id", sort=True)):
        bdf = bdf.sort_values("cycle").reset_index(drop=True)
        n_arr = bdf["cycle"].to_numpy(dtype=float)
        y_arr = bdf["soh"].to_numpy(dtype=float)

        battery_info[battery_id] = {
            "n_cycles": len(bdf),
            "cycle_min": float(n_arr.min()),
            "cycle_max": float(n_arr.max()),
            "soh_min": float(y_arr.min()),
            "soh_max": float(y_arr.max()),
        }

        ax = axes_flat[idx]
        ax.scatter(n_arr, y_arr, s=8, color="black", alpha=0.5, label="Observed data")

        all_fitted_models[battery_id] = {}
        for name in MODEL_NAMES:
            model_info, y_pred, n_params = fit_one_model(name, n_arr, y_arr)
            if model_info is None:
                continue
            metrics = utils.regression_metrics(y_arr, y_pred, n_params)
            all_fitted_models[battery_id][name] = model_info
            comparison_rows.append({
                "battery_id": battery_id,
                "model": name,
                **metrics,
            })
            ax.plot(n_arr, y_pred, linewidth=1.3, label=name)

        ax.axhline(config.SOH_FAILURE_THRESHOLD, color="red", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_title(f"Battery {battery_id}")
        ax.set_xlabel("Cycle")
        ax.set_ylabel("SoH (%)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

        log.info(f"Bateria {battery_id}: {len(all_fitted_models[battery_id])} modelo(s) ajustado(s) com sucesso")

    # remove eixos vazios (grid maior que o nº de baterias)
    for j in range(n_batteries, len(axes_flat)):
        fig.delaxes(axes_flat[j])

    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "05_candidate_models_comparison.png", dpi=config.FIGURE_DPI)
    plt.close(fig)

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(config.TABLES_DIR / "Table_05_Model_Comparison.csv", index=False)
    log.info(f"Tabela de comparação de modelos salva: {config.TABLES_DIR / 'Table_05_Model_Comparison.csv'}")

    utils.save_pickle(all_fitted_models, config.PROCESSED_DATA_DIR / "fitted_models.pkl")
    utils.save_pickle(battery_info, config.PROCESSED_DATA_DIR / "battery_data_info.pkl")
    log.info("Modelos ajustados e metadados salvos (fitted_models.pkl, battery_data_info.pkl)")
    log.info("Passo 6 concluído.")
    return comparison_df


if __name__ == "__main__":
    main()

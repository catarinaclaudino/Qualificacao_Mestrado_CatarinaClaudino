"""
07_uncertainty_quantification.py
==================================
PASSO 8: Quantificação de incerteza.

- Usa o desvio-padrão residual (s_e) do melhor modelo (Passo 7)
- Constrói intervalos de predição de 95% (t de Student, considerando tanto a
  incerteza do modelo quanto a variabilidade da medição)
- Analisa como a incerteza cresce à medida que se extrapola além do intervalo
  de ciclos observado
- Distingue visualmente predições dentro do range observado (interpolação)
  de predições extrapoladas

Desde 2026-09-07, `best_models.pkl` guarda, por bateria, um dict de
"roles" ("selected" e, opcionalmente, "comparison" - ver
config.COMPARISON_MODEL_NAME, pedido EXPLÍCITO da pesquisadora, NAO uma
inferência deste código). Este script roda o MESMO procedimento de
incerteza para CADA role presente, produzindo tabelas/figuras separadas
(sufixo de arquivo distingue o role/modelo) - a lógica de cálculo em si
é INALTERADA.

Saída:
    tables/Table_08_Uncertainty_Metrics.csv (agora com coluna "role")
    figures/11_prediction_intervals.png (+ _comparison_<Modelo>.png se aplicável)

Pode ser executado isoladamente (requer 06_residual_analysis_and_validation.py):
    python 07_uncertainty_quantification.py
"""

import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import config
import utils

log = utils.get_logger("07_uncertainty_quantification")


def _import_step05():
    base_dir = Path(config.__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("step05", base_dir / "05_degradation_models.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["step05"] = mod
    spec.loader.exec_module(mod)
    return mod


step05 = _import_step05()


def main():
    log.info("=" * 70)
    log.info("PASSO 8: QUANTIFICAÇÃO DE INCERTEZA")
    log.info("=" * 70)

    in_path = config.PROCESSED_DATA_DIR / "best_models.pkl"
    if not in_path.exists():
        log.error(f"Arquivo não encontrado: {in_path}. "
                  f"Rode antes: python 06_residual_analysis_and_validation.py")
        raise SystemExit(1)

    best_models = utils.load_pickle(in_path)

    uncertainty_rows = []

    for battery_id, entries_by_role in best_models.items():
      for role, info in entries_by_role.items():
        n_arr = info["cycle"]
        y_arr = info["soh"]
        resid_std = info["resid_std"]
        n_params = info["n_params"]

        cycle_max_observed = n_arr.max()
        extra_span = 0.30 * (n_arr.max() - n_arr.min())  # 30% de extrapolação além do observado
        n_extrap = np.linspace(cycle_max_observed, cycle_max_observed + extra_span, 60)
        n_full_range = np.concatenate([np.linspace(n_arr.min(), cycle_max_observed, 200), n_extrap[1:]])

        y_pred_full_range = step05.predict_with_stored_model(info["model_info"], n_full_range)

        # margem de predição (mesma fórmula usada no Passo 10 para propagar este
        # intervalo em um range de N_F_hat/RUL - ver utils.prediction_interval_margin)
        margin, t_crit, dof = utils.prediction_interval_margin(
            n_full_range, n_arr, resid_std, n_params, config.ALPHA)

        lower = y_pred_full_range - margin
        upper = y_pred_full_range + margin
        is_extrapolated = n_full_range > cycle_max_observed

        margin_at_last_observed = margin[~is_extrapolated][-1] if np.any(~is_extrapolated) else np.nan
        margin_at_extrap_end = margin[-1]

        uncertainty_rows.append({
            "battery_id": battery_id,
            "role": role,
            "best_model": info["model_name"],
            "resid_std_se": resid_std,
            "t_critical_95pct": t_crit,
            "dof": dof,
            "margin_at_last_observed_cycle": margin_at_last_observed,
            "margin_at_30pct_extrapolation": margin_at_extrap_end,
            "uncertainty_growth_factor": (
                margin_at_extrap_end / margin_at_last_observed
                if margin_at_last_observed and margin_at_last_observed > 0 else np.nan
            ),
        })

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.scatter(n_arr, y_arr, s=10, color="black", alpha=0.5, label="Observed data")
        ax.plot(n_full_range[~is_extrapolated], y_pred_full_range[~is_extrapolated],
                color="tab:blue", linewidth=1.5, label=f"Fitted model ({info['model_name']})")
        ax.plot(n_full_range[is_extrapolated], y_pred_full_range[is_extrapolated],
                color="tab:blue", linewidth=1.5, linestyle="--", label="Model extrapolation")
        ax.fill_between(n_full_range, lower, upper, color="tab:blue", alpha=0.15,
                         label="95% prediction interval")
        ax.axvline(cycle_max_observed, color="gray", linestyle=":", linewidth=1)
        ax.axhline(config.SOH_FAILURE_THRESHOLD, color="red", linestyle="--", linewidth=1,
                   label=f"Functional Failure threshold ({config.SOH_FAILURE_THRESHOLD:.0f}%)")
        ax.set_xlabel("Cycle")
        ax.set_ylabel("SoH (%)")
        title_suffix = "" if role == "selected" else f" (comparison model: {info['model_name']})"
        ax.set_title(f"Prediction Intervals (95%) — {battery_id}{title_suffix}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        battery_suffix = "" if len(best_models) == 1 else f"_{battery_id}"
        role_suffix = "" if role == "selected" else f"_comparison_{info['model_name']}"
        fig.savefig(config.FIGURES_DIR / f"11_prediction_intervals{battery_suffix}{role_suffix}.png",
                    dpi=config.FIGURE_DPI)
        plt.close(fig)

        log.info(f"Bateria {battery_id} [role={role}, modelo={info['model_name']}]: s_e={resid_std:.3f} | "
                  f"margem @ último ciclo observado={margin_at_last_observed:.3f} p.p. | "
                  f"margem @ 30% de extrapolação={margin_at_extrap_end:.3f} p.p.")

    pd.DataFrame(uncertainty_rows).to_csv(config.TABLES_DIR / "Table_08_Uncertainty_Metrics.csv", index=False)
    log.info(f"Tabela de incerteza salva em: {config.TABLES_DIR / 'Table_08_Uncertainty_Metrics.csv'}")
    log.info("Passo 8 concluído.")
    return pd.DataFrame(uncertainty_rows)


if __name__ == "__main__":
    main()

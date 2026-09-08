"""
09_failure_analysis_and_rul.py
=================================
PASSO 10: Análise de falha e RUL (Remaining Useful Life).

- Falha Funcional: critério convencional de SoH = 80% (config.SOH_FAILURE_THRESHOLD)
  * N_F_hat é estimado por interpolação (se o SoH já cruzou o limiar dentro do
    range observado) ou por extrapolação do melhor modelo (Passo 7)
- RUL(N_c) = N_F_hat - N_c, calculado para cada ciclo observado; decresce
  monotonicamente à medida que os ciclos avançam
- Intervalo P-F = N_F_hat - N_P (apenas se uma Potencial Falha foi detectada
  no Passo 9); caso contrário, não é calculado

Desde 2026-09-07, `best_models.pkl` e `pettitt_results.pkl` guardam, por
bateria, um dict de "roles" ("selected" e, opcionalmente, "comparison" -
ver config.COMPARISON_MODEL_NAME, pedido EXPLÍCITO da pesquisadora). Este
script roda a MESMA análise de falha/RUL para CADA role presente.

ADENDO 4 (2026-09-07) - Range de incerteza do RUL, pedido EXPLÍCITO da
pesquisadora. Antes deste adendo, N_F_hat e RUL eram valores pontuais
únicos, desconectados da banda de predição de 95% já calculada no Passo 8
(Table_08/Figura 11, em pontos percentuais de SoH). Agora esta mesma banda
(via `utils.prediction_interval_margin`, idêntica ao Passo 8) é propagada
para o domínio de ciclos: procuramos onde a curva SoH_modelo(N) - margem(N)
cruza o limiar (N_F_hat_lower_bound_95pct, falha MAIS PRECOCE/pessimista) e
onde SoH_modelo(N) + margem(N) cruza o limiar (N_F_hat_upper_bound_95pct,
falha MAIS TARDIA/otimista). RUL_lower_bound_cycles(N_c) e
RUL_upper_bound_cycles(N_c) seguem da mesma forma. Isso NAO estava definido
no README original (Step 8 define apenas a banda sobre a curva de SoH;
Step 10 definia RUL como valor único) - foi expressamente autorizado nesta
conversa antes de qualquer implementação.

Saída:
    tables/Table_10_Failure_and_RUL_Analysis.csv (colunas "role"/"model_name" +
        N_F_hat_lower_bound_95pct/N_F_hat_upper_bound_95pct)
    tables/processed_data_RUL_detail.csv (colunas "role"/"model_name" +
        RUL_lower_bound_cycles/RUL_upper_bound_cycles)
    figures/13_failure_analysis.png (+ _comparison_<Modelo>.png se aplicável) -
        agora com faixa sombreada do range de N_F_hat
    figures/14_rul_vs_cycle.png (+ _comparison_<Modelo>.png se aplicável) -
        agora com faixa sombreada do range de RUL

Pode ser executado isoladamente (requer os Passos 6, 7 e 8):
    python 09_failure_analysis_and_rul.py
"""

import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq

import config
import utils

log = utils.get_logger("09_failure_analysis_and_rul")


def _import_step05():
    base_dir = Path(config.__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("step05", base_dir / "05_degradation_models.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["step05"] = mod
    spec.loader.exec_module(mod)
    return mod


step05 = _import_step05()


def estimate_failure_cycle(model_info, cycle_arr, soh_arr, threshold):
    """
    Estima o ciclo em que SoH cruza `threshold`.
    - Se o SoH observado já cruzou o limiar, interpola linearmente entre os
      dois ciclos observados mais próximos do cruzamento.
    - Caso contrário, extrapola o modelo ajustado buscando a raiz de
      f(N) = modelo(N) - threshold além do último ciclo observado.
    Retorna (N_F_hat, method).
    """
    below = soh_arr <= threshold
    if np.any(below):
        idx_cross = np.argmax(below)  # primeiro índice onde soh <= threshold
        if idx_cross == 0:
            return float(cycle_arr[0]), "observed (first cycle already below threshold)"
        n0, n1 = cycle_arr[idx_cross - 1], cycle_arr[idx_cross]
        s0, s1 = soh_arr[idx_cross - 1], soh_arr[idx_cross]
        if s1 == s0:
            n_f = n1
        else:
            n_f = n0 + (threshold - s0) * (n1 - n0) / (s1 - s0)
        return float(n_f), "interpolation (observed data)"

    # extrapolação: busca raiz do modelo além do range observado
    f = lambda n: step05.predict_with_stored_model(model_info, np.array([n]))[0] - threshold  # noqa: E731
    n_max = cycle_arr.max()
    search_upper = n_max
    step = max((n_max - cycle_arr.min()) * 0.5, 50)
    found = False
    for _ in range(40):
        search_upper += step
        if f(n_max) * f(search_upper) < 0:
            found = True
            break
    if not found:
        return np.nan, "extrapolation (model does not cross threshold within a reasonable horizon)"
    n_f = brentq(f, n_max, search_upper)
    return float(n_f), "extrapolation (fitted model)"


def _bound_curve(model_info, n_arr, resid_std, n_params, alpha, sign):
    """
    Retorna uma função SoH_bound(n) = modelo(n) + sign*margem_95%(n), usando
    EXATAMENTE a mesma margem do Passo 8 (`utils.prediction_interval_margin`).
    sign=-1 -> banda inferior de SoH (cenário pessimista: cruza o limiar MAIS
    CEDO); sign=+1 -> banda superior de SoH (cenário otimista: cruza MAIS
    TARDE).
    """
    def curve(n_query):
        n_query = np.atleast_1d(np.asarray(n_query, dtype=float))
        y = step05.predict_with_stored_model(model_info, n_query)
        margin, _, _ = utils.prediction_interval_margin(n_query, n_arr, resid_std, n_params, alpha)
        return y + sign * margin
    return curve


def _find_curve_crossing(curve_fn, n_search_start, n_span_ref, threshold,
                          search_multiplier=10, grid_points=6000):
    """
    Varre um grid denso a partir de `n_search_start` (tipicamente o primeiro
    ciclo observado) até `n_search_start + search_multiplier * n_span_ref`,
    procurando o primeiro cruzamento de `curve_fn(n) - threshold`, e refina
    com brentq. Usado para propagar a banda de predição de 95% (Passo 8)
    para um range de N_F_hat (Adendo 4). Retorna (n_root, found, method).
    """
    span = max(n_span_ref, 1.0)
    n_grid = np.linspace(n_search_start, n_search_start + search_multiplier * span + 1000, grid_points)
    f_grid = curve_fn(n_grid) - threshold
    sign_changes = np.where(np.diff(np.sign(f_grid)) != 0)[0]
    if len(sign_changes) == 0:
        return np.nan, False, "not found within search horizon"
    idx = sign_changes[0]
    lo, hi = float(n_grid[idx]), float(n_grid[idx + 1])
    f = lambda n: float(curve_fn(np.array([n]))[0]) - threshold  # noqa: E731
    try:
        n_root = brentq(f, lo, hi)
    except Exception:
        return np.nan, False, "root-finding failed"
    method = ("crossing within observed range" if n_root <= n_span_ref + n_search_start
              else "crossing via extrapolated bound curve")
    return float(n_root), True, method


def main():
    log.info("=" * 70)
    log.info("PASSO 10: ANÁLISE DE FALHA E RUL")
    log.info("=" * 70)

    best_models_path = config.PROCESSED_DATA_DIR / "best_models.pkl"
    pettitt_path = config.PROCESSED_DATA_DIR / "pettitt_results.pkl"
    if not best_models_path.exists():
        log.error(f"Arquivo não encontrado: {best_models_path}. "
                  f"Rode antes: python 06_residual_analysis_and_validation.py")
        raise SystemExit(1)

    best_models = utils.load_pickle(best_models_path)
    pettitt_results = utils.load_pickle(pettitt_path) if pettitt_path.exists() else {}

    rul_rows = []
    summary_rows = []

    for battery_id, entries_by_role in best_models.items():
      pettitt_results_battery = pettitt_results.get(battery_id, {})
      for role, info in entries_by_role.items():
        cycle = info["cycle"]
        soh = info["soh"]
        model_info = info["model_info"]
        resid_std = info["resid_std"]
        n_params = info["n_params"]

        n_f_hat, method = estimate_failure_cycle(model_info, cycle, soh, config.SOH_FAILURE_THRESHOLD)
        if not np.isnan(n_f_hat):
            log.info(f"Bateria {battery_id} [role={role}, modelo={info['model_name']}]: N_F_hat = "
                      f"{n_f_hat:.1f} (método: {method})")
        else:
            log.info(f"Bateria {battery_id} [role={role}, modelo={info['model_name']}]: N_F_hat "
                      f"não pôde ser estimado (método: {method})")

        # Adendo 4: propaga a banda de predição de 95% do Passo 8 para um
        # range de N_F_hat/RUL (pedido explícito da pesquisadora).
        lower_soh_curve = _bound_curve(model_info, cycle, resid_std, n_params, config.ALPHA, sign=-1)
        upper_soh_curve = _bound_curve(model_info, cycle, resid_std, n_params, config.ALPHA, sign=+1)
        n_f_hat_lower, found_lower, method_lower = _find_curve_crossing(
            lower_soh_curve, cycle.min(), cycle.max() - cycle.min(), config.SOH_FAILURE_THRESHOLD)
        n_f_hat_upper, found_upper, method_upper = _find_curve_crossing(
            upper_soh_curve, cycle.min(), cycle.max() - cycle.min(), config.SOH_FAILURE_THRESHOLD)
        if found_lower and found_upper:
            log.info(f"Bateria {battery_id} [role={role}]: range de 95% para N_F_hat = "
                      f"[{n_f_hat_lower:.1f}, {n_f_hat_upper:.1f}] "
                      f"(pessimista: {method_lower}; otimista: {method_upper})")
        else:
            log.warning(f"Bateria {battery_id} [role={role}]: range de 95% para N_F_hat não pôde "
                        f"ser totalmente estimado (lower found={found_lower}, upper found={found_upper})")

        for n_c, s in zip(cycle, soh):
            rul = n_f_hat - n_c if not np.isnan(n_f_hat) else np.nan
            rul_lower = (n_f_hat_lower - n_c) if not np.isnan(n_f_hat_lower) else np.nan
            rul_upper = (n_f_hat_upper - n_c) if not np.isnan(n_f_hat_upper) else np.nan
            rul_rows.append({
                "battery_id": battery_id,
                "role": role,
                "model_name": info["model_name"],
                "cycle_N_c": n_c,
                "soh_pct": s,
                "N_F_hat": n_f_hat,
                "RUL_cycles": rul,
                "RUL_lower_bound_cycles": rul_lower,
                "RUL_upper_bound_cycles": rul_upper,
            })

        pf_info = pettitt_results_battery.get(role, {})
        potential_failure = pf_info.get("potential_failure_detected", False)
        n_p = pf_info.get("cycle_at_change", np.nan)
        pf_interval = (n_f_hat - n_p) if (potential_failure and not np.isnan(n_f_hat) and not np.isnan(n_p)) else np.nan

        # Aviso metodológico: o README define P-F = N_F - N_P assumindo que a
        # Potencial Falha (N_P, mudança detectada nos RESÍDUOS) precede a Falha
        # Funcional (N_F_hat, cruzamento do limiar convencional de SoH). Como o
        # próprio README enfatiza ("Potencial Falha ≠ limiar arbitrário de SoH"),
        # essa ordem NÃO é garantida — aqui reportamos e sinalizamos explicitamente
        # quando N_P ocorre DEPOIS de N_F_hat, em vez de mascarar com um valor
        # negativo sem explicação.
        pf_interval_valid = bool(potential_failure and not np.isnan(pf_interval) and pf_interval >= 0)
        if potential_failure and not np.isnan(pf_interval) and pf_interval < 0:
            log.warning(
                f"Bateria {battery_id}: N_P ({n_p:.0f}) ocorre DEPOIS de N_F_hat ({n_f_hat:.0f}) — "
                f"o intervalo P-F convencional (N_F - N_P) resulta negativo e NÃO deve ser "
                f"interpretado como 'tempo de aviso antes da falha'. Isso indica que, para esta "
                f"bateria, o limiar convencional de 80% de SoH foi cruzado bem antes de qualquer "
                f"mudança estatisticamente detectável no padrão dos resíduos do modelo."
            )

        rul_at_last_lower = (n_f_hat_lower - cycle.max()) if not np.isnan(n_f_hat_lower) else np.nan
        rul_at_last_upper = (n_f_hat_upper - cycle.max()) if not np.isnan(n_f_hat_upper) else np.nan
        summary_rows.append({
            "battery_id": battery_id,
            "role": role,
            "best_model": info["model_name"],
            "SoH_failure_threshold_pct": config.SOH_FAILURE_THRESHOLD,
            "N_F_hat": n_f_hat,
            "N_F_estimation_method": method,
            "N_F_hat_lower_bound_95pct": n_f_hat_lower,
            "N_F_hat_upper_bound_95pct": n_f_hat_upper,
            "potential_failure_detected": potential_failure,
            "N_P_potential_failure_cycle": n_p,
            "PF_interval_cycles": pf_interval,
            "PF_interval_valid_ordering": pf_interval_valid,
            "last_observed_cycle": cycle.max(),
            "last_observed_soh_pct": soh[-1],
            "RUL_at_last_observed_cycle": (n_f_hat - cycle.max()) if not np.isnan(n_f_hat) else np.nan,
            "RUL_at_last_observed_cycle_lower_bound_95pct": rul_at_last_lower,
            "RUL_at_last_observed_cycle_upper_bound_95pct": rul_at_last_upper,
        })

        # --- Figura 13: análise de falha ---
        n_plot_candidates = [cycle.max() * 1.05]
        if not np.isnan(n_f_hat):
            n_plot_candidates.append(n_f_hat)
        if not np.isnan(n_f_hat_upper):
            n_plot_candidates.append(n_f_hat_upper * 1.02)
        n_plot_upper = max(n_plot_candidates)

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.scatter(cycle, soh, s=10, color="black", alpha=0.5, label="Observed SoH")
        n_plot = np.linspace(cycle.min(), n_plot_upper, 300)
        y_plot = step05.predict_with_stored_model(model_info, n_plot)
        ax.plot(n_plot, y_plot, color="tab:blue", linewidth=1.3, label=f"Model ({info['model_name']})")
        ax.axhline(config.SOH_FAILURE_THRESHOLD, color="red", linestyle="--", linewidth=1,
                   label=f"Functional Failure ({config.SOH_FAILURE_THRESHOLD:.0f}%)")
        if not np.isnan(n_f_hat):
            ax.axvline(n_f_hat, color="red", linestyle=":", linewidth=1.3,
                       label=f"N_F_hat = {n_f_hat:.0f}")
        if not np.isnan(n_f_hat_lower) and not np.isnan(n_f_hat_upper):
            ax.axvspan(n_f_hat_lower, n_f_hat_upper, color="red", alpha=0.12,
                       label=f"95% N_F_hat range [{n_f_hat_lower:.0f}, {n_f_hat_upper:.0f}]")
        if potential_failure and not np.isnan(n_p):
            ax.axvline(n_p, color="orange", linestyle=":", linewidth=1.3,
                       label=f"N_P (Potential Failure) = {n_p:.0f}")
        ax.set_xlabel("Cycle")
        ax.set_ylabel("SoH (%)")
        model_suffix_title = "" if role == "selected" else f" (comparison model: {info['model_name']})"
        ax.set_title(f"Failure Analysis — {battery_id}{model_suffix_title}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        battery_suffix = "" if len(best_models) == 1 else f"_{battery_id}"
        role_suffix = "" if role == "selected" else f"_comparison_{info['model_name']}"
        fig.savefig(config.FIGURES_DIR / f"13_failure_analysis{battery_suffix}{role_suffix}.png",
                    dpi=config.FIGURE_DPI)
        plt.close(fig)

        # --- Figura 14: RUL vs ciclo (Adendo 4: com faixa de incerteza de 95%) ---
        rul_series = n_f_hat - cycle if not np.isnan(n_f_hat) else np.full_like(cycle, np.nan)
        rul_lower_series = (n_f_hat_lower - cycle) if not np.isnan(n_f_hat_lower) else np.full_like(cycle, np.nan)
        rul_upper_series = (n_f_hat_upper - cycle) if not np.isnan(n_f_hat_upper) else np.full_like(cycle, np.nan)
        fig, ax = plt.subplots(figsize=(9, 6))
        if not np.isnan(n_f_hat_lower) and not np.isnan(n_f_hat_upper):
            ax.fill_between(cycle, rul_lower_series, rul_upper_series, color="tab:green", alpha=0.15,
                            label="95% RUL prediction range")
        ax.plot(cycle, rul_series, color="tab:green", linewidth=1.3, marker=".", markersize=3,
                label="RUL point estimate")
        ax.axhline(0, color="red", linestyle="--", linewidth=1)
        ax.set_xlabel("Current cycle (N_c)")
        ax.set_ylabel("Estimated RUL (cycles)")
        ax.set_title(f"Remaining Useful Life (RUL) vs. Cycle — {battery_id}{model_suffix_title}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(config.FIGURES_DIR / f"14_rul_vs_cycle{battery_suffix}{role_suffix}.png", dpi=config.FIGURE_DPI)
        plt.close(fig)

    pd.DataFrame(rul_rows).to_csv(config.TABLES_DIR / "processed_data_RUL_detail.csv", index=False)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(config.TABLES_DIR / "Table_10_Failure_and_RUL_Analysis.csv", index=False)

    log.info(f"Tabela salva em: {config.TABLES_DIR / 'Table_10_Failure_and_RUL_Analysis.csv'}")
    log.info("Passo 10 concluído.")
    return summary_df


if __name__ == "__main__":
    main()

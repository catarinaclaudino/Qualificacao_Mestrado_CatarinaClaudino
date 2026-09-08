"""
04_exploratory_analysis.py
============================
PASSO 5: Análise exploratória de degradação.

- Estatísticas descritivas (média, desvio-padrão, mín, máx, mediana)
- Taxa de degradação (variação de SoH ciclo-a-ciclo)
- Avaliação de monotonicidade (correlação de Spearman com o ciclo e
  percentual de passos não-crescentes/decrescentes)
- Visualizações exploratórias

Saída:
    tables/Table_02_Descriptive_Statistics.csv
    tables/Table_03_Degradation_Rates.csv
    tables/Table_04_Monotonicity_Assessment.csv
    figures/03_degradation_rate_variation.png
    figures/04_soh_distribution.png

Pode ser executado isoladamente (requer 03_soh_calculation.py):
    python 04_exploratory_analysis.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import config
import utils

log = utils.get_logger("04_exploratory_analysis")


def main():
    log.info("=" * 70)
    log.info("PASSO 5: ANÁLISE EXPLORATÓRIA DE DEGRADAÇÃO")
    log.info("=" * 70)

    in_path = config.PROCESSED_DATA_DIR / "soh_data.csv"
    if not in_path.exists():
        log.error(f"Arquivo não encontrado: {in_path}. Rode antes: python 03_soh_calculation.py")
        raise SystemExit(1)

    df = pd.read_csv(in_path)

    desc_rows, rate_rows, mono_rows = [], [], []
    fig_rate, ax_rate = plt.subplots(figsize=(9, 6))
    fig_dist, ax_dist = plt.subplots(figsize=(9, 6))

    for battery_id, bdf in df.groupby("battery_id", sort=True):
        bdf = bdf.sort_values("cycle").reset_index(drop=True)
        soh = bdf["soh"].to_numpy()
        cycle = bdf["cycle"].to_numpy()

        # --- Estatísticas descritivas ---
        desc_rows.append({
            "battery_id": battery_id,
            "n": len(soh),
            "mean_soh": np.mean(soh),
            "std_soh": np.std(soh, ddof=1),
            "min_soh": np.min(soh),
            "max_soh": np.max(soh),
            "median_soh": np.median(soh),
        })

        # --- Taxa de degradação ciclo-a-ciclo ---
        d_soh = np.diff(soh)
        d_cycle = np.diff(cycle)
        rate = d_soh / np.where(d_cycle == 0, np.nan, d_cycle)  # %/ciclo
        rate_rows.append({
            "battery_id": battery_id,
            "mean_rate_pct_per_cycle": np.nanmean(rate),
            "std_rate_pct_per_cycle": np.nanstd(rate, ddof=1),
            "min_rate_pct_per_cycle": np.nanmin(rate),
            "max_rate_pct_per_cycle": np.nanmax(rate),
            "n_transitions": len(rate),
        })
        ax_rate.plot(cycle[1:], rate, marker=".", markersize=3, linewidth=0.7, label=battery_id)

        # --- Monotonicidade ---
        rho, p_rho = stats.spearmanr(cycle, soh)
        frac_nonincreasing = np.mean(d_soh <= 0)
        mono_rows.append({
            "battery_id": battery_id,
            "spearman_rho": rho,
            "spearman_p_value": p_rho,
            "fraction_nonincreasing_steps": frac_nonincreasing,
            "is_predominantly_monotonic_decreasing": bool(frac_nonincreasing >= 0.90 and rho < 0),
        })

        ax_dist.hist(soh, bins=30, alpha=0.5, label=battery_id)

    pd.DataFrame(desc_rows).to_csv(config.TABLES_DIR / "Table_02_Descriptive_Statistics.csv", index=False)
    pd.DataFrame(rate_rows).to_csv(config.TABLES_DIR / "Table_03_Degradation_Rates.csv", index=False)
    pd.DataFrame(mono_rows).to_csv(config.TABLES_DIR / "Table_04_Monotonicity_Assessment.csv", index=False)
    log.info("Tabelas 02, 03 e 04 salvas em: " + str(config.TABLES_DIR))

    ax_rate.axhline(0, color="black", linewidth=0.8)
    ax_rate.set_xlabel("Cycle")
    ax_rate.set_ylabel("SoH change per cycle (p.p./cycle)")
    ax_rate.set_title("Cycle-to-Cycle Degradation Rate")
    ax_rate.legend()
    ax_rate.grid(alpha=0.3)
    fig_rate.tight_layout()
    fig_rate.savefig(config.FIGURES_DIR / "03_degradation_rate_variation.png", dpi=config.FIGURE_DPI)
    plt.close(fig_rate)

    ax_dist.set_xlabel("SoH (%)")
    ax_dist.set_ylabel("Frequency")
    ax_dist.set_title("Distribution of SoH Values")
    ax_dist.legend()
    ax_dist.grid(alpha=0.3)
    fig_dist.tight_layout()
    fig_dist.savefig(config.FIGURES_DIR / "04_soh_distribution.png", dpi=config.FIGURE_DPI)
    plt.close(fig_dist)

    log.info("Figuras 03 e 04 salvas.")
    log.info("Passo 5 concluído.")
    return pd.DataFrame(desc_rows), pd.DataFrame(rate_rows), pd.DataFrame(mono_rows)


if __name__ == "__main__":
    main()

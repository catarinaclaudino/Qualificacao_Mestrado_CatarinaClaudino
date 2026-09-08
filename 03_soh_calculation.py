"""
03_soh_calculation.py
=======================
PASSO 4: Cálculo do State of Health (SoH).

- SoH_i = (Q_i / Q_ref) x 100, com Q_ref = Q_1 (primeira capacidade válida)
- Cada bateria é normalizada INDEPENDENTEMENTE (nunca pela média da população)
- Gera as figuras 01 (capacidade vs ciclo) e 02 (SoH vs ciclo)
- Gera Table_01_SoH_Summary.csv

Saída:
    processed_data/soh_data.csv
    tables/Table_01_SoH_Summary.csv
    figures/01_discharge_capacity_vs_cycle.png
    figures/02_soh_vs_cycle.png

Pode ser executado isoladamente (requer 02_data_processing.py):
    python 03_soh_calculation.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config
import utils

log = utils.get_logger("03_soh_calculation")


def main():
    log.info("=" * 70)
    log.info("PASSO 4: CÁLCULO DO STATE OF HEALTH (SoH)")
    log.info("=" * 70)

    in_path = config.PROCESSED_DATA_DIR / "processed_data.csv"
    if not in_path.exists():
        log.error(f"Arquivo não encontrado: {in_path}. Rode antes: python 02_data_processing.py")
        raise SystemExit(1)

    df = pd.read_csv(in_path)
    log.info(f"Dataset padronizado carregado: {len(df)} linhas, "
              f"{df['battery_id'].nunique()} bateria(s)")

    summary_rows = []
    soh_frames = []

    fig_cap, ax_cap = plt.subplots(figsize=(9, 6))
    fig_soh, ax_soh = plt.subplots(figsize=(9, 6))

    for battery_id, bdf in df.groupby("battery_id", sort=True):
        bdf = bdf.sort_values("cycle").reset_index(drop=True)
        q_ref = bdf["discharge_capacity"].iloc[0]
        bdf["soh"] = bdf["discharge_capacity"] / q_ref * 100.0
        soh_frames.append(bdf)

        ax_cap.plot(bdf["cycle"], bdf["discharge_capacity"], marker=".", markersize=3,
                    linewidth=0.8, label=battery_id)
        ax_soh.plot(bdf["cycle"], bdf["soh"], marker=".", markersize=3,
                    linewidth=0.8, label=battery_id)

        summary_rows.append({
            "battery_id": battery_id,
            "n_cycles": len(bdf),
            "Q_ref_Ah": q_ref,
            "Q_final_Ah": bdf["discharge_capacity"].iloc[-1],
            "SoH_final_pct": bdf["soh"].iloc[-1],
            "SoH_min_pct": bdf["soh"].min(),
            "SoH_max_pct": bdf["soh"].max(),
            "cycle_first": bdf["cycle"].iloc[0],
            "cycle_last": bdf["cycle"].iloc[-1],
        })
        log.info(f"Bateria {battery_id}: Q_ref={q_ref:.4f} Ah | "
                  f"SoH final={bdf['soh'].iloc[-1]:.2f}% após {len(bdf)} ciclos")

    soh_data = pd.concat(soh_frames, ignore_index=True)
    out_path = config.PROCESSED_DATA_DIR / "soh_data.csv"
    soh_data.to_csv(out_path, index=False)
    log.info(f"Dados de SoH salvos em: {out_path}")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = config.TABLES_DIR / "Table_01_SoH_Summary.csv"
    summary_df.to_csv(summary_path, index=False)
    log.info(f"Tabela resumo salva em: {summary_path}")

    ax_cap.set_xlabel("Cycle")
    ax_cap.set_ylabel("Discharge capacity (Ah)")
    ax_cap.set_title("Discharge Capacity vs. Cycle")
    ax_cap.legend()
    ax_cap.grid(alpha=0.3)
    fig_cap.tight_layout()
    fig_cap.savefig(config.FIGURES_DIR / "01_discharge_capacity_vs_cycle.png", dpi=config.FIGURE_DPI)
    plt.close(fig_cap)

    ax_soh.axhline(config.SOH_FAILURE_THRESHOLD, color="red", linestyle="--", linewidth=1,
                    label=f"Functional Failure threshold ({config.SOH_FAILURE_THRESHOLD:.0f}%)")
    ax_soh.set_xlabel("Cycle")
    ax_soh.set_ylabel("SoH (%)")
    ax_soh.set_title("State of Health (SoH) vs. Cycle")
    ax_soh.legend()
    ax_soh.grid(alpha=0.3)
    fig_soh.tight_layout()
    fig_soh.savefig(config.FIGURES_DIR / "02_soh_vs_cycle.png", dpi=config.FIGURE_DPI)
    plt.close(fig_soh)

    log.info(f"Figuras salvas em: {config.FIGURES_DIR}")
    log.info("Passo 4 concluído.")
    return soh_data


if __name__ == "__main__":
    main()

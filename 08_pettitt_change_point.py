"""
08_pettitt_change_point.py
============================
PASSO 9: Detecção de ponto de mudança de Pettitt (Potencial Falha).

- O teste de Pettitt (não-paramétrico) é aplicado sobre os RESÍDUOS do
  melhor modelo (Passo 7), NÃO sobre o SoH bruto
- p < ALPHA indica um ponto de mudança estatisticamente significativo
- Critério de PERSISTÊNCIA: a mudança detectada deve persistir por
  PERSISTENCE_WINDOW observações subsequentes
- "Potencial Falha" só é declarada se AMBOS os critérios forem satisfeitos
  (significância estatística E persistência). Se qualquer um falhar,
  nenhuma Potencial Falha é reportada.

Importante (ver README):
    Potencial Falha != início físico da degradação
    Potencial Falha != limiar arbitrário de SoH
    Potencial Falha  = mudança detectável no padrão de degradação

Desde 2026-09-07, `best_models.pkl` guarda, por bateria, um dict de
"roles" ("selected" e, opcionalmente, "comparison" - ver
config.COMPARISON_MODEL_NAME, pedido EXPLÍCITO da pesquisadora). Este
script roda o MESMO teste de Pettitt para CADA role presente.

Saída:
    tables/Table_09_Pettitt_Change_Point_Detection.csv (agora com coluna "role")
    figures/12_pettitt_change_point_detection.png (+ _comparison_<Modelo>.png se aplicável)

Pode ser executado isoladamente (requer 06_residual_analysis_and_validation.py):
    python 08_pettitt_change_point.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import utils

log = utils.get_logger("08_pettitt_change_point")


def main():
    log.info("=" * 70)
    log.info("PASSO 9: DETECÇÃO DE PONTO DE MUDANÇA DE PETTITT")
    log.info("=" * 70)

    in_path = config.PROCESSED_DATA_DIR / "best_models.pkl"
    if not in_path.exists():
        log.error(f"Arquivo não encontrado: {in_path}. "
                  f"Rode antes: python 06_residual_analysis_and_validation.py")
        raise SystemExit(1)

    best_models = utils.load_pickle(in_path)
    results_rows = []
    pettitt_results = {}

    for battery_id, entries_by_role in best_models.items():
      pettitt_results[battery_id] = {}
      for role, info in entries_by_role.items():
        cycle = info["cycle"]
        residuals = info["residuals"]

        pettitt = utils.pettitt_test(residuals)
        tau_idx = pettitt["tau_index"]
        persists, agreement = utils.check_persistence(residuals, tau_idx, config.PERSISTENCE_WINDOW)

        is_significant = pettitt["p_value"] < config.ALPHA
        potential_failure_detected = bool(is_significant and persists)
        cycle_at_change = float(cycle[tau_idx]) if tau_idx is not None else np.nan

        results_rows.append({
            "battery_id": battery_id,
            "role": role,
            "best_model": info["model_name"],
            "pettitt_K_statistic": pettitt["K"],
            "p_value": pettitt["p_value"],
            "is_statistically_significant_alpha_0.05": bool(is_significant),
            "change_point_cycle_N_P": cycle_at_change,
            "persistence_agreement_fraction": agreement,
            "persistence_criterion_met": bool(persists),
            "potential_failure_detected": potential_failure_detected,
        })

        pettitt_results[battery_id][role] = {
            "model_name": info["model_name"],
            "tau_index": tau_idx,
            "cycle_at_change": cycle_at_change,
            "potential_failure_detected": potential_failure_detected,
            "p_value": pettitt["p_value"],
        }

        status = "DETECTADA" if potential_failure_detected else "NÃO detectada"
        log.info(f"Bateria {battery_id} [role={role}, modelo={info['model_name']}]: Falha Potencial {status} "
                  f"(K={pettitt['K']:.2f}, p={pettitt['p_value']:.4f}, "
                  f"N_P={cycle_at_change if not np.isnan(cycle_at_change) else 'N/A'}, "
                  f"persistência={'OK' if persists else 'falhou'})")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
        ax1.scatter(cycle, residuals, s=10, alpha=0.6, color="steelblue")
        ax1.axhline(0, color="gray", linestyle=":", linewidth=1)
        if tau_idx is not None:
            ax1.axvline(cycle[tau_idx], color="red", linestyle="--", linewidth=1.3,
                        label=f"Candidate change point (N={cycle[tau_idx]:.0f})")
        ax1.set_ylabel("Best model residual")
        status_suffix = "POTENTIAL FAILURE DETECTED" if potential_failure_detected else "no Potential Failure"
        model_suffix = "" if role == "selected" else f", comparison model: {info['model_name']}"
        ax1.set_title(f"Pettitt Test on Residuals — {battery_id}\n({status_suffix}{model_suffix})", fontsize=11)
        ax1.legend(fontsize=8)
        ax1.grid(alpha=0.3)

        if len(pettitt["U"]) > 0:
            ax2.plot(cycle[1:], pettitt["U"], color="black", linewidth=1.2)
            if tau_idx is not None and tau_idx < len(pettitt["U"]):
                ax2.axvline(cycle[tau_idx], color="red", linestyle="--", linewidth=1.3)
            ax2.axhline(0, color="gray", linestyle=":", linewidth=1)
        ax2.set_xlabel("Cycle")
        ax2.set_ylabel("Pettitt U_t statistic")
        ax2.grid(alpha=0.3)

        fig.tight_layout()
        battery_suffix = "" if len(best_models) == 1 else f"_{battery_id}"
        role_suffix = "" if role == "selected" else f"_comparison_{info['model_name']}"
        fig.savefig(config.FIGURES_DIR / f"12_pettitt_change_point_detection{battery_suffix}{role_suffix}.png",
                    dpi=config.FIGURE_DPI)
        plt.close(fig)

    pd.DataFrame(results_rows).to_csv(
        config.TABLES_DIR / "Table_09_Pettitt_Change_Point_Detection.csv", index=False)
    utils.save_pickle(pettitt_results, config.PROCESSED_DATA_DIR / "pettitt_results.pkl")

    log.info(f"Tabela salva em: {config.TABLES_DIR / 'Table_09_Pettitt_Change_Point_Detection.csv'}")
    log.info("Passo 9 concluído.")
    return pd.DataFrame(results_rows)


if __name__ == "__main__":
    main()

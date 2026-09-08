"""
06_residual_analysis_and_validation.py
========================================
PASSO 7: Selecao de modelo (hierarquica, treino-only) + validacao temporal.

Esta etapa foi reescrita para seguir EXATAMENTE a hierarquia de selecao de
modelos especificada.

Particao dos dados:
    Validacao temporal 70% treino / 30% teste, SEM embaralhamento, ordem
    cronologica preservada (config.TRAIN_FRACTION).

Hierarquia de selecao (NOVA - usa SOMENTE os dados de TREINO):
    Passo 1 - Ajustar cada um dos 5 modelos candidatos SOMENTE no treino.
    Passo 2 - DIAGNOSTICOS ESTATISTICOS (adequacao residual): teste de
              normalidade de Shapiro-Wilk sobre os residuos de TREINO
              (shapiro_pass = p > alpha). Este e um diagnostico formal
              PONTUAL sobre a normalidade dos residuos, REPORTADO na tabela
              de auditoria - NAO e tratado como teste de validade geral do
              modelo, NAO e combinado numericamente com outras metricas, e com n_treino~600
              o Shapiro-Wilk rejeita normalidade perfeita para TODOS os 5
              candidatos, travando toda a seleção - a pesquisadora autorizou
              manter Shapiro-Wilk como diagnóstico reportado, mas REMOVEU-O
              da condição de elegibilidade) NAO e mais usado como critério de
              elegibilidade (ver Passo 4).
    Passo 3 - PLAUSIBILIDADE FISICA: verifica se a trajetoria PREVISTA pelo
              modelo (ajustado no treino) e monotonicamente nao-crescente
              (dSoH/dN <= 0) sobre uma malha densa cobrindo o intervalo de
              ciclos efetivamente observado (treino+teste), com uma
              tolerancia numerica para ruido de ponto flutuante
              (config.MONOTONICITY_TOLERANCE). NAO impoe SoH>=0 ou SoH<=100.
    Passo 4 - ELEGIBILIDADE = plausibilidade fisica (monotonicity_pass)
              SOMENTE. Shapiro-Wilk (residual_adequacy_status) permanece
              reportado na tabela de auditoria para cada candidato, mas NAO
              entra no cálculo de elegibilidade, por decisão explícita da
              pesquisadora (ver nota no Passo 2).
    Passo 5 - SELECAO: entre os modelos elegiveis, escolhe o de menor BIC
              calculado EXCLUSIVAMENTE nos dados de treino (BIC_train).
    Passo 6 - DESEMPATE (somente se |BIC_a - BIC_b| < config.BIC_TIE_TOLERANCE
              entre o(s) candidato(s) mais proximo(s) do minimo): maior R2
              ajustado de treino -> menor RMSE de treino -> menor MAE de
              treino, NUNCA metricas de teste.
    Se NENHUM modelo for elegivel: o pipeline NAO seleciona nenhum modelo
    para aquela bateria (nao ha fallback silencioso) - o problema e
    reportado de forma explicita nos logs e na tabela de auditoria, e a
    bateria fica sem "melhor modelo" para os Passos 8-10.

VALIDACAO FINAL (teste/holdout) - SOMENTE PARA REPORTE, NUNCA PARA SELECAO:
    Depois de selecionado, o modelo escolhido e avaliado no conjunto de
    teste retido (R2, R2 ajustado, RMSE, MAE). Estas metricas SAO SAIDAS do
    processo, nunca ENTRADAS - o teste jamais participa da escolha do modelo.

Diagnosticos de residuos do melhor modelo (reajustado com TODOS os dados,
para uso downstream nos Passos 8/9) - PRESERVADOS SEM ALTERACAO em relacao a
versao anterior: residuos vs. ciclo, residuos vs. valores ajustados,
histograma, Q-Q plot, e o teste de Shapiro-Wilk sobre esses residuos
full-fit (Table_06 - um diagnostico DIFERENTE e PRE-EXISTENTE, distinto do
screening de Shapiro por-candidato feito SOMENTE no treino, acima).

MODELO DE COMPARACAO (config.COMPARISON_MODEL_NAME):
    Alem do modelo formalmente SELECIONADO pela hierarquia acima, se
    config.COMPARISON_MODEL_NAME estiver definido (e for diferente do
    selecionado), o pipeline reajusta ESSE modelo tambem com TODOS os dados
    e roda o MESMO conjunto de diagnosticos/figuras (06-10) e, downstream,
    os Passos 8/9/10 (07,08,09_*.py) TAMBEM para ele - gerando um segundo
    conjunto PARALELO e COMPLETO de saidas, rotulado role="comparison" nas
    tabelas e com sufixo "_comparison_<NomeDoModelo>" nos nomes de arquivo.
    Isto e uma analise de SENSIBILIDADE - NAO
    altera o resultado da selecao formal (Table_Model_Selection_Audit.csv
    continua mostrando "selected_model" exclusivamente conforme a hierarquia
    BIC). Ver docstring de config.COMPARISON_MODEL_NAME para o racional
    completo.

Saida:
    tables/Table_Model_Selection_Audit.csv   (NOVA - trilha de auditoria por
                                               candidato/bateria; ver secao
                                               "AUDITORIA DE SELECAO" abaixo)
    tables/Table_06_Shapiro_Wilk_Normality_Test.csv  (pre-existente, full-fit;
                                               agora com coluna "role" -
                                               "selected" e, se aplicavel,
                                               "comparison")
    tables/Table_07_Temporal_Validation_Metrics.csv  (metricas de teste -
                                               reporte final independente,
                                               NAO usado para selecao; agora
                                               com coluna "role")
    processed_data/best_models.pkl   (por bateria: dict {"selected": {...},
                                               "comparison": {...} (opcional)})
    figures/06_residuals_vs_cycle.png (+ _comparison_<Modelo>.png se aplicavel)
    figures/07_residuals_vs_fitted.png (+ _comparison_<Modelo>.png)
    figures/08_residuals_histogram.png (+ _comparison_<Modelo>.png)
    figures/09_residuals_qq_plot.png (+ _comparison_<Modelo>.png)
    figures/10_temporal_validation.png (+ _comparison_<Modelo>.png)

Pode ser executado isoladamente (requer 05_degradation_models.py):
    python 06_residual_analysis_and_validation.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import config
import utils
import importlib.util
import sys


def _import_step05():
    from pathlib import Path
    base_dir = Path(config.__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("step05", base_dir / "05_degradation_models.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["step05"] = mod
    spec.loader.exec_module(mod)
    return mod


log = utils.get_logger("06_residual_analysis_and_validation")
step05 = _import_step05()


def _shapiro(residuals):
    """Teste de Shapiro-Wilk. Retorna (statistic, p_value); NaN se n < 3."""
    residuals = np.asarray(residuals, dtype=float)
    if len(residuals) < 3:
        return np.nan, np.nan
    sw_n = min(len(residuals), 5000)
    stat, p_value = stats.shapiro(residuals[:sw_n])
    return float(stat), float(p_value)


def _process_model_for_diagnostics(
    role, model_name, battery_id, model_info_by_name, audit_rows_battery,
    n_arr, y_arr, n_train_arr, y_train_arr, n_test_arr, y_test_arr,
    validation_rows, shapiro_rows, suffix,
):
    """
    Processa os diagnosticos completos de UM modelo (role="selected" ou
    role="comparison") para UMA bateria: metricas de teste/holdout (reporte
    final, NAO usadas para selecao), reajuste com TODOS os dados, residuos,
    Shapiro-Wilk sobre o full-fit (Table_06) e as 4 figuras de diagnostico de
    residuos (06-09). Usado tanto para o modelo formalmente SELECIONADO pela
    hierarquia do Passo 7 quanto para o modelo de COMPARACAO opcional
    (config.COMPARISON_MODEL_NAME) de modo que ambos recebam EXATAMENTE o mesmo tratamento.

    Retorna a entrada a ser guardada em best_models[battery_id][role].
    """
    n_params_train = [r["n_parameters"] for r in audit_rows_battery if r["model_name"] == model_name][0]

    # VALIDACAO FINAL (teste/holdout) - SOMENTE REPORTE, NAO influencia a
    # selecao (que ja ocorreu, exclusivamente com dados de treino, no Passo 7).
    train_model_info = model_info_by_name[model_name]
    if len(n_test_arr) > 0:
        y_pred_test = step05.predict_with_stored_model(train_model_info, n_test_arr)
        test_metrics = utils.regression_metrics(y_test_arr, y_pred_test, n_params_train)
    else:
        test_metrics = {"r2": np.nan, "adj_r2": np.nan, "rmse": np.nan, "mae": np.nan}

    train_row = [r for r in audit_rows_battery if r["model_name"] == model_name][0]
    validation_rows.append({
        "battery_id": battery_id,
        "role": role,
        "selected_model": model_name,
        "n_train": len(n_train_arr),
        "n_test": len(n_test_arr),
        "train_r2": train_row["R2_train"],
        "train_rmse": train_row["RMSE_train"],
        "test_r2": test_metrics["r2"],
        "test_adjusted_r2": test_metrics["adj_r2"],
        "test_rmse": test_metrics["rmse"],
        "test_mae": test_metrics["mae"],
    })

    # reajusta o modelo com TODOS os dados (fit final para diagnostico/uso
    # posterior nos Passos 8 e 9) - logica INALTERADA em relacao a versao
    # anterior, agora aplicada de forma identica a qualquer modelo processado.
    full_model_info, y_pred_full, n_params_full = step05.fit_one_model(model_name, n_arr, y_arr)
    residuals = y_arr - y_pred_full
    full_metrics = utils.regression_metrics(y_arr, y_pred_full, n_params_full)

    entry = {
        "model_name": model_name,
        "role": role,
        "model_info": full_model_info,
        "n_params": n_params_full,
        "cycle": n_arr,
        "soh": y_arr,
        "y_pred": y_pred_full,
        "residuals": residuals,
        "resid_std": full_metrics["resid_std"],
        "full_fit_metrics": full_metrics,
    }

    # --- Shapiro-Wilk sobre os residuos do full-fit (Table_06) ---
    stat, p_value = _shapiro(residuals)
    shapiro_rows.append({
        "battery_id": battery_id,
        "role": role,
        "model_name": model_name,
        "shapiro_statistic": stat,
        "p_value": p_value,
        "reject_normality_at_alpha_0.05": bool(p_value < config.ALPHA) if not np.isnan(p_value) else None,
    })

    # --- Figuras de diagnostico de residuos ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(n_arr, residuals, s=10, alpha=0.6)
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Residual (observed SoH - predicted SoH)")
    ax.set_title(f"Residuals vs. Cycle \u2014 {battery_id} (model: {model_name})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / f"06_residuals_vs_cycle{suffix}.png", dpi=config.FIGURE_DPI)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_pred_full, residuals, s=10, alpha=0.6)
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Fitted value (predicted SoH, %)")
    ax.set_ylabel("Residual")
    ax.set_title(f"Residuals vs. Fitted Values \u2014 {battery_id} (model: {model_name})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / f"07_residuals_vs_fitted{suffix}.png", dpi=config.FIGURE_DPI)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(residuals, bins=30, color="steelblue", alpha=0.8)
    ax.set_xlabel("Residual")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Residuals Histogram \u2014 {battery_id} (model: {model_name})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / f"08_residuals_histogram{suffix}.png", dpi=config.FIGURE_DPI)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title(f"Residuals Q-Q Plot \u2014 {battery_id} (model: {model_name})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / f"09_residuals_qq_plot{suffix}.png", dpi=config.FIGURE_DPI)
    plt.close(fig)

    return entry


def _select_model(audit_rows_battery):
    """
    Aplica a hierarquia de selecao (Passos 5-6) sobre as linhas de auditoria
    (uma por modelo candidato) de UMA bateria.

    Retorna (selected_name_or_None, tie_break_used: bool, ranked_names: list)
    onde ranked_names e a lista de nomes elegiveis ordenada por BIC_train
    crescente (usada para atribuir selection_rank).
    """
    eligible = [r for r in audit_rows_battery if r["eligible_for_selection"]]
    if not eligible:
        return None, False, []

    eligible_sorted_by_bic = sorted(eligible, key=lambda r: r["BIC_train"])
    ranked_names = [r["model_name"] for r in eligible_sorted_by_bic]

    min_bic = eligible_sorted_by_bic[0]["BIC_train"]
    tied = [r for r in eligible_sorted_by_bic if (r["BIC_train"] - min_bic) < config.BIC_TIE_TOLERANCE]

    if len(tied) == 1:
        return tied[0]["model_name"], False, ranked_names

    # Desempate (Passo 6): maior R2_ajustado_train -> menor RMSE_train -> menor MAE_train
    tied_sorted = sorted(tied, key=lambda r: (-r["R2_adjusted_train"], r["RMSE_train"], r["MAE_train"]))
    return tied_sorted[0]["model_name"], True, ranked_names


def main():
    log.info("=" * 70)
    log.info("PASSO 7: SELECAO HIERARQUICA DE MODELO (TREINO-ONLY) + VALIDACAO TEMPORAL")
    log.info("=" * 70)

    in_path = config.PROCESSED_DATA_DIR / "soh_data.csv"
    if not in_path.exists():
        log.error(f"Arquivo n\u00e3o encontrado: {in_path}. Rode antes: python 03_soh_calculation.py")
        raise SystemExit(1)

    df = pd.read_csv(in_path)

    audit_rows_all = []       # tabela nova: Table_Model_Selection_Audit.csv
    validation_rows = []      # Table_07 (reporte final no teste; agora com coluna "role")
    shapiro_rows = []         # Table_06 (full-fit; agora com coluna "role")
    best_models = {}
    batteries_without_eligible_model = []

    # Modelo de comparacao (config.COMPARISON_MODEL_NAME) - NAO uma regra automatica deste codigo.
    # Ver docstring do modulo e de config.COMPARISON_MODEL_NAME.
    comparison_model_name = config.COMPARISON_MODEL_NAME
    any_comparison_processed = False

    n_batteries = df["battery_id"].nunique()
    fig_val, axes_val = plt.subplots(n_batteries, 1, figsize=(9, 5 * n_batteries), squeeze=False)
    fig_val_cmp, axes_val_cmp = plt.subplots(n_batteries, 1, figsize=(9, 5 * n_batteries), squeeze=False)

    for b_idx, (battery_id, bdf) in enumerate(df.groupby("battery_id", sort=True)):
        bdf = bdf.sort_values("cycle").reset_index(drop=True)
        n_arr = bdf["cycle"].to_numpy(dtype=float)
        y_arr = bdf["soh"].to_numpy(dtype=float)
        n_total = len(bdf)
        n_train = int(round(n_total * config.TRAIN_FRACTION))
        n_train = max(min(n_train, n_total - 2), 2)  # garante >=2 pontos em cada partic\u00e3o

        n_train_arr, n_test_arr = n_arr[:n_train], n_arr[n_train:]
        y_train_arr, y_test_arr = y_arr[:n_train], y_arr[n_train:]

        # malha densa para o teste de monotonicidade (Passo 3), cobrindo o
        # intervalo de ciclos EFETIVAMENTE OBSERVADO da bateria (treino+teste)
        mono_grid = np.linspace(n_arr.min(), n_arr.max(), config.MONOTONICITY_GRID_N_POINTS)

        audit_rows_battery = []
        model_info_by_name = {}

        for name in step05.MODEL_NAMES:
            # ------------------------------------------------------------------
            # PASSO 1 - AJUSTE (SOMENTE TREINO)
            # ------------------------------------------------------------------
            model_info, y_pred_train, n_params = step05.fit_one_model(name, n_train_arr, y_train_arr)

            if model_info is None:
                audit_rows_battery.append({
                    "battery_id": battery_id, "model_name": name,
                    "n_parameters": np.nan, "n_observations": len(n_train_arr),
                    "RSS_train": np.nan, "R2_train": np.nan, "R2_adjusted_train": np.nan,
                    "RMSE_train": np.nan, "MAE_train": np.nan, "BIC_train": np.nan,
                    "shapiro_statistic": np.nan, "shapiro_pvalue": np.nan, "shapiro_pass": False,
                    "residual_adequacy_status": "FAIL (fit failed)",
                    "monotonicity_pass": False, "max_increase_pct_soh": np.nan,
                    "physical_plausibility_status": "FAIL (fit failed)",
                    "eligible_for_selection": False, "fit_status": "failed",
                })
                continue

            model_info_by_name[name] = model_info
            train_metrics = utils.regression_metrics(y_train_arr, y_pred_train, n_params)

            # ------------------------------------------------------------------
            # PASSO 2 - DIAGNOSTICOS ESTATISTICOS (adequa\u00e7\u00e3o residual: Shapiro-Wilk
            # sobre os res\u00edduos de TREINO). Diagn\u00f3stico formal pontual, N\u00c3O um score
            # de valida\u00e7\u00e3o geral do modelo.
            # ------------------------------------------------------------------
            residuals_train = y_train_arr - y_pred_train
            shapiro_stat, shapiro_p = _shapiro(residuals_train)
            shapiro_pass = bool(shapiro_p > config.ALPHA) if not np.isnan(shapiro_p) else False
            residual_adequacy_status = "PASS" if shapiro_pass else "FAIL"
            # NOTA: residual_adequacy_status/shapiro_pass é REPORTADO abaixo mas,
            # por decisão explícita da pesquisadora, NAO participa do cálculo de
            # `eligible` (Passo 4) - ver docstring do módulo.

            # ------------------------------------------------------------------
            # PASSO 3 - PLAUSIBILIDADE FÍSICA (monotonicidade: dSoH/dN <= 0 sobre
            # malha densa, com tolerancia numérica para ruído de ponto flutuante).
            # NÃO impomos SoH>=0 nem SoH<=100 (não especificado pela pesquisadora).
            # ------------------------------------------------------------------
            y_mono_grid = step05.predict_with_stored_model(model_info, mono_grid)
            monotonicity_pass, max_increase = utils.check_monotonicity(y_mono_grid, config.MONOTONICITY_TOLERANCE)
            physical_plausibility_status = "PASS" if monotonicity_pass else "FAIL"

            # ------------------------------------------------------------------
            # PASSO 4 - ELEGIBILIDADE = SOMENTE plausibilidade física
            # (monotonicity_pass). Shapiro-Wilk (shapiro_pass) NAO entra aqui -
            # decisão explícita da pesquisadora (2026-09-07), pois com
            # n_treino~600 ele reprovava os 5 candidatos e travava toda a
            # seleção. Permanece reportado como diagnóstico na tabela.
            # ------------------------------------------------------------------
            eligible = monotonicity_pass

            bic_train = utils.compute_bic(train_metrics["rss"], train_metrics["n"], n_params)

            audit_rows_battery.append({
                "battery_id": battery_id, "model_name": name,
                "n_parameters": n_params, "n_observations": train_metrics["n"],
                "RSS_train": train_metrics["rss"], "R2_train": train_metrics["r2"],
                "R2_adjusted_train": train_metrics["adj_r2"], "RMSE_train": train_metrics["rmse"],
                "MAE_train": train_metrics["mae"], "BIC_train": bic_train,
                "shapiro_statistic": shapiro_stat, "shapiro_pvalue": shapiro_p, "shapiro_pass": shapiro_pass,
                "residual_adequacy_status": residual_adequacy_status,
                "monotonicity_pass": monotonicity_pass, "max_increase_pct_soh": max_increase,
                "physical_plausibility_status": physical_plausibility_status,
                "eligible_for_selection": eligible, "fit_status": "success",
            })

        # ------------------------------------------------------------------
        # PASSO 5-6 - SELECAO (menor BIC_train entre eleg\u00edveis; desempate por
        # R2_ajustado_train / RMSE_train / MAE_train) - NUNCA metricas de teste.
        # ------------------------------------------------------------------
        best_name, tie_break_used, ranked_names = _select_model(audit_rows_battery)

        rank_map = {name: i + 1 for i, name in enumerate(ranked_names)}
        for row in audit_rows_battery:
            row["selection_rank"] = rank_map.get(row["model_name"], np.nan)
            row["selected_model"] = (row["model_name"] == best_name) if best_name is not None else False
            row["tie_break_used_for_this_battery"] = tie_break_used
        audit_rows_all.extend(audit_rows_battery)

        if best_name is None:
            log.error(
                f"Bateria {battery_id}: NENHUM modelo candidato passou no critério de "
                f"elegíbilidade (plausibilidade física via monotonicidade; Shapiro-Wilk é "
                f"apenas diagnóstico reportado, não entra na elegíbilidade). Nenhum modelo "
                f"será selecionado para esta bateria - NAO há fallback automático. Motivo por candidato:"
            )
            for row in audit_rows_battery:
                log.error(
                    f"    {row['model_name']}: residual_adequacy={row['residual_adequacy_status']} "
                    f"(shapiro p={row['shapiro_pvalue']}), physical_plausibility="
                    f"{row['physical_plausibility_status']} (max_increase={row['max_increase_pct_soh']})"
                )
            batteries_without_eligible_model.append(battery_id)
            continue

        log.info(
            f"Bateria {battery_id}: modelo selecionado = '{best_name}' "
            f"(BIC_train={dict((r['model_name'], r['BIC_train']) for r in audit_rows_battery)[best_name]:.2f}"
            f"{', desempate aplicado' if tie_break_used else ''})"
        )

        # ------------------------------------------------------------------
        # DIAGNOSTICO COMPLETO DO MODELO SELECIONADO (mesma logica de antes,
        # agora fatorada em _process_model_for_diagnostics) - nomes de
        # arquivo/figuras INALTERADOS em relacao a versao anterior.
        # ------------------------------------------------------------------
        battery_suffix = "" if n_batteries == 1 else f"_{battery_id}"
        order = np.argsort(n_arr)

        selected_entry = _process_model_for_diagnostics(
            role="selected", model_name=best_name, battery_id=battery_id,
            model_info_by_name=model_info_by_name, audit_rows_battery=audit_rows_battery,
            n_arr=n_arr, y_arr=y_arr, n_train_arr=n_train_arr, y_train_arr=y_train_arr,
            n_test_arr=n_test_arr, y_test_arr=y_test_arr,
            validation_rows=validation_rows, shapiro_rows=shapiro_rows,
            suffix=battery_suffix,
        )
        entries_for_battery = {"selected": selected_entry}

        # --- Figura combinada de valida\u00e7\u00e3o temporal (treino/teste) - modelo SELECIONADO ---
        ax_val = axes_val[b_idx, 0]
        ax_val.scatter(n_train_arr, y_train_arr, s=10, color="tab:blue", label="Training (70%)")
        ax_val.scatter(n_test_arr, y_test_arr, s=10, color="tab:orange", label="Test (30%, held out)")
        ax_val.plot(n_arr[order], selected_entry["y_pred"][order], color="black", linewidth=1.3,
                    label=f"Fitted model ({best_name})")
        ax_val.axvline(n_train_arr[-1], color="gray", linestyle=":", linewidth=1)
        ax_val.set_title(f"Temporal Validation \u2014 {battery_id}")
        ax_val.set_xlabel("Cycle")
        ax_val.set_ylabel("SoH (%)")
        ax_val.legend()
        ax_val.grid(alpha=0.3)

        # ------------------------------------------------------------------
        # MODELO DE COMPARACAO (config.COMPARISON_MODEL_NAME) -
        # NAO participa da selecao formal - apenas recebe o MESMO tratamento
        # de diagnostico/downstream do modelo selecionado, para comparação.
        # ------------------------------------------------------------------
        if comparison_model_name and comparison_model_name != best_name:
            if comparison_model_name in model_info_by_name:
                comparison_suffix = (
                    f"_comparison_{comparison_model_name}" if n_batteries == 1
                    else f"_{battery_id}_comparison_{comparison_model_name}"
                )
                comparison_entry = _process_model_for_diagnostics(
                    role="comparison", model_name=comparison_model_name, battery_id=battery_id,
                    model_info_by_name=model_info_by_name, audit_rows_battery=audit_rows_battery,
                    n_arr=n_arr, y_arr=y_arr, n_train_arr=n_train_arr, y_train_arr=y_train_arr,
                    n_test_arr=n_test_arr, y_test_arr=y_test_arr,
                    validation_rows=validation_rows, shapiro_rows=shapiro_rows,
                    suffix=comparison_suffix,
                )
                entries_for_battery["comparison"] = comparison_entry
                any_comparison_processed = True

                ax_val_cmp = axes_val_cmp[b_idx, 0]
                ax_val_cmp.scatter(n_train_arr, y_train_arr, s=10, color="tab:blue", label="Training (70%)")
                ax_val_cmp.scatter(n_test_arr, y_test_arr, s=10, color="tab:orange", label="Test (30%, held out)")
                ax_val_cmp.plot(n_arr[order], comparison_entry["y_pred"][order], color="black", linewidth=1.3,
                                label=f"Fitted model ({comparison_model_name})")
                ax_val_cmp.axvline(n_train_arr[-1], color="gray", linestyle=":", linewidth=1)
                ax_val_cmp.set_title(f"Temporal Validation (comparison model: {comparison_model_name}) \u2014 {battery_id}")
                ax_val_cmp.set_xlabel("Cycle")
                ax_val_cmp.set_ylabel("SoH (%)")
                ax_val_cmp.legend()
                ax_val_cmp.grid(alpha=0.3)

                log.info(
                    f"Bateria {battery_id}: modelo de comparação '{comparison_model_name}' "
                    f"processado (Passos 7-10) em paralelo ao modelo selecionado '{best_name}', "
                    f"conforme pedido explícito da pesquisadora - rotulado role='comparison' "
                    f"nas tabelas e com sufixo '{comparison_suffix}' nos arquivos."
                )
            else:
                log.warning(
                    f"Bateria {battery_id}: modelo de comparação configurado "
                    f"('{comparison_model_name}') não pôde ser ajustado nos dados de "
                    f"treino desta bateria (fit_status='failed' na tabela de auditoria); "
                    f"análise de comparação pulada para esta bateria - reportado, sem fallback."
                )
        elif comparison_model_name and comparison_model_name == best_name:
            log.info(
                f"Bateria {battery_id}: modelo de comparação configurado "
                f"('{comparison_model_name}') coincide com o modelo selecionado; nenhuma "
                f"análise duplicada necessária."
            )

        best_models[battery_id] = entries_for_battery

    fig_val.tight_layout()
    fig_val.savefig(config.FIGURES_DIR / "10_temporal_validation.png", dpi=config.FIGURE_DPI)
    plt.close(fig_val)

    if any_comparison_processed:
        fig_val_cmp.tight_layout()
        fig_val_cmp.savefig(
            config.FIGURES_DIR / f"10_temporal_validation_comparison_{comparison_model_name}.png",
            dpi=config.FIGURE_DPI,
        )
    plt.close(fig_val_cmp)

    pd.DataFrame(audit_rows_all).to_csv(config.TABLES_DIR / "Table_Model_Selection_Audit.csv", index=False)
    pd.DataFrame(validation_rows).to_csv(config.TABLES_DIR / "Table_07_Temporal_Validation_Metrics.csv", index=False)
    pd.DataFrame(shapiro_rows).to_csv(config.TABLES_DIR / "Table_06_Shapiro_Wilk_Normality_Test.csv", index=False)
    utils.save_pickle(best_models, config.PROCESSED_DATA_DIR / "best_models.pkl")

    log.info("Tabelas Table_Model_Selection_Audit, 06 e 07, e o arquivo best_models.pkl foram salvos.")

    if batteries_without_eligible_model:
        log.error(
            f"ATEN\u00c7\u00c3O: {len(batteries_without_eligible_model)} bateria(s) sem modelo eleg\u00edvel: "
            f"{batteries_without_eligible_model}. Consulte Table_Model_Selection_Audit.csv para o "
            f"motivo detalhado por candidato. Nenhum fallback foi aplicado; estas baterias N\u00c3O "
            f"aparecem em best_models.pkl e, portanto, N\u00c3O ser\u00e3o processadas pelos Passos 8, 9 e 10."
        )

    log.info("Passo 7 conclu\u00eddo.")
    return best_models


if __name__ == "__main__":
    main()

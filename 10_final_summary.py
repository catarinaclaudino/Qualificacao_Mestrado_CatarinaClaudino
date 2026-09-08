"""
10_final_summary.py
=====================
Gera um relatório resumo final consolidando as principais tabelas do pipeline.

Saída:
    <OUTPUT_DIR>/RELATORIO_FINAL.md

Pode ser executado isoladamente (idealmente após todos os passos anteriores):
    python 10_final_summary.py
"""

import pandas as pd

import config
import utils

log = utils.get_logger("10_final_summary")


def to_md(df):
    """df.to_markdown() com fallback caso o pacote 'tabulate' não esteja instalado."""
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return df.to_string(index=False)


def read_table(name):
    path = config.TABLES_DIR / name
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        # Arquivo existe mas está vazio (ex.: nenhuma bateria teve modelo
        # selecionado) - não é um erro do pipeline, apenas ausência de linhas.
        return None


def main():
    log.info("=" * 70)
    log.info("RESUMO FINAL")
    log.info("=" * 70)

    soh_summary = read_table("Table_01_SoH_Summary.csv")
    model_comp = read_table("Table_05_Model_Comparison.csv")
    selection_audit = read_table("Table_Model_Selection_Audit.csv")
    validation = read_table("Table_07_Temporal_Validation_Metrics.csv")
    shapiro = read_table("Table_06_Shapiro_Wilk_Normality_Test.csv")
    pettitt = read_table("Table_09_Pettitt_Change_Point_Detection.csv")
    failure = read_table("Table_10_Failure_and_RUL_Analysis.csv")

    lines = []
    lines.append("# Relatório Final — Análise de Degradação da Bateria CALCE CS2-35\n")
    lines.append(f"Diretório de dados: `{config.DATA_DIR}`\n")
    lines.append(f"Diretório de resultados: `{config.OUTPUT_DIR}`\n")

    if getattr(config, "COMPARISON_MODEL_NAME", None):
        lines.append(
            f"\n**Nota metodológica:** por pedido explícito da pesquisadora "
            f"(2026-09-07), as tabelas abaixo incluem, além do modelo "
            f"formalmente SELECIONADO pela hierarquia BIC_train (coluna "
            f"`role`='selected'), uma linha adicional de COMPARAÇÃO para o "
            f"modelo `{config.COMPARISON_MODEL_NAME}` (coluna `role`='comparison'). "
            f"O modelo de comparação NAO altera a seleção formal - é processado "
            f"em paralelo apenas para fins de análise comparativa, com as mesmas "
            f"figuras e tabelas de diagnóstico/incerteza/Pettitt/falha-RUL geradas "
            f"para ambos.\n"
        )

    if soh_summary is not None:
        lines.append("\n## 1. Resumo de SoH\n")
        lines.append(to_md(soh_summary))

    if selection_audit is not None:
        lines.append("\n## 2. Trilha de auditoria da seleção de modelo (Passo 7, hierarquia treino-only)\n")
        lines.append(
            "Adequação residual = Shapiro-Wilk sobre os resíduos de TREINO "
            "(p > 0.05) - reportado como diagnóstico formal, mas (por decisão "
            "explícita da pesquisadora em 2026-09-07, pois travava toda a "
            "seleção com n_treino~600) NAO entra no cálculo de elegibilidade. "
            "Plausibilidade física = monotonicidade da trajetória prevista "
            "(dSoH/dN <= 0) sobre a malha densa de ciclos observados. "
            "Elegível = SOMENTE plausibilidade física. Selecionado = menor "
            "BIC_train entre os elegíveis (com desempate documentado). O "
            "conjunto de TESTE NAO participa desta seleção.\n"
        )
        lines.append(to_md(selection_audit.round(6)))
        if validation is None or validation.empty:
            lines.append(
                "\n**ATENÇÃO: nenhum modelo candidato atingiu a elegíbilidade "
                "(plausibilidade física via monotonicidade) para nenhuma bateria. "
                "Nenhum modelo foi selecionado, e portanto nenhuma bateria foi "
                "processada pelos Passos 8, 9 e 10 (quantificação de incerteza, "
                "detecção de ponto de mudança e análise de falha/RUL). Isto NAO "
                "foi contornado com um fallback automático - ver tabela acima "
                "para o motivo de reprovação de cada candidato.**\n"
            )

    if validation is not None and not validation.empty:
        lines.append("\n## 2b. Modelo selecionado por bateria — validação final no teste (30%, retido)\n")
        lines.append(to_md(validation.round(4)))

    if model_comp is not None:
        lines.append("\n## 3. Comparação de todos os modelos candidatos (ajuste completo)\n")
        lines.append(to_md(model_comp.round(4)))

    if shapiro is not None:
        lines.append("\n## 4. Normalidade dos resíduos (Shapiro-Wilk)\n")
        lines.append(to_md(shapiro))

    if pettitt is not None:
        lines.append("\n## 5. Detecção de Potencial Falha (Teste de Pettitt)\n")
        lines.append(to_md(pettitt))

    if failure is not None:
        lines.append("\n## 6. Falha Funcional, RUL e Intervalo P-F\n")
        lines.append(to_md(failure))

    report_text = "\n".join(str(x) for x in lines) + "\n"
    out_path = config.OUTPUT_DIR / "RELATORIO_FINAL.md"
    out_path.write_text(report_text, encoding="utf-8")
    log.info(f"Relatório final salvo em: {out_path}")
    print("\n" + report_text)
    return report_text


if __name__ == "__main__":
    main()

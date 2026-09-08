"""
02_data_processing.py
=======================
PASSO 3: Processamento dos dados brutos.

Reescrito após auditoria (ver AUDITORIA_CS2_35.md) com as seguintes decisões
metodológicas EXPLICITAMENTE AUTORIZADAS pela pesquisadora:

1. Fonte de capacidade por ciclo: quando o arquivo Arbin tem a aba
   Statistics, ela é a fonte OFICIAL (cada linha de Statistics já é o valor
   cumulativo de Discharge_Capacity(Ah) ao final daquele ciclo — verificado
   numericamente contra a aba Channel). A aba Channel só é usada quando o
   arquivo NÃO possui aba Statistics. As duas fontes nunca são misturadas no
   mesmo cálculo de capacidade.
2. Eixo de ciclo EXPERIMENTAL vs. eixo ANALÍTICO: o número de ciclos
   verdadeiramente tentados em cada arquivo é sempre lido da aba Channel
   (presente nos 25 arquivos), usada apenas para definir o deslocamento
   (offset) do ciclo global entre arquivos. O ciclo GLOBAL nunca é
   renumerado/comprimido para esconder ciclos excluídos: se um ciclo for
   excluído, o número dele simplesmente não aparece na tabela final (a
   lacuna é preservada), em vez de os ciclos seguintes serem deslocados
   para trás.
3. Ciclos genuinamente incompletos (teste interrompido antes do fim do
   ciclo) são EXCLUÍDOS do dataset analítico, não interpolados. Dois casos
   objetivos, sem limiares inventados:
   (a) arquivo tem aba Statistics, mas ela não tem entrada para o último
       ciclo local do Channel -> esse ciclo nunca foi fechado pelo
       equipamento -> excluído;
   (b) arquivo só tem aba Channel e o último ciclo local mostra
       Discharge_Capacity(Ah) com diff (max-min) EXATAMENTE zero -> ciclo
       fisicamente impossível de ser um ciclo completo de descarga ->
       excluído.
4. "V-dips" (quedas isoladas com recuperação completa no ciclo seguinte,
   fora de fronteira de arquivo) continuam sendo detectados pelo mesmo
   critério já usado anteriormente, mas agora são EXCLUÍDOS do dataset
   (não mais interpolados), preservando a lacuna no eixo de ciclos.
5. O limiar arbitrário de anomalia ">1.5x a capacidade de referência" foi
   removido por não ter base na metodologia (README não o define). O único
   filtro físico remanescente é "capacidade <= 0", que é uma verificação de
   validade física básica, não um critério estatístico inventado.

Saída:
    processed_data/processed_data.csv         (battery_id, cycle, discharge_capacity)
    processed_data/processed_data_full.csv    (+ source_file, local_cycle, data_source)
    processed_data/excluded_cycles.csv        (todos os ciclos excluídos, com motivo)
    tables/Table_00_Data_Quality_Report.csv

Pode ser executado isoladamente (requer que 01_load_data.py já tenha rodado):
    python 02_data_processing.py
"""

import numpy as np
import pandas as pd

import config
import utils

log = utils.get_logger("02_data_processing")


def process_battery(bdf: pd.DataFrame, battery_id: str, cycle_col: str, cap_col: str):
    """
    Constrói a tabela ciclo-a-ciclo de uma bateria a partir dos dados brutos
    (todas as abas, todos os arquivos), aplicando as regras 1-3 descritas no
    cabeçalho do módulo. Retorna (per_cycle_df, exclusion_rows, file_summary_rows).
    """
    bdf = bdf.copy()
    bdf["__sheet__"] = bdf["__sheet__"].astype(str)
    is_channel = bdf["__sheet__"].str.contains("Channel", case=False, na=False)
    is_stats = bdf["__sheet__"].str.contains("Statistics", case=False, na=False)

    # ordem cronológica dos arquivos (necessária para o offset do ciclo global)
    file_order = (
        bdf.groupby("source_file")["Date_Time"].min().sort_values().index.tolist()
        if "Date_Time" in bdf.columns
        else sorted(bdf["source_file"].unique())
    )

    offset = 0
    per_cycle_rows = []
    exclusion_rows = []
    file_summary_rows = []

    for fname in file_order:
        fmask = bdf["source_file"] == fname
        ch = bdf.loc[fmask & is_channel].copy()
        st = bdf.loc[fmask & is_stats].copy()

        # eixo EXPERIMENTAL: nº de ciclos realmente tentados neste arquivo,
        # sempre a partir do Channel (presente em 100% dos arquivos)
        if len(ch):
            n_true_cycles = int(np.nanmax(ch[cycle_col].to_numpy(dtype=float)))
        elif len(st):
            n_true_cycles = int(np.nanmax(st[cycle_col].to_numpy(dtype=float)))
        else:
            n_true_cycles = 0

        n_kept, n_excl_incomplete = 0, 0

        if len(st) > 0:
            # --- fonte oficial: Statistics ---
            data_source = "Statistics"
            st_vals = (
                st.groupby(cycle_col, sort=True)[cap_col]
                .first()
                .sort_index()
            )
            st_vals = st_vals[st_vals.index.notna()]
            cap_diff = st_vals.diff()
            if len(cap_diff):
                cap_diff.iloc[0] = st_vals.iloc[0]  # contador cumulativo começa em 0 no início do arquivo

            for local_cyc, cap in cap_diff.items():
                global_cyc = offset + int(local_cyc)
                per_cycle_rows.append({
                    "battery_id": battery_id, "cycle": global_cyc,
                    "discharge_capacity": cap, "source_file": fname,
                    "local_cycle": int(local_cyc), "data_source": data_source,
                })
                n_kept += 1

            # ciclos do Channel que não têm entrada correspondente em Statistics
            # (o equipamento nunca fechou/registrou o resumo desse ciclo)
            ch_cycles = set(int(v) for v in ch[cycle_col].dropna().unique())
            st_cycles = set(int(v) for v in cap_diff.index)
            missing = sorted(ch_cycles - st_cycles)
            for lc in missing:
                exclusion_rows.append({
                    "battery_id": battery_id, "source_file": fname, "local_cycle": lc,
                    "global_cycle": offset + lc,
                    "reason": "incomplete_cycle_no_statistics_summary",
                })
                n_excl_incomplete += 1
        else:
            # --- fonte única disponível: Channel ---
            data_source = "Channel"
            g = ch.groupby(cycle_col, sort=True)[cap_col]
            cmax, cmin = g.max(), g.min()
            diff = (cmax - cmin).sort_index()
            diff = diff[diff.index.notna()]

            for local_cyc, cap in diff.items():
                global_cyc = offset + int(local_cyc)
                if cap <= 0:
                    exclusion_rows.append({
                        "battery_id": battery_id, "source_file": fname, "local_cycle": int(local_cyc),
                        "global_cycle": global_cyc,
                        "reason": "incomplete_cycle_zero_or_negative_capacity",
                    })
                    n_excl_incomplete += 1
                    continue
                per_cycle_rows.append({
                    "battery_id": battery_id, "cycle": global_cyc,
                    "discharge_capacity": cap, "source_file": fname,
                    "local_cycle": int(local_cyc), "data_source": data_source,
                })
                n_kept += 1

        file_summary_rows.append({
            "battery_id": battery_id, "source_file": fname,
            "n_true_cycles_experimental": n_true_cycles,
            "capacity_data_source": data_source,
            "n_cycles_kept": n_kept,
            "n_cycles_excluded_incomplete": n_excl_incomplete,
        })

        offset += n_true_cycles

    per_cycle = pd.DataFrame(per_cycle_rows).sort_values("cycle").reset_index(drop=True)
    return per_cycle, exclusion_rows, file_summary_rows


def detect_and_exclude_v_dips(per_cycle: pd.DataFrame, battery_id: str):
    """
    Detecta quedas isoladas de um único ciclo com recuperação completa no
    ciclo seguinte (mesmo critério já usado e apresentado à pesquisadora),
    mas agora EXCLUI esses ciclos do dataset analítico em vez de
    interpolá-los (autorizado explicitamente: "Excluir esses ciclos da
    análise"). A lacuna no eixo de ciclos é preservada.
    """
    cap_arr = per_cycle["discharge_capacity"].to_numpy(dtype=float)
    cyc_arr = per_cycle["cycle"].to_numpy()
    n = len(cap_arr)
    if n < 3:
        return per_cycle, []

    ref_cap = cap_arr[0] if cap_arr[0] > 0 else np.nanmedian(cap_arr)
    dip_thresh = config.V_DIP_RELATIVE_THRESHOLD * ref_cap if ref_cap else np.inf

    v_dip_idx = []
    for i in range(1, n - 1):
        drop_before = cap_arr[i - 1] - cap_arr[i]
        recover_after = cap_arr[i + 1] - cap_arr[i]
        if drop_before > dip_thresh and recover_after > dip_thresh:
            v_dip_idx.append(i)

    excl_rows = []
    if v_dip_idx:
        log.warning(
            f"Bateria {battery_id}: {len(v_dip_idx)} ciclo(s) isolado(s) tipo 'V-dip' "
            f"detectado(s) e EXCLUÍDO(S) do dataset analítico (ciclos: "
            f"{[int(cyc_arr[i]) for i in v_dip_idx]})"
        )
        for i in v_dip_idx:
            excl_rows.append({
                "battery_id": battery_id, "source_file": per_cycle["source_file"].iloc[i],
                "local_cycle": per_cycle["local_cycle"].iloc[i], "global_cycle": int(cyc_arr[i]),
                "reason": "isolated_v_dip_excluded",
            })
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[v_dip_idx] = False
        per_cycle = per_cycle.loc[keep_mask].reset_index(drop=True)

    return per_cycle, excl_rows


def main():
    log.info("=" * 70)
    log.info("PASSO 3: PROCESSAMENTO DOS DADOS (pós-auditoria)")
    log.info("=" * 70)

    raw_path = config.PROCESSED_DATA_DIR / "01_raw_consolidated.csv"
    if not raw_path.exists():
        log.error(f"Arquivo não encontrado: {raw_path}. Rode antes: python 01_load_data.py")
        raise SystemExit(1)

    raw = pd.read_csv(raw_path, low_memory=False)
    log.info(f"Dados brutos carregados: {len(raw)} linhas")

    cycle_col = utils.find_column(raw.columns, config.CYCLE_KEYWORDS)
    cap_col = utils.find_column(raw.columns, config.CAPACITY_KEYWORDS)
    if cycle_col is None or cap_col is None:
        log.error(f"Não foi possível identificar as colunas de ciclo/capacidade. "
                  f"Colunas disponíveis: {list(raw.columns)}")
        raise SystemExit(1)
    if "__sheet__" not in raw.columns or "source_file" not in raw.columns:
        log.error("Colunas '__sheet__'/'source_file' ausentes no dataset bruto — "
                  "necessárias para separar as representações Channel/Statistics.")
        raise SystemExit(1)
    log.info(f"Coluna de ciclo identificada: '{cycle_col}'")
    log.info(f"Coluna de capacidade identificada: '{cap_col}'")

    raw[cycle_col] = pd.to_numeric(raw[cycle_col], errors="coerce")
    raw[cap_col] = pd.to_numeric(raw[cap_col], errors="coerce")
    raw = raw.dropna(subset=[cycle_col, cap_col])

    quality_rows = []
    battery_frames = []
    all_exclusions = []
    all_file_summaries = []

    for battery_id, bdf in raw.groupby("battery_id", sort=True):
        n_before = len(bdf)

        per_cycle, excl_incomplete, file_summary = process_battery(bdf, battery_id, cycle_col, cap_col)
        all_file_summaries.extend(file_summary)

        n_after_extraction = len(per_cycle)

        # duplicatas de ciclo global (não esperadas; se ocorrerem, são
        # registradas explicitamente em vez de descartadas silenciosamente)
        n_dupe_cycles = int(per_cycle["cycle"].duplicated().sum())
        if n_dupe_cycles:
            dupe_rows = per_cycle[per_cycle["cycle"].duplicated(keep=False)]
            log.warning(f"Bateria {battery_id}: {n_dupe_cycles} ciclo(s) global(is) duplicado(s) "
                        f"encontrado(s) — mantendo a primeira ocorrência. Detalhe: "
                        f"{dupe_rows[['cycle','source_file','local_cycle']].to_dict('records')}")
            per_cycle = per_cycle.drop_duplicates(subset="cycle", keep="first").reset_index(drop=True)

        # V-dips: excluídos (não interpolados), conforme autorizado
        per_cycle, excl_vdip = detect_and_exclude_v_dips(per_cycle, battery_id)

        all_exclusions.extend(excl_incomplete)
        all_exclusions.extend(excl_vdip)

        n_final = len(per_cycle)
        sources_used = sorted(set(fs["capacity_data_source"] for fs in file_summary))

        log.info(
            f"Bateria {battery_id}: {n_before} linhas brutas -> {n_after_extraction} ciclos extraídos -> "
            f"{n_final} ciclos válidos finais "
            f"(fontes de capacidade usadas: {sources_used}; "
            f"{len(excl_incomplete)} ciclo(s) incompleto(s) excluído(s); "
            f"{len(excl_vdip)} V-dip(s) excluído(s); "
            f"{n_dupe_cycles} duplicata(s) de ciclo global)"
        )

        quality_rows.append({
            "battery_id": battery_id,
            "n_raw_rows": n_before,
            "n_cycles_extracted": n_after_extraction,
            "n_cycles_final_valid": n_final,
            "n_incomplete_cycles_excluded": len(excl_incomplete),
            "n_v_dip_cycles_excluded": len(excl_vdip),
            "n_duplicate_global_cycle_entries": n_dupe_cycles,
            "capacity_sources_used": ", ".join(sources_used),
            "reference_capacity_Q1": per_cycle["discharge_capacity"].iloc[0] if n_final else np.nan,
            "cycle_axis_min": int(per_cycle["cycle"].min()) if n_final else np.nan,
            "cycle_axis_max": int(per_cycle["cycle"].max()) if n_final else np.nan,
        })

        if n_final >= config.MIN_CYCLES_REQUIRED:
            battery_frames.append(per_cycle)
        else:
            log.warning(f"Bateria {battery_id} descartada: apenas {n_final} ciclos válidos "
                        f"(< MIN_CYCLES_REQUIRED={config.MIN_CYCLES_REQUIRED})")

    if not battery_frames:
        log.error("Nenhuma bateria com dados suficientes após o processamento. Abortando.")
        raise SystemExit(1)

    processed_full = pd.concat(battery_frames, ignore_index=True)
    processed_full.to_csv(config.PROCESSED_DATA_DIR / "processed_data_full.csv", index=False)

    processed = processed_full[["battery_id", "cycle", "discharge_capacity"]].copy()
    out_path = config.PROCESSED_DATA_DIR / "processed_data.csv"
    processed.to_csv(out_path, index=False)
    log.info(f"Dataset padronizado salvo em: {out_path}  ({len(processed)} linhas)")
    log.info("NOTA: o eixo 'cycle' preserva o ciclo experimental real — pode conter lacunas "
             "onde ciclos foram excluídos (ver processed_data/excluded_cycles.csv).")

    excl_df = pd.DataFrame(all_exclusions)
    excl_df.to_csv(config.PROCESSED_DATA_DIR / "excluded_cycles.csv", index=False)
    log.info(f"Ciclos excluídos (com motivo) salvos em: "
             f"{config.PROCESSED_DATA_DIR / 'excluded_cycles.csv'} ({len(excl_df)} linha(s))")

    fsum_df = pd.DataFrame(all_file_summaries)
    fsum_df.to_csv(config.PROCESSED_DATA_DIR / "file_processing_summary.csv", index=False)

    quality_df = pd.DataFrame(quality_rows)
    quality_path = config.TABLES_DIR / "Table_00_Data_Quality_Report.csv"
    quality_df.to_csv(quality_path, index=False)
    log.info(f"Relatório de qualidade salvo em: {quality_path}")
    log.info("Passo 3 concluído.")
    return processed


if __name__ == "__main__":
    main()

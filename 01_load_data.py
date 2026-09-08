"""
01_load_data.py
================
PASSO 2: Carregamento e consolidação dos dados.

- Lê todos os arquivos suportados (.xlsx/.xls/.csv/.txt/.json) em config.DATA_DIR
- Para arquivos Excel, examina TODAS as planilhas e mantém apenas aquelas cujas
  colunas contêm palavras-chave de ciclo E capacidade (evita assumir nomes/índices
  fixos de planilha, ex.: "Channel_1-006" vs "Channel_1-008")
- Se houver subpastas em config.DATA_DIR, cada subpasta é tratada como uma
  bateria diferente (battery_id = nome da subpasta). Caso contrário, todos os
  arquivos pertencem a uma única bateria (battery_id = nome da pasta de dados).
- Ordena os arquivos cronologicamente (pela própria coluna de data, quando
  disponível; senão pelo nome do arquivo) e concatena tudo preservando a
  ordem temporal.

Saída:
    processed_data/01_raw_consolidated.csv

Pode ser executado isoladamente:
    python 01_load_data.py
"""

from pathlib import Path

import pandas as pd

import config
import utils

log = utils.get_logger("01_load_data")


def battery_id_for_file(path: Path) -> str:
    """Se o arquivo estiver dentro de uma subpasta de DATA_DIR, usa o nome da
    subpasta como battery_id. Caso contrário, usa o nome da própria DATA_DIR."""
    rel = path.relative_to(config.DATA_DIR)
    if len(rel.parts) > 1:
        return rel.parts[0]
    return config.DATA_DIR.name or config.BATTERY_ID_DEFAULT


def load_excel_file(path: Path) -> pd.DataFrame:
    """Lê todas as planilhas de um .xlsx/.xls e retorna a concatenação das
    planilhas que contêm colunas de ciclo E capacidade reconhecíveis."""
    frames = []
    try:
        xl = pd.ExcelFile(path)
    except Exception as e:  # noqa: BLE001
        log.warning(f"Falha ao abrir {path.name}: {e}")
        return pd.DataFrame()

    for sheet in xl.sheet_names:
        try:
            df = pd.read_excel(xl, sheet_name=sheet)
        except Exception as e:  # noqa: BLE001
            log.warning(f"Falha ao ler a planilha '{sheet}' de {path.name}: {e}")
            continue
        if df.empty:
            continue
        cycle_col = utils.find_column(df.columns, config.CYCLE_KEYWORDS)
        cap_col = utils.find_column(df.columns, config.CAPACITY_KEYWORDS)
        if cycle_col is None or cap_col is None:
            continue  # provavelmente a planilha "Info" ou similar
        df = df.copy()
        df["__sheet__"] = sheet
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_csv_txt_file(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception as e:  # noqa: BLE001
        log.warning(f"Falha ao ler {path.name}: {e}")
        return pd.DataFrame()
    return df


def frame_signature(df: pd.DataFrame):
    """Assinatura de conteúdo (independente do nome do arquivo) usada para
    detectar arquivos DUPLICADOS — um problema real e documentado no dataset
    CALCE (ex.: dois arquivos com nomes de data diferentes podem conter
    exatamente as mesmas linhas, provavelmente por re-exportação/re-upload).
    Se não for possível calcular, retorna None (duplicidade não verificada).
    """
    date_col = utils.find_column(df.columns, config.DATE_KEYWORDS)
    cap_col = utils.find_column(df.columns, config.CAPACITY_KEYWORDS)
    cyc_col = utils.find_column(df.columns, config.CYCLE_KEYWORDS)
    cols = [c for c in [date_col, cyc_col, cap_col] if c is not None]
    if not cols:
        return None
    try:
        sub = df[cols].astype(str)
        h = int(pd.util.hash_pandas_object(sub, index=False).sum())
    except Exception:  # noqa: BLE001
        return None
    return (len(df), h)


def main():
    log.info("=" * 70)
    log.info("PASSO 2: CARREGAMENTO DE DADOS")
    log.info("=" * 70)

    files = utils.discover_data_files(config.DATA_DIR)
    if not files:
        log.error(f"Nenhum arquivo de dados encontrado em {config.DATA_DIR}")
        raise SystemExit(1)

    all_frames = []
    seen_signatures = {}  # signature -> nome do primeiro arquivo com essa assinatura
    n_duplicates_skipped = 0
    duplicate_files_skipped = []
    for f in files:
        log.info(f"Lendo: {f.relative_to(config.DATA_DIR)}")
        if f.suffix.lower() in (".xlsx", ".xls"):
            df = load_excel_file(f)
        elif f.suffix.lower() in (".csv", ".txt"):
            df = load_csv_txt_file(f)
        else:
            log.warning(f"Formato ainda não suportado, ignorando: {f.name}")
            continue

        if df.empty:
            log.warning(f"  -> Nenhuma planilha/dado com colunas de ciclo+capacidade em {f.name}")
            continue

        sig = frame_signature(df)
        if sig is not None and sig in seen_signatures:
            log.warning(
                f"  -> IGNORADO: conteúdo idêntico ao arquivo '{seen_signatures[sig]}' "
                f"(provável arquivo duplicado — mesmas datas/ciclos/capacidades)."
            )
            n_duplicates_skipped += 1
            duplicate_files_skipped.append({"file": f.name, "duplicate_of": seen_signatures[sig]})
            continue
        if sig is not None:
            seen_signatures[sig] = f.name

        df["source_file"] = f.name
        df["battery_id"] = battery_id_for_file(f)
        all_frames.append(df)
        log.info(f"  -> {len(df)} linhas carregadas")

    if n_duplicates_skipped:
        log.warning(f"Total de arquivos duplicados detectados e ignorados: {n_duplicates_skipped}")

    if not all_frames:
        log.error("Nenhum dado válido foi carregado de nenhum arquivo. Abortando.")
        raise SystemExit(1)

    raw = pd.concat(all_frames, ignore_index=True, sort=False)
    log.info(f"Total consolidado (todas as baterias/arquivos): {len(raw)} linhas, "
              f"{raw['battery_id'].nunique()} bateria(s): {sorted(raw['battery_id'].unique())}")

    # ------------------------------------------------------------------
    # Ordenação cronológica: usa a coluna de data quando disponível;
    # caso contrário, ordena por (battery_id, nome_do_arquivo) e mantém a
    # ordem original das linhas dentro do arquivo (comportamento estável).
    # ------------------------------------------------------------------
    date_col = utils.find_column(raw.columns, config.DATE_KEYWORDS)
    if date_col is not None:
        raw[date_col] = pd.to_datetime(raw[date_col], errors="coerce")
        if raw[date_col].notna().any():
            log.info(f"Coluna de data/hora identificada: '{date_col}' — ordenando cronologicamente.")
            # ordem dos arquivos por menor data observada (preserva ordem interna do arquivo)
            file_order = (
                raw.groupby("source_file")[date_col].min().sort_values().index.tolist()
            )
            raw["__file_order__"] = raw["source_file"].map({f: i for i, f in enumerate(file_order)})
            raw = raw.sort_values(["battery_id", "__file_order__", date_col], kind="stable")
            raw = raw.drop(columns="__file_order__")
        else:
            log.warning("Coluna de data encontrada mas vazia/ilegível — ordenando por nome de arquivo.")
            raw = raw.sort_values(["battery_id", "source_file"], kind="stable")
    else:
        log.warning("Nenhuma coluna de data/hora identificada — ordenando por nome de arquivo "
                    "(convenção CALCE: nome do arquivo reflete a data de download).")
        raw = raw.sort_values(["battery_id", "source_file"], kind="stable")

    raw = raw.reset_index(drop=True)

    out_path = config.PROCESSED_DATA_DIR / "01_raw_consolidated.csv"
    raw.to_csv(out_path, index=False)
    log.info(f"Dados consolidados salvos em: {out_path}  ({len(raw)} linhas)")

    utils.save_json(
        {
            "n_files_loaded": len(all_frames),
            "n_duplicate_files_skipped": n_duplicates_skipped,
            "duplicate_files_skipped": duplicate_files_skipped,
        },
        config.PROCESSED_DATA_DIR / "01_load_stats.json",
    )
    log.info("Passo 2 concluído.")
    return raw


if __name__ == "__main__":
    main()

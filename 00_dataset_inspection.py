"""
00_dataset_inspection.py
=========================
PASSO 1: Inspeção do dataset.

- Localiza todos os arquivos de dados em config.DATA_DIR (recursivamente)
- Identifica formatos e tamanhos
- Para arquivos Excel, lista as planilhas (sheets) e um preview de colunas
- Salva um relatório de inspeção em processed_data/00_dataset_inspection.json

Pode ser executado isoladamente:
    python 00_dataset_inspection.py
"""

from pathlib import Path

import pandas as pd

import config
import utils

log = utils.get_logger("00_dataset_inspection")


def inspect_excel(path: Path) -> dict:
    info = {"sheets": {}}
    try:
        xl = pd.ExcelFile(path)
        for sheet in xl.sheet_names:
            try:
                preview = pd.read_excel(xl, sheet_name=sheet, nrows=3)
                info["sheets"][sheet] = {
                    "n_columns_preview": preview.shape[1],
                    "columns": [str(c) for c in preview.columns],
                }
            except Exception as e:  # noqa: BLE001
                info["sheets"][sheet] = {"error": str(e)}
    except Exception as e:  # noqa: BLE001
        info["error"] = str(e)
    return info


def inspect_csv_txt(path: Path) -> dict:
    try:
        preview = pd.read_csv(path, nrows=3, sep=None, engine="python")
        return {"columns": [str(c) for c in preview.columns]}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def main():
    log.info("=" * 70)
    log.info("PASSO 1: INSPEÇÃO DO DATASET")
    log.info("=" * 70)
    log.info(f"Diretório de dados: {config.DATA_DIR}")

    if not config.DATA_DIR.exists():
        log.error(f"Diretório de dados não encontrado: {config.DATA_DIR}")
        log.error("Verifique o caminho definido em config.DATA_DIR ou na variável "
                   "de ambiente BATTERY_DATA_DIR.")
        raise SystemExit(1)

    files = utils.discover_data_files(config.DATA_DIR)
    log.info(f"Arquivos de dados encontrados: {len(files)}")

    report = {"data_dir": str(config.DATA_DIR), "n_files": len(files), "files": {}}

    total_size = 0
    ext_counts = {}
    for f in files:
        size = f.stat().st_size
        total_size += size
        ext_counts[f.suffix.lower()] = ext_counts.get(f.suffix.lower(), 0) + 1

        entry = {"size_bytes": size, "extension": f.suffix.lower()}
        if f.suffix.lower() in (".xlsx", ".xls"):
            entry.update(inspect_excel(f))
        elif f.suffix.lower() in (".csv", ".txt"):
            entry.update(inspect_csv_txt(f))
        report["files"][str(f.relative_to(config.DATA_DIR))] = entry

    log.info(f"Tamanho total dos dados: {total_size / (1024 * 1024):.2f} MB")
    for ext, n in ext_counts.items():
        log.info(f"  {ext or '(sem extensão)'}: {n} arquivo(s)")

    # Estrutura de diretórios (subpastas = possíveis baterias distintas)
    subdirs = sorted([p.name for p in config.DATA_DIR.iterdir() if p.is_dir()])
    report["subdirectories"] = subdirs
    if subdirs:
        log.info(f"Subdiretórios encontrados (possíveis baterias): {subdirs}")
    else:
        log.info("Nenhum subdiretório encontrado — todos os arquivos pertencem a uma única bateria.")

    utils.save_json(report, config.PROCESSED_DATA_DIR / "00_dataset_inspection.json")
    log.info(f"Relatório salvo em: {config.PROCESSED_DATA_DIR / '00_dataset_inspection.json'}")
    log.info("Passo 1 concluído.")
    return report


if __name__ == "__main__":
    main()

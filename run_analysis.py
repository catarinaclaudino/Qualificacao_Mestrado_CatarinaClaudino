"""
run_analysis.py
=================
Script mestre de execução do pipeline de análise de degradação da bateria
CALCE CS2-35.

Uso:
    python run_analysis.py --all      # roda todos os 11 passos em sequência
    python run_analysis.py 4          # roda apenas o passo 4
    python run_analysis.py --list     # lista os passos disponíveis
"""

import argparse
import importlib.util
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

STEPS = [
    ("00", "00_dataset_inspection.py", "Inspeção do dataset"),
    ("01", "01_load_data.py", "Carregamento de dados"),
    ("02", "02_data_processing.py", "Processamento de dados"),
    ("03", "03_soh_calculation.py", "Cálculo de SoH"),
    ("04", "04_exploratory_analysis.py", "Análise exploratória"),
    ("05", "05_degradation_models.py", "Modelagem de degradação"),
    ("06", "06_residual_analysis_and_validation.py", "Análise de resíduos e validação temporal"),
    ("07", "07_uncertainty_quantification.py", "Quantificação de incerteza"),
    ("08", "08_pettitt_change_point.py", "Detecção de ponto de mudança (Pettitt)"),
    ("09", "09_failure_analysis_and_rul.py", "Análise de falha e RUL"),
    ("10", "10_final_summary.py", "Relatório final"),
]


def load_module(filename):
    spec = importlib.util.spec_from_file_location(filename[:-3], BASE_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_step(step_id):
    for sid, filename, desc in STEPS:
        if sid == step_id or sid.lstrip("0") == step_id.lstrip("0"):
            print(f"\n{'#' * 70}\n# PASSO {sid}: {desc}\n{'#' * 70}")
            t0 = time.time()
            mod = load_module(filename)
            mod.main()
            print(f"[OK] Passo {sid} concluído em {time.time() - t0:.1f}s")
            return True
    print(f"Passo '{step_id}' não encontrado. Use --list para ver os passos disponíveis.")
    return False


def main():
    parser = argparse.ArgumentParser(description="Pipeline de análise de degradação CALCE CS2-35")
    parser.add_argument("step", nargs="?", help="Número do passo a executar (ex.: 3, 03)")
    parser.add_argument("--all", action="store_true", help="Roda todos os passos em sequência")
    parser.add_argument("--list", action="store_true", help="Lista os passos disponíveis")
    args = parser.parse_args()

    if args.list:
        print("Passos disponíveis:")
        for sid, filename, desc in STEPS:
            print(f"  {sid}: {desc}  ({filename})")
        return

    if args.all:
        t_start = time.time()
        for sid, filename, desc in STEPS:
            run_step(sid)
        print(f"\nPipeline completo executado em {time.time() - t_start:.1f}s")
        return

    if args.step:
        run_step(args.step.zfill(2) if args.step.isdigit() else args.step)
        return

    parser.print_help()


if __name__ == "__main__":
    main()

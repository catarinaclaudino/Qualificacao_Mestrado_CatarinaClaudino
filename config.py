"""
config.py
=========
Configuração centralizada do pipeline de análise de degradação da bateria
CALCE CS2-35 (Mestrado ITA - Catarina Claudino).

Só é necessário editar os caminhos abaixo se os diretórios do seu computador
mudarem. As variáveis de ambiente BATTERY_DATA_DIR / BATTERY_OUTPUT_DIR têm
prioridade sobre os valores fixos (útil para testar em outra máquina/SO sem
alterar o código-fonte).
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. CAMINHOS DE ENTRADA E SAÍDA
# ---------------------------------------------------------------------------

# Diretório com os dados brutos da bateria CS2_35 (arquivos .xlsx do CALCE)
DATA_DIR = Path(os.environ.get(
    "BATTERY_DATA_DIR",
    r"colocar caminho aqui",
))

# Diretório onde TODOS os resultados (figuras, tabelas, dados processados,
# logs) serão gravados
OUTPUT_DIR = Path(os.environ.get(
    "BATTERY_OUTPUT_DIR",
    r"colocar caminho aqui",
))

FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
PROCESSED_DATA_DIR = OUTPUT_DIR / "processed_data"
LOGS_DIR = OUTPUT_DIR / "logs"

for _d in (OUTPUT_DIR, FIGURES_DIR, TABLES_DIR, PROCESSED_DATA_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Nome padrão da bateria, usado quando não é possível inferir um
# identificador a partir dos próprios dados (ex.: coluna Battery/Cell ausente).
BATTERY_ID_DEFAULT = "CS2_35"

# ---------------------------------------------------------------------------
# 2. PARÂMETROS DA ANÁLISE (Ver LEIA-ME.md para a justificativa de cada um)
# ---------------------------------------------------------------------------

TRAIN_FRACTION = 0.70          # Validação temporal: 70% treino / 30% teste
ALPHA = 0.05                   # Nível de significância (testes estatísticos)
SOH_FAILURE_THRESHOLD = 80.0   # Critério de Falha Funcional (%), convencional
PERSISTENCE_WINDOW = 5         # Nº de observações para o critério de persistência (Pettitt)
POLYNOMIAL_DEGREES = [2]
RANDOM_SEED = 42
FIGURE_DPI = 300
MIN_CYCLES_REQUIRED = 20       # Mínimo de ciclos para considerar a bateria válida

# Limiar (fração da capacidade de referência Q1) para detectar e corrigir
# anomalias isoladas de um único ciclo ("V-dips": queda abrupta seguida de
# recuperação total no ciclo seguinte — artefato de medição, não degradação
# real). Ver Passo 3 (02_data_processing.py) e Table_00 para os ciclos afetados.
V_DIP_RELATIVE_THRESHOLD = 0.05

# ---------------------------------------------------------------------------
# 2b. HIERARQUIA DE SELEÇÃO DE MODELOS (Passo 7) - conforme especificação
#     adequação residual (Shapiro-Wilk) + plausibilidade física
#     (monotonicidade) determinam ELEGIBILIDADE; BIC (calculado SOMENTE nos
#     dados de treino/ajuste) decide entre os elegiveis; o conjunto de teste
#     NUNCA participa da seleÇão, apenas da validação final independente.
# ---------------------------------------------------------------------------

BIC_TIE_TOLERANCE = 2.0
# ESCOLHA DE IMPLEMENTAÇÃO (não especificada explicitamente na metodologia):
# quando dois ou mais modelos elegíveis têm |BIC_a - BIC_b| < esta tolerancia,
# são tratados como "praticamente indistinguíveis" e o critério secundário da
# Seção 11.6 do prompt (R² ajustado > RMSE treino > MAE treino) decide o
# desempate. Segue a convenção usual de magnitude de Kass & Raftery (1995,
# J. Am. Stat. Assoc._90(430):773-795): |ΔBIC| < 2 é "não mais que uma mera
# menção". Sinalizado aqui para revisão/aprovação da pesquisadora.

MONOTONICITY_TOLERANCE = 1e-6
# ESCOLHA DE IMPLEMENTAÇÃO: tolerância absoluta (em pontos percentuais de SoH)
# abaixo da qual um aumento entre pontos consecutivos da trajetória prevista
# é tratado como ruído numérico de ponto flutuante, e não como um aumento
# fisicamente significativo de SoH, ao avaliar o critério de monotonicidade/
# plausibilidade física (Seção 7 do prompt). Sinalizado para revisão da
# pesquisadora.

MONOTONICITY_GRID_N_POINTS = 2000
# ESCOLHA DE IMPLEMENTAÇÃO: número de pontos igualmente espaçados usados para
# construir a malha densa de ciclos sobre a qual a monotonicidade de cada
# modelo candidato é avaliada. A malha cobre o intervalo de ciclos
# efetivamente OBSERVADO da bateria (treino + teste, ou seja, o domínio em
# que o modelo é de fato usado no restante do pipeline) - NAO um intervalo
# extrapolado arbitrariamente além do dataset.
# ---------------------------------------------------------------------------
# 2c. MODELO SECUNDÁRIO PARA COMPARAÇÃO (Passos 8, 9 e 10) -
#     NÃO é uma regra automática de "segundo
#     melhor modelo" inferida por este código - é uma escolha pontual dela
#     (Polynomial_2 e Power_Law ficaram estatisticamente quase
#     indistinguíveis pelo BIC_train; ver Table_Model_Selection_Audit.csv,
#     onde ambos estão dentro de BIC_TIE_TOLERANCE do mínimo).
#
#     Quando definido (!= None), o pipeline roda TAMBÉM os Passos 8
#     (incerteza), 9 (Pettitt) e 10 (falha/RUL), além dos diagnósticos de
#     resíduo do Passo 7 (figuras 06-09) e da validação temporal (figura 10),
#     para este modelo - gerando um conjunto PARALELO e COMPLETO de
#     figuras/tabelas, rotulado role="comparison", ao lado do conjunto do
#     modelo formalmente selecionado (role="selected").
#
#     Isto NÃO altera a seleção formal feita pela hierarquia BIC no Passo 7 -
#     o "selected_model"/"selection_rank" em Table_Model_Selection_Audit.csv
#     continuam refletindo exclusivamente o resultado da hierarquia
#     (Shapiro-diagnóstico -> monotonicidade-elegibilidade -> BIC_train ->
#     desempate). O modelo de comparação é uma análise de SENSIBILIDADE
#     adicional, solicitada explicitamente, não um segundo "vencedor".
#
#     Defina como None para desativar (comportamento original: um único
#     modelo por bateria, nenhuma tabela/figura adicional).
# ---------------------------------------------------------------------------
COMPARISON_MODEL_NAME = "Power_Law"

# ---------------------------------------------------------------------------
# 3. PALAVRAS-CHAVE PARA IDENTIFICAÇÃO AUTOMÁTICA DE COLUNAS (Passo 3)
#    Nenhum nome de coluna é assumido a priori: procuramos por estas
#    palavras-chave (normalizadas) dentro dos nomes de colunas reais.
# ---------------------------------------------------------------------------

BATTERY_ID_KEYWORDS = ["battery_id", "battery", "cell_id", "cell", "bateria"]
CYCLE_KEYWORDS = ["cycle_index", "cycle_number", "cycle", "ciclo"]
CAPACITY_KEYWORDS = [
    "discharge_capacity", "discharge_cap", "dischg_cap",
    "capacity_ah", "capacity", "capacidade",
]
DATE_KEYWORDS = ["date_time", "datetime", "date", "timestamp", "data"]
FULL_CYCLE_FLAG_KEYWORDS = ["is_fc_data", "fc_data"]

SUPPORTED_EXTENSIONS = (".xlsx", ".xls", ".csv", ".txt", ".json")

# Como rodar no VS Code — Análise de Degradação CALCE CS2-35

Este projeto implementa as 10 etapas descritas no README original da metodologia (inspeção do
dataset → carregamento → processamento → cálculo de SoH → análise exploratória → modelos de
degradação → validação/resíduos → incerteza → teste de Pettitt → falha funcional/RUL → resumo
final).

## 1. Pré-requisitos

- Python 3.9 ou mais recente instalado no Windows.
- VS Code com a extensão "Python" (Microsoft) instalada.
- Os dados da bateria CS2_35 (arquivos `.xlsx` do CALCE) já devem estar em:
  `C:\Users\catar\OneDrive\Área de Trabalho\MESTRADO\TESTE_METODOLOGIA\CS2_35`

## 2. Preparar o ambiente

Abra a pasta deste projeto no VS Code (`Arquivo > Abrir Pasta...`). Depois, abra um terminal
integrado (`Terminal > Novo Terminal`) e rode:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

(Toda vez que abrir o VS Code novamente para trabalhar neste projeto, ative o ambiente virtual de
novo com `.venv\Scripts\activate` antes de rodar os scripts.)

## 3. Conferir os caminhos em `config.py`

Os caminhos já estão configurados por padrão para:

- **Dados de entrada:** `C:\Users\catar\OneDrive\Área de Trabalho\MESTRADO\TESTE_METODOLOGIA\CS2_35`
- **Resultados de saída:** `C:\Users\catar\OneDrive\Área de Trabalho\MESTRADO\TESTE_METODOLOGIA`

Não é necessário editar nada se esses forem os caminhos corretos no seu computador. Se algum dia
os diretórios mudarem, basta editar as duas linhas correspondentes em `config.py` (linhas ~21-31).

## 4. Rodar a análise completa

No terminal do VS Code (com o ambiente virtual ativado):

```powershell
python run_analysis.py --all
```

Isso executa as 11 etapas em sequência (inspeção do dataset até o resumo final) e demora menos
de um minuto. Ao final, você verá as pastas `figures/`, `tables/`, `processed_data/` e `logs/`
criadas dentro de `TESTE_METODOLOGIA` (o diretório de saída configurado), além do arquivo
`RELATORIO_FINAL.md` com um resumo consolidado de todos os resultados numéricos.

### Rodar apenas uma etapa específica

Se quiser rodar só uma etapa (por exemplo, para depurar ou reprocessar depois de adicionar novos
arquivos de dados), use:

```powershell
python run_analysis.py --list          # lista todas as etapas disponíveis com seus números
python run_analysis.py 05              # roda só a etapa 05 (modelos de degradação)
```

Atenção: como as etapas dependem dos resultados salvos pelas anteriores (arquivos em
`processed_data/`), rode-as sempre em ordem crescente na primeira vez.

## 5. O que verificar depois de rodar

- `figures/`: 14 gráficos (capacidade vs. ciclo, SoH vs. ciclo, comparação de modelos, resíduos,
  teste de Pettitt, análise de falha e RUL, etc.).
- `tables/`: 11 tabelas em CSV (`Table_00` a `Table_10`), com o relatório de qualidade dos dados,
  comparação de modelos, resultados do teste de Pettitt, e a tabela final de falha/RUL.
- `RELATORIO_FINAL.md`: resumo em Markdown de todos os números principais, pronto para consulta
  rápida ou para copiar trechos para a dissertação.
- `logs/pipeline.log`: registro detalhado de cada execução.
- `processed_data/excluded_cycles.csv`: toda exclusão de dado feita pelo Passo 3 (ciclo
  incompleto ou "V-dip"), com motivo explícito — nenhum valor é modificado ou interpolado
  silenciosamente; o que não pode ser usado com segurança é excluído e documentado aqui.
- `processed_data/file_processing_summary.csv`: qual fonte de capacidade (Statistics ou
  Channel) foi usada em cada arquivo.

## 6. Leia também

- `AUDITORIA_CS2_35.md`: auditoria metodológica completa do pipeline em relação aos dados reais
  e ao README original, com as decisões que exigiram autorização explícita.
- `RELATORIO_AUDITORIA_E_CORRECOES.md`: o que foi corrigido no código após a auditoria, e a
  comparação "antes vs. depois" dos resultados numéricos.
- `ANALISE_PLAUSIBILIDADE.md`: análise escrita avaliando se os resultados obtidos na validação com
  os dados reais fazem sentido físico/estatístico, comparando com valores da literatura sobre a
  célula CALCE CS2-35 (nota: escrita antes da auditoria — ver nota de atualização no topo do
  documento).

## Observação importante sobre os dados

O pipeline foi validado (nesta entrega) com os 25 arquivos `.xlsx` originais do CALCE para a
célula CS2_35, baixados diretamente de
[web.calce.umd.edu/batteries/data/CS2_35.zip](https://web.calce.umd.edu/batteries/data/CS2_35.zip).
Se a pasta `CS2_35` na sua máquina tiver um conjunto diferente de arquivos (mais ou menos ciclos),
os números finais (Q_ref, N_F_hat, N_P, etc.) serão recalculados automaticamente a partir dos seus
dados — o código não depende de valores fixos.

#import quant_tools as qt
import numpy as np
import matplotlib.pyplot as plt
#import os
import pandas as pd
import statsmodels.api as sm
import seaborn as sns

#os.chdir('C:\\Users\\u010583\\Documents\\Python\\csv(s)')
#os.getcwd()
dados = pd.read_excel("C:\\Users\\JoãoCampos\\Documents\\Summer Job _ Detalhes\\Métricas _ Rascunhos\\Zeno e Benchmarks .xlsx")
print(dados)

# Ajuste do índice
dados.index = dados['Dates']
dados.drop(["Dates"], axis=1, inplace=True)
dados.head()

# Adicionando Taxas
pct_change1 = dados.pct_change()

taxa_praticada = 1 / 100
taxa_anualizada = ((taxa_praticada + 1) ** (1 / 365)) - 1

colors = ['#0063DE', '#C00000', "#7F7F7F"]

# Retorno Acumulado
normalizado = dados / dados.iloc[0]
retorno_acumulado = normalizado - 1
retorno_acumulado.plot(figsize=(12, 6), color=colors)
plt.title("Retorno Acumulado")
plt.show()

# Correlação
plt.figure(figsize=(4, 4))
parametros = {
    "shrink": 1,
    "extend": "min",
    "extendfrac": .1,
    "drawedges": True,
    "ticks": [-1, -.8, -.6, -.4, -.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
}

correlacao = pct_change1.corr()
ax = sns.heatmap(correlacao,
                 vmin=0, vmax=1, annot=True, cmap="BrBG", xticklabels=True, yticklabels=True, robust=True,
                 cbar_kws=parametros, linewidths=.9)

sm.graphics.plot_corr(correlacao, xnames=correlacao.columns)
plt.title("Matriz de Correlação")
plt.rcParams["figure.figsize"] = (10, 5)
plt.show()

# Risco x Retorno
desvpad = np.std(pct_change1, axis=0)
vol_anual = (desvpad * np.sqrt(365)) * 100
retorno_acumulado = ((dados.iloc[-1] / dados.iloc[0]) - 1)
retorno_anualizado = (((dados.iloc[-1] / dados.iloc[0]) ** (365 / len(dados))) - 1) * 100

vol_anual = vol_anual[retorno_anualizado.index]

print(len(vol_anual), len(retorno_anualizado))

eficiencia = retorno_anualizado / vol_anual

plt.scatter(vol_anual, retorno_anualizado)
plt.title("Risco x Retorno")
plt.xlabel("Risco (%)")
plt.ylabel("Retorno (%)")
plt.show()

# Janelas Moveis de Retorno
## 3 anos
janelas_moveis_retorno = dados
retornos = janelas_moveis_retorno.pct_change(1095)
retornos_anualiz = (retornos + 1).pow(1 / 3)
retornos_final = retornos_anualiz - 1
retornos_final.plot(figsize=(12, 6), color=colors)
plt.title("Janelas Móveis de Retorno - 3 Anos")
plt.show()

## 5 anos
retornos_1 = janelas_moveis_retorno.pct_change(1825)
retornos_anualiz_1 = (retornos_1 + 1).pow(1 / 5)
retornos_final_1 = retornos_anualiz_1 - 1
retornos_final_1.plot(figsize=(12, 6), color=colors)
plt.title("Janelas Móveis de Retorno - 5 Anos")
plt.show()

# Comparação T. Rowe Price US Smaller Companies (Espelho BTG) x Russel 2500 Net 30% Return
## Janelas de 3 anos
retornos_positivos = retornos_final["Zeno Global FI Financeiro Em Ações IE Resp Limitada "] - retornos_final[
    "Ibovespa "]
print(retornos_positivos.describe())

pos_count, neg_count = 0, 0
for retorno in retornos_positivos:
    if retorno >= 0:
        pos_count += 1
    else:
        neg_count += 1

print("Taxa de sucesso contra Russel 2500 em janelas de 3 anos:", pos_count / (4743 * 100))

retornos_positivos.plot()
plt.title("Retornos Positivos - 3 Anos")
plt.show()

## Janelas de 5 anos
retornos_positivos1 = retornos_final_1["Zeno Global FI Financeiro Em Ações IE Resp Limitada "] - retornos_final_1[
    "Ibovespa "]
print(retornos_positivos1.describe())

pos_count, neg_count = 0, 0
for retorno in retornos_positivos1:
    if retorno >= 0:
        pos_count += 1
    else:
        neg_count += 1

print("Taxa de sucesso contra Ibovespa em 5 anos:", pos_count / len(retornos_positivos1))

retornos_positivos1.plot(color=colors)
plt.title("Retornos Positivos - 5 Anos")
plt.show()

# Janelas Moveis de Volatilidade
janelas_vol = pct_change1.rolling(12).std()
janelas_vol_final = janelas_vol * np.sqrt(12)
janelas_vol_final.plot(figsize=(12, 6), color=colors)
plt.title("Janelas Móveis de Volatilidade")
plt.show()

# Drawndown
wea_idx = (pct_change1 + 1).cumprod()
picos = wea_idx.cummax()
drawdown = (wea_idx - picos) / picos

plt.figure(figsize=(12, 6))

# Plote a série Ibovespa
plt.fill_between(drawdown.index, drawdown['Ibovespa '], color=colors[2], label='Ibovespa ')

# Plote a série S&P 500 primeiro
plt.fill_between(drawdown.index, drawdown['S&P 500 (em reais)'], color=colors[1], label='S&P 500 (em reais)')

# Plote a série Zeno por cima
plt.fill_between(drawdown.index, drawdown['Zeno Global FI Financeiro Em Ações IE Resp Limitada '], color=colors[0], label='Zeno Global FI Financeiro Em Ações IE Resp Limitada ')

plt.title("Drawdown")
plt.legend(loc='best')
plt.show()

drawdown_min = drawdown[drawdown < 0]
max_ddw = drawdown.min()
print("Máximo Drawdown:", max_ddw)

# Downside Risk
downside = pct_change1[pct_change1 > 0]
downside_vol = np.std(downside, axis=0)
downside_vol = downside_vol[retorno_anualizado.index]
indice_sortino = retorno_anualizado / downside_vol

plt.scatter(downside_vol, retorno_anualizado)
plt.title("Índice de Sortino")
plt.xlabel("Downside (%)")
plt.ylabel("Retorno (%)")
plt.show()

#qt.plot_head_sns(etfs_past_pf.d_returns(is_portfolio=False), size=(8, 6))

def create_dashboard(fundo_especifico):
    dados_fundo = dados[dados["Fundo"] == fundo_especifico].copy()



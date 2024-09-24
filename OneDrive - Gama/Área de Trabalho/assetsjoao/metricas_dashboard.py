import os
import pandas as pd
import numpy as np
import pdfplumber
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import panel as pn
import textwrap
from datetime import datetime, timedelta
import plotly.io as pio
import time
import shutil
import io
from PIL import Image
from bokeh.models.widgets import Div

pio.kaleido.scope.plotlyjs = "https://cdn.plot.ly/plotly-latest.min.js"

pn.extension('fast')

custom_css = """
<style>
.bk-btn-primary {
    background-color: #1a1a75 !important;
    border-color: #1a1a75 !important;
}
.bk-btn-primary.bk-active {
    background-color: #000066 !important;
    border-color: #000066 !important;
}
</style>
"""

# Add custom CSS to the document
pn.config.raw_css.append(custom_css)

# Determine the file path dynamically
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "Zeno e Benchmarks .xlsx")

# Verify that the file path is correct
print("File path:", file_path)

# Load the data with two headers
dados = pd.read_excel(file_path, header=[0, 1])

# Ajustar o índice e remover a coluna de datas
dados.index = pd.to_datetime(dados[('Unnamed: 0_level_0', 'Dates')])
dados.drop(columns=[('Unnamed: 0_level_0', 'Dates')], inplace=True)

# Verificar os nomes das colunas
print("Nomes das colunas antes de droplevel:", dados.columns)

# Extrair apenas os fundos e benchmarks
fundos = dados.xs('Fundo', axis=1, level=0)
benchmarks = dados.xs('Benchmark', axis=1, level=0)

# Verificar os fundos e benchmarks extraídos
print("Fundos:", fundos.columns)
print("Benchmarks:", benchmarks.columns)

# Concatenar fundos e benchmarks para formar o DataFrame final
dados = pd.concat([fundos, benchmarks], axis=1)

# Verificar os nomes das colunas após a concatenação
print("Nomes das colunas após concatenação:", dados.columns)

warbook_path = "BasedadosWarbook.xlsm"

comentarios = pd.read_excel(warbook_path, sheet_name='Comentarios', header=None)
comentarios = comentarios.drop(index=0).reset_index(drop=True)
comentarios.columns = comentarios.iloc[0]
comentarios = comentarios.drop(index=0).reset_index(drop=True)

ideias = pd.read_excel(warbook_path, sheet_name='Ideias', header=None)
ideias = ideias.drop(index=0).reset_index(drop=True)
ideias.columns = ideias.iloc[0]
ideias = ideias.drop(index=0).reset_index(drop=True)

# Função para filtrar dados
def filtrar_dados_por_periodo(dados, periodo):
    hoje = dados.index.max()
    if periodo == '1M':
        inicio = hoje - timedelta(days=30)
    elif periodo == '3M':
        inicio = hoje - timedelta(days=90)
    elif periodo == '6M':
        inicio = hoje - timedelta(days=180)
    elif periodo == '12M':
        inicio = hoje - timedelta(days=365)
    elif periodo == 'YTD':
        inicio = datetime(hoje.year, 1, 1)
    elif periodo == '5Y':
        inicio = hoje - timedelta(days=5 * 365)
    else:
        inicio = dados.index.min()

    dados_filtrados = dados[dados.index >= inicio]
    print(f"Dados filtrados para o período '{periodo}':\n{dados_filtrados.head()}")  # Adicionar print para depuração

    return dados_filtrados


# Função para criar o dashboard
def home_page(fundo_especifico, periodo):
    # Filtrar dados para o fundo específico e benchmarks
    print(f"Fundo selecionado: {fundo_especifico}")
    print(f"Colunas disponíveis: {dados.columns}")

    # Filtrar e normalizar os dados com base no período selecionado
    fundos_colunas = [col for col in dados.columns if fundo_especifico in col]
    benchmarks_colunas = [col for col in benchmarks.columns]
    colunas_selecionadas = fundos_colunas + benchmarks_colunas

    if not fundos_colunas:
        return pn.pane.Markdown(f"### Fundo '{fundo_especifico}' não encontrado.")

    dados_fundo = dados[colunas_selecionadas]
    dados_fundo = filtrar_dados_por_periodo(dados_fundo, periodo)

    # Cálculos necessários
    pct_change1 = dados_fundo.pct_change()

    taxa_praticada = 1 / 100
    taxa_anualizada = ((taxa_praticada + 1) ** (1 / 365)) - 1

    colors = ['#0063DE', '#C00000', "#7F7F7F"]

    fundo_coluna = fundos_colunas[0]

    normalizado = dados_fundo / dados_fundo.iloc[0]
    retorno_acumulado = normalizado - 1

    # Cálculo de retorno anualizado
    retorno_anualizado = (((dados_fundo.iloc[-1] / dados_fundo.iloc[0]) ** (365 / len(dados_fundo))) - 1) * 100

    # Gráfico de retorno acumulado
    fig_ret_acum = px.line(retorno_acumulado, title=None)
    fig_ret_acum.update_layout(
        xaxis_title="Data",
        yaxis_title="Retorno",
        xaxis=dict(
            tickmode='linear',
            dtick='M6',
            tickformat='%b-%y',
            tickangle=-45
        ), legend=dict(
            title=None,
            x=0.5,
            y=1.1,
            traceorder='normal',
            orientation='h',
            xanchor='center',
            yanchor='top',
            font=dict(
                family="Helvetica, sans-serif",
                size=12,
                color='black'
            )
        ),
        template='plotly_white',
        margin=dict(l=20, r=20, t=20, b=20)
    )
    for i, color in enumerate(colors):
        if i < len(fig_ret_acum.data):  # Garantir que há um traço para cada cor
            fig_ret_acum.data[i].update(line=dict(color=color))

    fig_ret_acum.add_shape(
        type="rect",
        x0=0, y0=0, x1=1, y1=1,
        xref="paper", yref="paper",
        line=dict(color="black", width=2)
    )

    pio.write_image(fig_ret_acum, 'fig_ret_acum.svg')

    # Gráfico de correlação
    correlacao = pct_change1.corr()

    # Melhorar os labels longos
    def wrap_labels(label):
        return '<br>'.join(textwrap.wrap(label, width=15))

    correlacao.columns = [wrap_labels(label) for label in correlacao.columns]
    correlacao.index = [wrap_labels(label) for label in correlacao.index]

    fig_corr = px.imshow(correlacao, text_auto='.2f', title=None,
                         color_continuous_scale=["#0063DE", "#F1CD1C", "#C00000"])

    fig_corr.update_layout(
        margin=dict(l=40, r=40, t=40, b=40),  # Ajustar margens para um padding adequado
        coloraxis_colorbar=dict(
            title="Correlação",
            tickvals=[0, 0.5, 1],
            ticktext=["Baixa", "Média", "Alta"],
            thickness=15,
            title_side="right",
            title_font=dict(size=12, family="Helvetica, sans-serif"),
            tickfont=dict(size=10, family="Helvetica, sans-serif"),
            ticks="outside"  # Mostrar ticks fora da barra de cores
        ),
        xaxis=dict(
            side='bottom',
            tickangle=-45,
            tickfont=dict(size=10, family="Helvetica, sans-serif")
        ),
        yaxis=dict(
            ticksuffix=" ",
            tickfont=dict(size=10, family="Helvetica, sans-serif")
        ),
        template='plotly_white',
        plot_bgcolor='rgba(0,0,0,0)',  # Deixar o fundo transparente
        paper_bgcolor='rgba(0,0,0,0)'  # Deixar o fundo do papel transparente
    )

    fig_corr.add_annotation(
        x=1.08, y=0,
        text="- Baixa",
        showarrow=False,
        font=dict(size=10, family="Helvetica, sans-serif"),
        xref="paper", yref="paper",
        yshift=2,
        xshift=11  # Mover o texto "Baixa" para a direita
    )

    pio.write_image(fig_corr, 'fig_corr.svg')

    # Gráfico de drawdown
    wea_idx = (pct_change1 + 1).cumprod()
    picos = wea_idx.cummax()
    drawdown = (wea_idx - picos) / picos

    fig_drawdown = go.Figure()
    for i, column in enumerate(drawdown.columns):
        fig_drawdown.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown[column],
            fill='tonexty',
            name=column,
            line=dict(width=0),  # Remove a linha
            fillcolor=colors[i % len(colors)]  # Preenchimento sólido
        ))
    fig_drawdown.update_layout(
        xaxis=dict(
            tickangle=-45
        ), legend=dict(
            title=None,
            x=0.5,
            y=1.1,
            traceorder='normal',
            orientation='h',
            xanchor='center',
            yanchor='top',
            font=dict(
                family="Helvetica, sans-serif",
                size=12,
                color='black'
            )
        ),
        template='plotly_white',
        margin=dict(l=20, r=20, t=20, b=20)
    )

    fig_drawdown.add_shape(
        type="rect",
        x0=0, y0=0, x1=1, y1=1,
        xref="paper", yref="paper",
        line=dict(color="black", width=2)
    )

    pio.write_image(fig_drawdown, 'fig_drawdown.svg')

    # Risco x Retorno
    desvpad = np.std(pct_change1, axis=0)
    vol_anual = (desvpad * np.sqrt(365)) * 100
    retorno_acumulado = ((dados_fundo.iloc[-1] / dados_fundo.iloc[0]) - 1)
    retorno_anualizado = (((dados_fundo.iloc[-1] / dados_fundo.iloc[0]) ** (365 / len(dados_fundo))) - 1) * 100

    vol_anual = vol_anual[retorno_anualizado.index]

    # Eficiência
    eficiencia = retorno_anualizado / vol_anual

    # Quebrar os textos longos
    nome_com_quebra = [textwrap.fill(label, width=15) for label in vol_anual.index]

    # Criar um DataFrame para facilitar a manipulação dos dados
    risco_retorno_df = pd.DataFrame({
        'Volatilidade Anual (%)': vol_anual,
        'Retorno Anualizado (%)': retorno_anualizado,
        'Nome': nome_com_quebra
    })

    # Função para determinar a posição do texto
    def determine_text_position(x, y, x_min, x_max, y_min, y_max):
        x_mid = (x_max + x_min) / 2
        y_mid = (y_max + y_min) / 2
        if x < x_mid:
            x_pos = 'right'
        else:
            x_pos = 'left'
        if y < y_mid:
            y_pos = 'top'
        else:
            y_pos = 'bottom'
        return f'{y_pos} {x_pos}'

    # Obter os valores mínimo e máximo dos eixos
    x_min, x_max = vol_anual.min(), vol_anual.max()
    y_min, y_max = retorno_anualizado.min(), retorno_anualizado.max()

    # Aplicar a função aos dados
    risco_retorno_df['textposition'] = risco_retorno_df.apply(
        lambda row: determine_text_position(row['Volatilidade Anual (%)'], row['Retorno Anualizado (%)'], x_min, x_max,
                                            y_min, y_max), axis=1
    )

    # Gráfico de scatter
    fig_risco_retorno = px.scatter(risco_retorno_df, x='Volatilidade Anual (%)', y='Retorno Anualizado (%)',
                                   color='Nome',
                                   text='Nome', title=None)

    # Atualizar as posições do texto
    fig_risco_retorno.update_traces(textposition=risco_retorno_df['textposition'])

    fig_risco_retorno.update_layout(
        xaxis_title='Risco (%)',
        yaxis_title='Retorno (%)',
        xaxis=dict(tickangle=-45),
        legend=dict(
            title=None,
            x=0.5,
            y=1.1,
            traceorder='normal',
            orientation='h',
            xanchor='center',
            yanchor='top',
            font=dict(
                family="Helvetica, sans-serif",
                size=12,
                color='black'
            )
        ),
        template='plotly_white',
        margin=dict(l=20, r=20, t=20, b=20)
    )

    fig_risco_retorno.add_shape(
        type="rect",
        x0=0, y0=0, x1=1, y1=1,
        xref="paper", yref="paper",
        line=dict(color="black", width=2)
    )

    pio.write_image(fig_risco_retorno, 'fig_risco_retorno.svg')

    # Janela Movel 3 anos
    if len(dados_fundo) > 252 * 3:
        retornos = dados_fundo.pct_change(1095)
        retornos_anualiz = (retornos + 1).pow(1 / 3)
        retornos_final = retornos_anualiz - 1

        # Identificar o primeiro índice válido
        first_valid_index_3 = retornos.first_valid_index()

        # Converter o índice válido para uma posição
        first_valid_position_3 = retornos.index.get_loc(first_valid_index_3)

        # Acessar a data correspondente no DataFrame original 'dados_fundo'
        first_valid_date_3 = dados_fundo.index[first_valid_position_3]

        fig_janela_3 = px.line(retornos_final, title=None)
        fig_janela_3.update_layout(xaxis_range=[first_valid_date_3, dados_fundo.index[-1]])
        fig_janela_3.update_layout(
            xaxis_title="Data",
            yaxis_title="Retorno",
            xaxis=dict(
                tickmode='linear',
                dtick='M6',
                tickformat='%b-%y',
                tickangle=-45
            ), legend=dict(
                title=None,
                x=0.5,
                y=1.1,
                traceorder='normal',
                orientation='h',
                xanchor='center',
                yanchor='top',
                font=dict(
                    family="Helvetica, sans-serif",
                    size=12,
                    color='black'
                )
            ),
            template='plotly_white',
            margin=dict(l=20, r=20, t=20, b=20)
        )
        for i, color in enumerate(colors):
            if i < len(fig_janela_3.data):  # Garantir que há um traço para cada cor
                fig_janela_3.data[i].update(line=dict(color=color))

        fig_janela_3.add_shape(
            type="rect",
            x0=0, y0=0, x1=1, y1=1,
            xref="paper", yref="paper",
            line=dict(color="black", width=2)
        )

        pio.write_image(fig_janela_3, 'fig_janela_3.svg')

    def plot_janelas_moveis_5_anos():
        if len(dados_fundo) > 252 * 5:
            retornos_1 = dados_fundo.pct_change(1825)
            retornos_anualiz_1 = (retornos_1 + 1).pow(1 / 5)
            retornos_final_1 = retornos_anualiz_1 - 1

            first_valid_index_5 = retornos_1.first_valid_index()
            first_valid_position_5 = retornos_1.index.get_loc(first_valid_index_5)
            first_valid_date_5 = dados_fundo.index[first_valid_position_5]

            fig = px.line(retornos_final_1, title=None)
            fig.update_layout(xaxis_range=[first_valid_date_5, dados_fundo.index[-1]])
            fig.update_layout(
                xaxis_title="Data",
                yaxis_title="Retorno",
                xaxis=dict(
                    tickmode='linear',
                    dtick='M6',
                    tickformat='%b-%y',
                    tickangle=-45
                ), legend=dict(
                    title=None,
                    x=0.5,
                    y=1.1,
                    traceorder='normal',
                    orientation='h',
                    xanchor='center',
                    yanchor='top',
                    font=dict(
                        family="Helvetica, sans-serif",
                        size=12,
                        color='black'
                    )
                ),
                template='plotly_white',
                margin=dict(l=20, r=20, t=20, b=20)
            )

            for i, color in enumerate(colors):
                if i < len(fig.data):  # Garantir que há um traço para cada cor
                    fig.data[i].update(line=dict(color=color))

            fig.add_shape(
                type="rect",
                x0=0, y0=0, x1=1, y1=1,
                xref="paper", yref="paper",
                line=dict(color="black", width=2)
            )

            return fig

    def plot_janelas_moveis_volatilidade():
        janelas_vol = pct_change1.rolling(12).std()
        janelas_vol_final = janelas_vol * np.sqrt(12)

        fig = px.line(janelas_vol_final, title=None)
        fig.update_layout(
            xaxis_title="Data",
            yaxis_title="Retorno",
            xaxis=dict(
                tickmode='linear',
                dtick='M6',
                tickformat='%b-%y',
                tickangle=-45
            ), legend=dict(
                title=None,
                x=0.5,
                y=1.1,
                traceorder='normal',
                orientation='h',
                xanchor='center',
                yanchor='top',
                font=dict(
                    family="Helvetica, sans-serif",
                    size=12,
                    color='black'
                )
            ),
            template='plotly_white',
            margin=dict(l=20, r=20, t=20, b=20)
        )
        for i, color in enumerate(colors):
            if i < len(fig.data):  # Garantir que há um traço para cada cor
                fig.data[i].update(line=dict(color=color))

        fig.add_shape(
            type="rect",
            x0=0, y0=0, x1=1, y1=1,
            xref="paper", yref="paper",
            line=dict(color="black", width=2)
        )

        return fig

    def plot_indice_sortino():
        downside = pct_change1[pct_change1 > 0]
        downside_vol = np.std(downside, axis=0)
        downside_vol = downside_vol[retorno_anualizado.index]
        indice_sortino = retorno_anualizado / downside_vol

        fig = px.scatter(x=downside_vol, y=retorno_anualizado, title=None)
        fig.update_xaxes(title_text='Downside (%)')
        fig.update_yaxes(title_text='Retorno (%)')
        fig.update_layout(
            xaxis=dict(
                tickangle=-45
            ), legend=dict(
                title=None,
                x=0.5,
                y=1.1,
                traceorder='normal',
                orientation='h',
                xanchor='center',
                yanchor='top',
                font=dict(
                    family="Helvetica, sans-serif",
                    size=12,
                    color='black'
                )
            ),
            template='plotly_white',
            margin=dict(l=20, r=20, t=20, b=20)
        )

        # Aplicar as cores usando update_traces
        for i, color in enumerate(colors):
            if i < len(fig.data):  # Garantir que há um traço para cada cor
                fig.data[i].update(marker=dict(color=color))

        fig.add_shape(
            type="rect",
            x0=0, y0=0, x1=1, y1=1,
            xref="paper", yref="paper",
            line=dict(color="black", width=2)
        )

        return fig

    # Função para calcular a correlação móvel
    def calcular_correlacao_movel(dados_fundo, fundo_coluna, benchmarks_colunas, janela):
        # Calcular os retornos diários
        retornos = dados_fundo[[fundo_coluna] + benchmarks_colunas].pct_change()

        # Inicializar um DataFrame para armazenar as correlações móveis
        correlacoes_movel = pd.DataFrame(index=retornos.index, columns=benchmarks_colunas)

        # Calcular a correlação móvel para cada benchmark usando operações vetorizadas
        for benchmark in benchmarks_colunas:
            # Calcular a correlação móvel
            correlacoes_movel[benchmark] = retornos[fundo_coluna].rolling(window=janela).corr(retornos[benchmark])

        # Remover os valores nulos
        correlacoes_movel = correlacoes_movel.dropna()

        # Renomear as colunas para fundo_especifico vs benchmark
        correlacoes_movel.columns = [f"{fundo_coluna} vs {benchmark}" for benchmark in benchmarks_colunas]

        return correlacoes_movel

    # Função para criar o gráfico de correlação móvel
    def plot_correlacao_movel(correlacoes, janela):
        fig = px.line(correlacoes, title=None)
        fig.update_layout(
            xaxis_title="Data",
            yaxis_title="Correlação",
            xaxis=dict(
                tickmode='linear',
                dtick='M6',
                tickformat='%b-%y',
                tickangle=-45
            ), legend=dict(
                title=None,
                x=0.5,
                y=1.1,
                traceorder='normal',
                orientation='h',
                xanchor='center',
                yanchor='top',
                font=dict(
                    family="Helvetica, sans-serif",
                    size=12,
                    color='black'
                )
            ),
            template='plotly_white',
            margin=dict(l=20, r=20, t=20, b=20)
        )
        return fig

    # Função para calcular a volatilidade móvel
    def calcular_volatilidade_movel(dados_fundo, fundo_coluna, benchmarks_colunas, janela):
        # Calcular os retornos diários
        retornos = dados_fundo[[fundo_coluna] + benchmarks_colunas].pct_change()

        # Calcular a volatilidade móvel anualizada
        volatilidade_movel = retornos.rolling(window=janela).std() * np.sqrt(252)

        # Remover os valores nulos
        volatilidade_movel = volatilidade_movel.dropna()

        return volatilidade_movel

    # Função para criar o gráfico de volatilidade móvel
    def plot_volatilidade_movel(volatilidade, janela):
        fig = px.line(volatilidade, title=None)
        fig.update_traces(mode='lines', line_shape='linear')  # Definir o modo de renderização e forma da linha
        fig.update_layout(
            xaxis_title="Data",
            yaxis_title="Volatilidade Anualizada",
            xaxis=dict(
                tickmode='linear',
                dtick='M6',
                tickformat='%b-%y',
                tickangle=-45
            ), legend=dict(
                title=None,
                x=0.5,
                y=1.1,
                traceorder='normal',
                orientation='h',
                xanchor='center',
                yanchor='top',
                font=dict(
                    family="Helvetica, sans-serif",
                    size=12,
                    color='black'
                )
            ),
            template='plotly_white',
            margin=dict(l=20, r=20, t=20, b=20)
        )
        return fig

    # Função para calcular o retorno acumulado anual
    def calcular_retorno_acumulado_anual(dados_fundo, fundo_coluna, benchmarks_colunas):
        # Calcular os retornos diários
        retornos_diarios = dados_fundo[[fundo_coluna] + benchmarks_colunas].pct_change()

        # Resample para obter os retornos anuais acumulados
        retornos_anuais = (retornos_diarios + 1).resample('YE').prod() - 1

        # Filtrar para os últimos 5 anos
        retornos_anuais_5anos = retornos_anuais.loc[retornos_anuais.index.year >= (datetime.now().year - 5)]

        return retornos_anuais_5anos

    # Função para criar o gráfico de retorno acumulado anual
    def plot_retorno_acumulado_anual(retornos_anuais):
        # Redefinir o índice e adicionar a coluna de anos
        retornos_anuais['Ano'] = retornos_anuais.index.year.astype(str)

        # Derreter o DataFrame para long format
        retornos_anuais_melted = retornos_anuais.melt(id_vars=['Ano'], var_name='Categoria', value_name='Retorno')

        # Criar o gráfico de barras
        fig = px.bar(retornos_anuais_melted, x='Ano', y='Retorno', color='Categoria', barmode='group')

        fig.update_layout(
            xaxis_title="Ano",
            yaxis_title="Retorno Acumulado",
            xaxis=dict(
                tickmode='linear',
                dtick='Y1',
                tickformat='%Y',
            ), legend=dict(
                title=None,
                x=0.5,
                y=1.1,
                traceorder='normal',
                orientation='h',
                xanchor='center',
                yanchor='top',
                font=dict(
                    family="Helvetica, sans-serif",
                    size=12,
                    color='black'
                )
            ),
            template='plotly_white',
            margin=dict(l=20, r=20, t=20, b=20)
        )
        return fig

    def calcular_beta_movel(dados_fundo, fundo_coluna, benchmarks_colunas, janela):
        # Calcular os retornos diários
        retornos = dados_fundo[[fundo_coluna] + benchmarks_colunas].pct_change().dropna()

        # Inicializar um DataFrame para armazenar os betas móveis
        betas_movel = pd.DataFrame(index=retornos.index, columns=benchmarks_colunas)

        # Calcular o beta móvel para cada benchmark usando operações vetorizadas
        for benchmark in benchmarks_colunas:
            # Calcular a covariância móvel
            cov_movel = retornos[fundo_coluna].rolling(window=janela).cov(retornos[benchmark])

            # Calcular a variância móvel
            var_movel = retornos[benchmark].rolling(window=janela).var()

            # Calcular o beta móvel
            betas_movel[benchmark] = cov_movel / var_movel

        # Remover os valores nulos
        betas_movel = betas_movel.dropna()

        # Renomear as colunas para fundo_especifico vs benchmark
        betas_movel.columns = [f"{fundo_coluna} vs {benchmark}" for benchmark in benchmarks_colunas]

        return betas_movel

    def plot_beta_movel(betas, janela):
        fig = px.line(betas, title=None)
        fig.update_layout(
            xaxis_title="Data",
            yaxis_title="Beta",
            xaxis=dict(
                tickmode='linear',
                dtick='M6',
                tickformat='%b-%y',
                tickangle=-45
            ), legend=dict(
                title=None,
                x=0.5,
                y=1.1,
                traceorder='normal',
                orientation='h',
                xanchor='center',
                yanchor='top',
                font=dict(
                    family="Helvetica, sans-serif",
                    size=12,
                    color='black'
                )
            ),
            template='plotly_white',
            margin=dict(l=20, r=20, t=20, b=20)
        )
        return fig

    def calcular_beta_inicio(dados_fundo, fundo_coluna, benchmarks_colunas):
        # Calcular os retornos diários
        retornos = dados_fundo[[fundo_coluna] + benchmarks_colunas].pct_change().dropna()

        # Inicializar um dicionário para armazenar os betas
        betas = {}

        # Calcular o beta para cada benchmark
        for benchmark in benchmarks_colunas:
            # Calcular a covariância entre o fundo e o benchmark
            cov = retornos[fundo_coluna].cov(retornos[benchmark])

            # Calcular a variância do benchmark
            var = retornos[benchmark].var()

            # Calcular o beta
            beta = cov / var

            # Armazenar o beta calculado
            betas[benchmark] = beta

        return betas

    def criar_tabela_beta(betas, fundo_coluna):
        # Criar um DataFrame com os betas
        df_betas = pd.DataFrame(betas, index=[f'Beta desde o início: {fundo_coluna}']).T
        df_betas.columns = ['Beta']

        # Criar a tabela usando pn.pane.DataFrame com sizing_mode="stretch_both"
        tabela = pn.pane.DataFrame(df_betas, width=400, height=200)

        return tabela

    # Cálculo e gráfico de correlação móvel de 12 meses
    correlacoes_12m = calcular_correlacao_movel(dados_fundo, fundo_coluna, benchmarks_colunas, 252)
    fig_correlacao_12m = plot_correlacao_movel(correlacoes_12m, 12)
    # pio.write_image(fig_correlacao_12m, 'fig_correlacao_12m.svg')

    # Cálculo e gráfico de correlação móvel de 3 meses
    correlacoes_3m = calcular_correlacao_movel(dados_fundo, fundo_coluna, benchmarks_colunas, 63)
    fig_correlacao_3m = plot_correlacao_movel(correlacoes_3m, 3)
    # pio.write_image(fig_correlacao_3m, 'fig_correlacao_3m.svg')

    # Cálculo e gráfico de volatilidade móvel de 3 meses
    volatilidade_3m = calcular_volatilidade_movel(dados_fundo, fundo_coluna, benchmarks_colunas, 63)
    fig_volatilidade_3m = plot_volatilidade_movel(volatilidade_3m, 3)
    # pio.write_image(fig_volatilidade_3m, 'volatilidade_movel_3m.svg')

    # Cálculo e gráfico de volatilidade móvel de 6 meses
    volatilidade_6m = calcular_volatilidade_movel(dados_fundo, fundo_coluna, benchmarks_colunas, 126)
    fig_volatilidade_6m = plot_volatilidade_movel(volatilidade_6m, 6)
    # pio.write_image(fig_volatilidade_6m, 'volatilidade_movel_6m.svg')

    # Cálculo e gráfico de volatilidade móvel de 12 meses
    volatilidade_12m = calcular_volatilidade_movel(dados_fundo, fundo_coluna, benchmarks_colunas, 252)
    fig_volatilidade_12m = plot_volatilidade_movel(volatilidade_12m, 12)
    # pio.write_image(fig_volatilidade_12m, 'volatilidade_movel_12m.svg')

    # Cálculo e gráfico de retorno acumulado anual dos últimos 5 anos
    retornos_anuais = calcular_retorno_acumulado_anual(dados_fundo, fundo_coluna, benchmarks_colunas)
    fig_retorno_anuais = plot_retorno_acumulado_anual(retornos_anuais)
    # pio.write_image(fig_retorno_anuais, 'fig_retorno_anuais.svg')

    # Cálculo e gráfico de beta móvel de 12 meses
    beta_3m = calcular_beta_movel(dados_fundo, fundo_coluna, benchmarks_colunas, 63)
    fig_beta_3m = plot_beta_movel(beta_3m, 3)
    # pio.write_image(fig_beta_3m, 'fig_beta_3m.svg')

    beta_6m = calcular_beta_movel(dados_fundo, fundo_coluna, benchmarks_colunas, 126)
    fig_beta_6m = plot_beta_movel(beta_6m, 6)
    # pio.write_image(fig_beta_6m, 'fig_beta_6m.png')

    beta_12m = calcular_beta_movel(dados_fundo, fundo_coluna, benchmarks_colunas, 252)
    fig_beta_12m = plot_beta_movel(beta_12m, 12)
    # pio.write_image(fig_beta_12m, 'fig_beta_12m.png')

    # Cálculo do beta desde o início
    betas_inicio = calcular_beta_inicio(dados_fundo, fundo_coluna, benchmarks_colunas)

    # Criação da tabela com os valores de beta
    tabela_beta_inicio = criar_tabela_beta(betas_inicio, fundo_coluna)

    pane_ret_acum = pn.pane.Plotly(fig_ret_acum, width=666, height=420)

    pane_drawdown = pn.pane.Plotly(fig_drawdown, width=666, height=420)

    pane_corr = pn.pane.Plotly(fig_corr, width=666, height=420)

    pane_risco_retorno = pn.pane.Plotly(fig_risco_retorno, width=666, height=420)

    if len(dados_fundo) > 252 * 3:
        #pio.write_image(fig_janela_3, 'fig_janela_3.svg')
        pane_janelas_moveis_3_anos = pn.pane.Plotly(fig_janela_3, width=666, height=420)
    else:
        pane_janelas_moveis_3_anos = None

    if len(dados_fundo) > 252 * 5:
        pane_janelas_moveis_5_anos = pn.pane.Plotly(plot_janelas_moveis_5_anos(), width=666, height=420)
    else:
        pane_janelas_moveis_5_anos = None

    pane_janelas_moveis_volatilidade = pn.pane.Plotly(plot_janelas_moveis_volatilidade(), width=666, height=420)

    pane_indice_sortino = pn.pane.Plotly(plot_indice_sortino(), width=666, height=420)

    pane_correlacao_12m = pn.pane.Plotly(fig_correlacao_12m, width=666, height=420)

    pane_correlacao_3m = pn.pane.Plotly(fig_correlacao_3m, width=666, height=420)

    pane_volatilidade_3m = pn.pane.Plotly(fig_volatilidade_3m, width=666, height=420)

    pane_volatilidade_6m = pn.pane.Plotly(fig_volatilidade_6m, width=666, height=420)

    pane_volatilidade_12m = pn.pane.Plotly(fig_volatilidade_12m, width=666, height=420)

    pane_retorno_anuais = pn.pane.Plotly(fig_retorno_anuais, width=666, height=420)

    pane_beta_3m = pn.pane.Plotly(fig_beta_3m, width=666, height=420)

    pane_beta_6m = pn.pane.Plotly(fig_beta_6m, width=666, height=420)

    pane_beta_12m = pn.pane.Plotly(fig_beta_12m, width=666, height=420)

    # Métricas de Retorno
    rows = [
        pn.pane.Markdown('## Métricas de Retorno'),
        pn.Row(
            pn.Column(
                pn.Card(pane_ret_acum, title="Retorno Acumulado", collapsed=True)
            ),
            pn.Column(
                pn.Card(pane_retorno_anuais, title="Retorno Acumulado Anual (Últimos 5 Anos)", collapsed=True)
            )
        ),
        pn.Row(
            pn.Column(
                pn.Card(pane_janelas_moveis_3_anos, title="Janela Móvel de Retorno - 3 Anos", collapsed=True)
            ),
            pn.Column(
                pn.Card(pane_janelas_moveis_5_anos, title="Janela Móvel de Retorno - 5 Anos", collapsed=True)
            )
        ),
        pn.Row(
            pn.Column(
                pn.Card(pane_risco_retorno, title="Risco x Retorno", collapsed=True)
            ),
            pn.Column(
                pn.Card(pane_indice_sortino, title="Índice de Sortino", collapsed=True)
            )
        ),
        pn.Row(
            pn.Column(
                pn.Card(pane_drawdown, title="Drawdown", collapsed=True)
            )
        ),

        pn.Spacer(height=20),

        # Métricas de Volatilidade
        pn.pane.Markdown('## Métricas de Volatilidade'),
        pn.Row(
            pn.Column(
                pn.Card(pane_janelas_moveis_volatilidade, title="Janela Móvel de Volatilidade", collapsed=True)
            ),
            pn.Column(
                pn.Card(pane_volatilidade_3m, title="Volatilidade Móvel (3 Meses)", collapsed=True)
            )
        ),
        pn.Row(
            pn.Column(
                pn.Card(pane_volatilidade_6m, title="Volatilidade Móvel (6 Meses)", collapsed=True)
            ),
            pn.Column(
                pn.Card(pane_volatilidade_12m, title="Volatilidade Móvel (12 Meses)", collapsed=True)
            )
        ),

        pn.Spacer(height=20),

        # Métricas de Correlação
        pn.pane.Markdown('## Métricas de Correlação'),
        pn.Row(
            pn.Column(
                pn.Card(pane_corr, title="Matriz de Correlação", collapsed=True)
            ),
            pn.Column(
                pn.Card(pane_correlacao_12m, title="Correlação Móvel (12 Meses)", collapsed=True)
            )
        ),
        pn.Row(
            pn.Column(
                pn.Card(pane_correlacao_3m, title="Correlação Móvel (3 Meses)", collapsed=True)
            )
        ),

        pn.Spacer(height=20),

        # Métricas de Beta
        pn.pane.Markdown('## Métricas de Beta'),
        pn.Row(
            pn.Column(
                pn.Card(pane_beta_3m, title="Beta Móvel (3 Meses)", collapsed=True)
            ),
            pn.Column(
                pn.Card(pane_beta_6m, title="Beta Móvel (6 Meses)", collapsed=True)
            )
        ),
        pn.Row(
            pn.Column(
                pn.Card(pane_beta_12m, title="Beta Móvel (12 Meses)", collapsed=True)
            ),
            pn.Column(
                pn.Card(tabela_beta_inicio, title="Beta Desde o Início", collapsed=True)
            )
        ),
    ]

    home = pn.Column(
        pn.pane.Markdown(f"# Análise do Fundo: {fundo_especifico}"),
        *rows,
    )

    # Retornar os gráficos no layout do panel
    return home

def comment_page(fundo_especifico):

    first_column = comentarios.columns[0]
    comentarios[first_column] = pd.to_datetime(comentarios[first_column])
    comentarios.sort_values(by=first_column, ascending=False, inplace=True)
    comentarios[first_column] = pd.to_datetime(comentarios[first_column]).dt.strftime('%B %Y')
    comentarios[fundo_especifico] = comentarios[fundo_especifico].str.replace('\n', ' ')
    comentarios_filtrados = comentarios[[first_column, fundo_especifico]]

    multi_choice = pn.widgets.MultiChoice(name='Selecione as Datas',
                                          options=list(comentarios_filtrados[first_column].unique()))

    # Função para atualizar o DataFrame com base nas datas selecionadas
    def update_df(selected_dates):
        if not selected_dates:
            filtered_df = comentarios_filtrados
        else:
            filtered_df = comentarios_filtrados[comentarios_filtrados[first_column].isin(selected_dates)]

        # Converter o DataFrame filtrado para HTML com estilos CSS
        html = filtered_df.to_html(escape=False, index=False)

        # Incorporar CSS diretamente no HTML
        html = f"""
        <style>
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 12px;
            border: 1px solid #ddd;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
            font-weight: bold;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        a {{
            color: #3498db;
            text-decoration: none;
        }}
        a:hover {{
            text-decoration: underline;
        }}
        .bk-data-table td {{
            text-align: justify;
            white-space: pre-wrap; /* Para manter as quebras de linha */
        }}
        </style>
        {html}
        """
        return pn.pane.HTML(html, sizing_mode='stretch_width')

    # Vincular a função update_df ao widget MultiChoice
    df_pane = pn.bind(update_df, multi_choice.param.value)

    # Retornar a coluna com o CSS, o widget MultiChoice e o DataFrame renderizado
    return pn.Column("# Comentários Mensais", multi_choice, df_pane)


def make_clickable(val):
    if isinstance(val, str) and (val.startswith('http://') or val.startswith('https://')):
        return f'<a href="{val}" target="_blank">{val}</a>'
    else:
        return val.replace('\n', '<br>') if isinstance(val, str) else val


def ideas_page(fundo_especifico):
    # Nome da primeira coluna
    first_column = ideias.columns[0]

    # Converter a coluna de data para datetime
    ideias[first_column] = pd.to_datetime(ideias[first_column])

    # Ordenar o DataFrame pela coluna de data (mais recente primeiro)
    ideias.sort_values(by=first_column, ascending=False, inplace=True)

    # Formatar a coluna de data para exibição
    ideias[first_column] = ideias[first_column].dt.strftime('%B %Y')

    # Substituir '\n' por '<br>' para manter quebras de linha e aplicar a função make_clickable
    ideias[fundo_especifico] = ideias[fundo_especifico].apply(make_clickable)

    # Filtrar o DataFrame para as colunas desejadas
    ideias_filtrados = ideias[[first_column, fundo_especifico]]

    # Criar o widget MultiChoice
    multi_choice = pn.widgets.MultiChoice(name='Selecione as Datas',
                                          options=list(ideias_filtrados[first_column].unique()))

    # Função para atualizar o DataFrame com base nas datas selecionadas
    def update_df(selected_dates):
        if not selected_dates:
            filtered_df = ideias_filtrados
        else:
            filtered_df = ideias_filtrados[ideias_filtrados[first_column].isin(selected_dates)]

        # Converter o DataFrame filtrado para HTML com estilos CSS
        html = filtered_df.to_html(escape=False, index=False)

        # Incorporar CSS diretamente no HTML
        html = f"""
        <style>
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 12px;
            border: 1px solid #ddd;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
            font-weight: bold;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        a {{
            color: #3498db;
            text-decoration: none;
        }}
        a:hover {{
            text-decoration: underline;
        }}
        .bk-data-table td {{
            text-align: justify;
            white-space: pre-wrap; /* Para manter as quebras de linha */
        }}
        </style>
        {html}
        """
        return pn.pane.HTML(html, sizing_mode='stretch_width')

    # Vincular a função update_df ao widget MultiChoice
    df_pane = pn.bind(update_df, multi_choice.param.value)

    # Retornar a coluna com o CSS, o widget MultiChoice e o DataFrame renderizado
    return pn.Column("# Ideias", multi_choice, df_pane)


def extract_pages(pdf_path, page_numbers):
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num in page_numbers:
            page = pdf.pages[page_num - 1]
            pages.append(page)
    return pages


# Função para converter páginas em imagens
def pages_to_images(pages):
    images = []
    for page in pages:
        pil_image = page.to_image()
        img_byte_arr = io.BytesIO()
        pil_image.original.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        image = Image.open(img_byte_arr)
        images.append(image)
    return images


# Função para exibir páginas extraídas no Panel, duas imagens por row
def display_pdf_pages(pdf_path, page_numbers, scale_factor=0.5):
    pages = extract_pages(pdf_path, page_numbers)
    images = pages_to_images(pages)

    # Ajustar tamanho das imagens usando o fator de escala
    pn_images = []
    for image in images:
        width, height = image.size
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        pn_image = pn.pane.PNG(image, width=new_width, height=new_height)
        pn_images.append(pn_image)

    # Criar layout com duas imagens por row
    rows = []
    for i in range(0, len(pn_images), 2):
        rows.append(pn.Row(*pn_images[i:i + 2]))

    return pn.Column(*rows)

# Exemplo de uso
# pdf_path = "C:\\Users\\JoãoCampos\\Documents\\Summer Job _ Detalhes\\202406 GAMA - Cenário Global.pdf"  # Caminho para o seu arquivo PDF
# page_numbers = [1, 2, 3, 4, 5]  # Páginas que você quer extrair e exibir
# pdf_display=display_pdf_pages(pdf_path, page_numbers)

def summary_page(fundo_especifico):
    return pn.Column('# Resumo',
                     pdf_display)


# Create a dictionary to store the pages
pages = {
    'Métricas': home_page,
    'Comentários': comment_page,
    'Ideias': ideas_page,
    'Resumo': summary_page
}

# Define the navigation widget
navigation = pn.widgets.RadioButtonGroup(name='Navigation', options=['Métricas', 'Comentários', 'Ideias', 'Resumo'], button_type='primary', value='Métricas')
print(comentarios)

# Function to return the selected page content
def get_page_content(navigation_value, fundo_value, periodo_value):
    if navigation_value == 'Métricas':
        return home_page(fundo_value.strip(), periodo_value)
    else:
        return pages[navigation_value](fundo_value.strip())

# Adicionar um seletor de período
period_select = pn.widgets.Select(name='Selecione o Período', options={
    'Último 1 mês': '1M',
    'Últimos 3 meses': '3M',
    'Últimos 6 meses': '6M',
    'Últimos 12 meses': '12M',
    'YTD': 'YTD',
    'Últimos 5 anos': '5Y',
    'Tudo': 'ALL'
}, value='ALL')

# Seletor de fundos

fundo_select = pn.widgets.Select(name='Selecione o Fundo', options=list(fundos.columns.unique()))

# Bind the navigation value to the content function
page_content = pn.bind(get_page_content, navigation.param.value, fundo_select.param.value, period_select.param.value)

# HTML for the header with a logo and title
header_html = """
<div style="display: flex; align-items: center; height: 150px;">
    <img src="assets/image.png" alt="Logo" style="height: 45px; margin-right: 40px;">
    <h1>Dashboard de Métricas</h1>
</div>
"""

# Create a template and add the HTML header
template = pn.template.MaterialTemplate(
    title='',
    header_background='#2a2a5d',
    sidebar_width=400,
)
template.header.append(pn.pane.HTML(header_html))

# Add the navigation and page content to the template
template.sidebar.append(pn.pane.Markdown("## Navegador"))
template.sidebar.append(navigation)
template.sidebar.append(pn.Spacer(height=5))
template.sidebar.append(pn.pane.Markdown("## Selecione o Fundo para Análise"))
template.sidebar.append(fundo_select)
template.sidebar.append(pn.pane.Markdown("## Selecione o Período para Análise"))
template.sidebar.append(period_select)
template.main.append(page_content)


# Função de carregamento
@pn.state.onload
def carregar_graficos():
    template.main.clear()
    template.main.append(page_content)


# Setup static directory for assets
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

# Copy the logo image to the static directory
shutil.copy('image.png', static_dir)

# Serve the dashboard
pn.serve(template, show=True, static_dirs={'/assets': static_dir})
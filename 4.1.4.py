import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

# Caminho do arquivo CSV
caminho_arquivo = r'C:\Users\ruirm\OneDrive\Ambiente de Trabalho\ANADI_projeto\AIRPOL_data.csv'

# Carregar os dados do CSV
dados = pd.read_csv(caminho_arquivo, sep=';')

# Função para converter strings para números
def converter_para_float(valor):
    if isinstance(valor, str):
        valor = valor.strip().replace(',', '.')
        try:
            return float(valor)
        except ValueError:
            return np.nan
    return valor

# Converter coluna 'Value' para float
dados['Value'] = dados['Value'].apply(converter_para_float)

# Normalizar colunas para evitar problemas de maiúsculas/minúsculas e espaços extras
dados['Country'] = dados['Country'].str.strip().str.title()
dados['Outcome'] = dados['Outcome'].str.strip().str.upper()

# Filtrar apenas os dados referentes a STROKE para os países desejados
paises = ["Spain", "France", "Italy", "Greece"]
dados_filtrados = dados[(dados['Country'].isin(paises)) & (dados['Outcome'] == 'STROKE')]

# Verificar se há dados filtrados
print("Valores únicos em 'Outcome':", dados['Outcome'].unique())
print("Valores únicos em 'Country':", dados['Country'].unique())
print(f"Número de linhas após filtragem: {dados_filtrados.shape[0]}")

# Calcular estatísticas
dados_estatisticas = []
for pais in paises:
    df_pais = dados_filtrados[dados_filtrados['Country'] == pais]['Value'].dropna()
    
    if df_pais.empty:  # Se não houver valores, pular o país
        print(f"Aviso: Nenhum dado disponível para {pais}")
        continue
    
    media = np.mean(df_pais)
    q1 = np.percentile(df_pais, 25)
    mediana = np.percentile(df_pais, 50)
    q3 = np.percentile(df_pais, 75)
    desvio_padrao = np.std(df_pais, ddof=1)
    assimetria = skew(df_pais, nan_policy='omit')
    curtose_val = kurtosis(df_pais, nan_policy='omit')
    
    dados_estatisticas.append([pais, media, q1, mediana, q3, desvio_padrao, assimetria, curtose_val])

# Criar DataFrame com os resultados
df_resultado = pd.DataFrame(
    dados_estatisticas, 
    columns=["País", "Média", "Q1", "Mediana", "Q3", "Desvio Padrão", "Assimetria", "Curtose"]
)

# Arredondar para 4 casas decimais
df_resultado = df_resultado.round(4)

# Exibir a tabela
print("\nTabela de Estatísticas:")
print(df_resultado)

# Criar gráfico de barras com a média (com valores acima das barras, removendo 0.0000)
plt.figure(figsize=(12, 6))
ax = sns.barplot(x=df_resultado["País"], y=df_resultado["Média"], palette="viridis")
plt.title("Média de Mortes Prematuras por STROKE")
plt.xlabel("País")
plt.ylabel("Média de Mortes")

# Adicionar valores acima das barras, evitando valores próximos de zero
for p in ax.patches:
    height = p.get_height()
    if not np.isnan(height) and abs(height) > 0.001:  # Ignorar valores menores que 0.001
        ax.annotate(f'{height:.4f}', 
                    (p.get_x() + p.get_width() / 2., height), 
                    ha='center', va='bottom', fontsize=10, color='black', xytext=(0, 5), 
                    textcoords='offset points')
plt.show()

# Preparar dados para o gráfico de barras agrupadas
df_melted = df_resultado.melt(id_vars=["País"], 
                              value_vars=["Média", "Q1", "Mediana", "Q3", "Desvio Padrão", "Assimetria", "Curtose"],
                              var_name="Estatística", 
                              value_name="Valor")

# Criar gráfico de barras agrupadas (com valores acima das barras e mais espaçamento)
plt.figure(figsize=(16, 10))  # Aumentar o tamanho para melhor espaçamento
ax = sns.barplot(x="País", y="Valor", hue="Estatística", data=df_melted, palette="muted", dodge=True)
plt.title("Estatísticas do Número de Mortes Prematuras por STROKE por País")
plt.xlabel("País")
plt.ylabel("Valor")
plt.legend(title="Estatística", bbox_to_anchor=(1.05, 1), loc='upper left')

# Ajustar o espaçamento interno entre barras
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)  # Manter rótulos horizontais
for p in ax.patches:
    height = p.get_height()
    if not np.isnan(height):  # Verificar se o valor não é NaN
        ax.annotate(f'{height:.4f}', 
                    (p.get_x() + p.get_width() / 2., height), 
                    ha='center', va='bottom', fontsize=8, color='black', xytext=(0, 5), 
                    textcoords='offset points')

plt.tight_layout()
plt.show()
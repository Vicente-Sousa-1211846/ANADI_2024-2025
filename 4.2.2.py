import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

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

# Verificar valores únicos para entender os dados
print("Valores únicos em 'Outcome':", dados['Outcome'].unique())
print("Valores únicos em 'Country':", dados['Country'].unique())

# Filtrar dados de Portugal e Albânia
dados_portugal = dados[dados['Country'] == 'Portugal']['Value'].dropna()
dados_albania = dados[dados['Country'] == 'Albania']['Value'].dropna()

# Verificar se há dados disponíveis
print(f"Número de observações para Portugal: {dados_portugal.shape[0]}")
print(f"Número de observações para Albânia: {dados_albania.shape[0]}")

# Calcular estatísticas descritivas
estatisticas = []

# Estatísticas para Portugal
if not dados_portugal.empty:
    media_pt = np.mean(dados_portugal)
    q1_pt = np.percentile(dados_portugal, 25)
    mediana_pt = np.percentile(dados_portugal, 50)
    q3_pt = np.percentile(dados_portugal, 75)
    desvio_padrao_pt = np.std(dados_portugal, ddof=1)
    estatisticas.append(['Portugal', media_pt, q1_pt, mediana_pt, q3_pt, desvio_padrao_pt])
else:
    print("Aviso: Nenhum dado disponível para Portugal")

# Estatísticas para Albânia
if not dados_albania.empty:
    media_al = np.mean(dados_albania)
    q1_al = np.percentile(dados_albania, 25)
    mediana_al = np.percentile(dados_albania, 50)
    q3_al = np.percentile(dados_albania, 75)
    desvio_padrao_al = np.std(dados_albania, ddof=1)
    estatisticas.append(['Albania', media_al, q1_al, mediana_al, q3_al, desvio_padrao_al])
else:
    print("Aviso: Nenhum dado disponível para Albânia")

# Criar DataFrame com os resultados
df_estatisticas = pd.DataFrame(
    estatisticas, 
    columns=["País", "Média", "Q1", "Mediana", "Q3", "Desvio Padrão"]
)

# Arredondar para 4 casas decimais
df_estatisticas = df_estatisticas.round(4)

# Exibir a tabela de estatísticas
print("\nTabela de Estatísticas:")
print(df_estatisticas)

# Realizar teste t independente (one-tailed)
# H0: μPortugal ≥ μAlbania
# H1: μPortugal < μAlbania
if not dados_portugal.empty and not dados_albania.empty:
    t_stat, p_value = ttest_ind(dados_portugal, dados_albania, alternative='less')
    print("\nResultados do teste t:")
    print(f"Estatística t: {t_stat:.4f}")
    print(f"Valor p: {p_value:.4f}")
    
    # Interpretar os resultados
    alpha = 0.05
    if p_value < alpha:
        conclusao = f"Como p-value ({p_value:.4f}) < alpha ({alpha}), rejeitamos a hipótese nula."
        conclusao += "\nConclusão: Há evidência estatística de que o nível médio de poluição em Portugal é inferior ao da Albânia."
    else:
        conclusao = f"Como p-value ({p_value:.4f}) >= alpha ({alpha}), não rejeitamos a hipótese nula."
        conclusao += "\nConclusão: Não há evidência estatística suficiente para afirmar que o nível médio de poluição em Portugal é inferior ao da Albânia."
    
    print(conclusao)
else:
    print("Não foi possível realizar o teste estatístico devido à falta de dados.")

# Criar um gráfico de barras comparativo de estatísticas
plt.figure(figsize=(14, 8))

# Preparar dados para o gráfico de barras
df_melted = df_estatisticas.melt(id_vars=["País"], 
                                  value_vars=["Média", "Mediana", "Desvio Padrão"],
                                  var_name="Estatística", 
                                  value_name="Valor")

# Criar gráfico de barras agrupadas
ax = sns.barplot(x="País", y="Valor", hue="Estatística", data=df_melted, palette="viridis")

# Adicionar valores acima das barras
for p in ax.patches:
    height = p.get_height()
    if not np.isnan(height):
        ax.annotate(f'{height:.4f}', 
                   (p.get_x() + p.get_width() / 2., height), 
                   ha='center', va='bottom', fontsize=10, color='black', xytext=(0, 5), 
                   textcoords='offset points')

# Adicionar título e rótulos
plt.title("Comparação dos Níveis de Poluição Atmosférica", fontsize=16, fontweight='bold', pad=20)
plt.xlabel("País", fontsize=14, labelpad=15)
plt.ylabel("Valor", fontsize=14, labelpad=15)
plt.legend(title="Estatística", fontsize=12)
plt.grid(True, axis='y', linestyle='--', alpha=0.5)

# Adicionar os resultados do teste t no gráfico
if 't_stat' in locals() and 'p_value' in locals():
    resultado = f"Teste t: {t_stat:.4f} | p-value: {p_value:.4f}"
    if p_value < alpha:
        resultado += "\nPortugal tem poluição significativamente menor (p<0.05)"
    else:
        resultado += "\nDiferença não significativa estatisticamente (p≥0.05)"
    
    plt.figtext(0.5, 0.01, resultado, ha='center', fontsize=12, 
                bbox=dict(facecolor='lightgray', alpha=0.5, boxstyle='round,pad=0.5'))

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.show()

# Criar gráfico de barras só com a média (versão simplificada)
plt.figure(figsize=(12, 6))
ax = sns.barplot(x=df_estatisticas["País"], y=df_estatisticas["Média"], palette="viridis")
plt.title("Média dos Níveis de Poluição Atmosférica", fontsize=14)
plt.xlabel("País", fontsize=12)
plt.ylabel("Média", fontsize=12)

# Adicionar valores acima das barras
for p in ax.patches:
    height = p.get_height()
    if not np.isnan(height):
        ax.annotate(f'{height:.4f}', 
                   (p.get_x() + p.get_width() / 2., height), 
                   ha='center', va='bottom', fontsize=10, color='black', xytext=(0, 5), 
                   textcoords='offset points')

# Adicionar resultado do teste t como texto
if 't_stat' in locals() and 'p_value' in locals():
    plt.figtext(0.5, 0.01, f"Teste t: {t_stat:.4f} | p-value: {p_value:.4f} (α=0.05)", 
                ha='center', fontsize=10)

plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.show()
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Carregar os dados do CSV usando ponto e vírgula como separador
caminho_arquivo = r'C:\Users\ruirm\OneDrive\Ambiente de Trabalho\ANADI_projeto\AIRPOL_data.csv'
dados = pd.read_csv(caminho_arquivo, sep=';')

# Função para converter strings para números
def converter_para_float(valor):
    if isinstance(valor, str):
        # Remover espaços
        valor = valor.strip()
        # Substituir vírgulas por pontos
        valor = valor.replace(',', '.')
        try:
            return float(valor)
        except ValueError:
            return np.nan
    return valor

# Converter coluna 'Value' para float
dados['Value'] = dados['Value'].apply(converter_para_float)

# Filtrar apenas os dados referentes ao O3
dados_o3 = dados[dados['Air_Pollutant'] == 'O3']

# IMPORTANTE: Tratar valores outliers para melhorar a visualização
# Calcular percentil 95 e usar como limite superior para detectar outliers
p95 = np.percentile(dados_o3['Value'].dropna(), 95)
print(f"Percentil 95 dos valores de O3: {p95:.2f}")

# Filtrar para usar apenas valores dentro de um limite razoável (até o percentil 95)
dados_o3_filtrados = dados_o3[dados_o3['Value'] <= p95]

# Calcular a média de O3 por região (NUTS Code)
media_o3_por_regiao = dados_o3_filtrados.groupby('NUTS_Code')['Value'].mean().reset_index()

# Ordenar os dados por valor médio para melhor visualização
media_o3_por_regiao = media_o3_por_regiao.sort_values('Value', ascending=False)

# Mostrar informações sobre os dados para diagnóstico
print(f"Número total de regiões: {len(media_o3_por_regiao)}")
print(f"Valor mínimo: {media_o3_por_regiao['Value'].min():.2f}")
print(f"Valor máximo após filtro: {media_o3_por_regiao['Value'].max():.2f}")

# Selecionar apenas as top 20 regiões para melhor visualização
top_regioes = media_o3_por_regiao.head(20)

# Criar o gráfico usando apenas as top regiões
plt.figure(figsize=(12, 8))

# Configurar estilo para um gráfico mais limpo e profissional
sns.set_style("whitegrid")

# Criar um gráfico de barras com as top regiões
ax = sns.barplot(x='NUTS_Code', y='Value', data=top_regioes, palette='viridis')

# Adicionar título e rótulos com fonte melhorada
plt.title('Top 20 Regiões com Maiores Níveis Médios de O3', fontsize=16, fontweight='bold')
plt.xlabel('Região (NUTS Code)', fontsize=14)
plt.ylabel('Nível Médio de O3 (μg/m³)', fontsize=14)

# Rotacionar os rótulos do eixo x para melhor visualização
plt.xticks(rotation=45, ha='right', fontsize=10)

# Adicionar uma linha horizontal para a média geral das top regiões
media_geral = top_regioes['Value'].mean()
plt.axhline(y=media_geral, color='red', linestyle='--', 
            label=f'Média Top 20: {media_geral:.2f} μg/m³')

# Adicionar valores nas barras para facilitar a leitura
for i, valor in enumerate(top_regioes['Value']):
    ax.text(i, valor + 0.5, f'{valor:.2f}', ha='center',
            va='bottom', fontsize=9, fontweight='bold')

# Adicionar legenda com posição fixa
plt.legend(loc='upper right')

# Ajustar limites do eixo Y para melhor visualização (margem de 10% acima do valor máximo)
valor_maximo = top_regioes['Value'].max() * 1.1
plt.ylim(0, valor_maximo)

# Ajustar o layout
plt.tight_layout()

# Salvar o gráfico (opcional)
# plt.savefig('niveis_o3_por_regiao.png', dpi=300, bbox_inches='tight')

# Mostrar o gráfico
plt.show()

# Identificar a região com nível médio de O3 mais elevado
regiao_max_o3 = media_o3_por_regiao.iloc[0]
print(f"\nA região com o nível médio de O3 mais elevado é {regiao_max_o3['NUTS_Code']} com um valor de {regiao_max_o3['Value']:.2f} μg/m³")

# Filtrar apenas regiões portuguesas (códigos NUTS que começam com PT)
regioes_pt = media_o3_por_regiao[media_o3_por_regiao['NUTS_Code'].str.startswith('PT', na=False)]

if not regioes_pt.empty:
    print(f"\nNúmero de regiões portuguesas encontradas: {len(regioes_pt)}")
    
    # Criar um gráfico só com regiões portuguesas
    plt.figure(figsize=(10, 6))
    
    # Criar um gráfico de barras para regiões portuguesas
    ax_pt = sns.barplot(x='NUTS_Code', y='Value', data=regioes_pt, palette='coolwarm')
    
    # Adicionar título e rótulos
    plt.title('Níveis Médios de O3 nas Regiões de Portugal', fontsize=16, fontweight='bold')
    plt.xlabel('Região (NUTS Code)', fontsize=14)
    plt.ylabel('Nível Médio de O3 (μg/m³)', fontsize=14)
    
    # Rotacionar os rótulos do eixo x para melhor visualização
    plt.xticks(rotation=45, ha='right', fontsize=10)
    
    # Adicionar valores nas barras
    for i, valor in enumerate(regioes_pt['Value']):
        ax_pt.text(i, valor + 0.2, f'{valor:.2f}', ha='center', 
                   va='bottom', fontsize=9, fontweight='bold')
    
    # Ajustar os limites do eixo Y
    plt.ylim(0, regioes_pt['Value'].max() * 1.1)
    
    # Ajustar o layout
    plt.tight_layout()
    
    # Mostrar o gráfico
    plt.show()
    
    # Encontrar a região de Portugal com o nível médio de O3 mais elevado
    regiao_max_o3_pt = regioes_pt.iloc[0]
    print(f"\nA região de Portugal com o nível médio de O3 mais elevado é {regiao_max_o3_pt['NUTS_Code']} com um valor de {regiao_max_o3_pt['Value']:.2f} μg/m³")
else:
    print("\nNenhuma região portuguesa (código NUTS começando com 'PT') foi encontrada nos dados.")
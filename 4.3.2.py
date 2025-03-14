import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import scipy.stats as stats

# Carregar os dados
try:
    df = pd.read_csv('AIRPOL_data.csv', delimiter=';', encoding='utf-8', decimal=',')
except:
    df = pd.read_csv('AIRPOL_data.csv', delimiter=';', encoding='latin1')

# Filtrar dados para Alemanha e PM2.5
germany_pm25 = df[(df['Country'] == 'Germany') & (df['Air_Pollutant'] == 'PM2.5')].copy()

# Verificar se temos dados suficientes
print(f"Número de registros para Alemanha e PM2.5: {len(germany_pm25)}")

# Preparar variáveis
if len(germany_pm25) > 0:
    # Converter para tipos numéricos
    germany_pm25['Air_Pollution_Average[ug/m3]'] = pd.to_numeric(germany_pm25['Air_Pollution_Average[ug/m3]'],
                                                                 errors='coerce')
    germany_pm25['Populated_Area[km2]'] = pd.to_numeric(germany_pm25['Populated_Area[km2]'], errors='coerce')
    germany_pm25['Value'] = pd.to_numeric(germany_pm25['Value'], errors='coerce')

    # Definir variáveis para o modelo
    X1 = germany_pm25['Air_Pollution_Average[ug/m3]']  # Nível médio de poluição
    X2 = germany_pm25['Populated_Area[km2]']  # Área da região afetada
    Y = germany_pm25['Value']  # Número de mortes prematuras (assumindo que é esta coluna)

    # Verificar dados faltantes
    missing_data = pd.DataFrame({
        'X1_missing': X1.isna().sum(),
        'X2_missing': X2.isna().sum(),
        'Y_missing': Y.isna().sum()
    }, index=['Contagem'])
    print("\nDados faltantes:")
    print(missing_data)

    # Remover linhas com dados faltantes
    data_clean = pd.DataFrame({'X1': X1, 'X2': X2, 'Y': Y}).dropna()
    print(f"\nNúmero de registros após remoção de dados faltantes: {len(data_clean)}")

    if len(data_clean) > 5:  # Verificar se temos dados suficientes para análise
        # Estatísticas descritivas
        print("\nEstatísticas descritivas:")
        print(data_clean.describe())

        # a) Modelo de regressão linear
        X = data_clean[['X1', 'X2']]
        y = data_clean['Y']

        # Adicionar constante para o intercepto
        X_sm = sm.add_constant(X)

        # Ajustar o modelo
        model = sm.OLS(y, X_sm).fit()

        # Resumo do modelo
        print("\nResumo do modelo de regressão:")
        print(model.summary())

        # b) Verificar condições sobre os resíduos
        residuos = model.resid
        fitted_values = model.fittedvalues

        # Gráfico de resíduos vs valores ajustados
        plt.figure(figsize=(10, 6))
        plt.scatter(fitted_values, residuos)
        plt.axhline(y=0, color='r', linestyle='-')
        plt.xlabel('Valores Ajustados')
        plt.ylabel('Resíduos')
        plt.title('Resíduos vs Valores Ajustados')
        plt.savefig('residuos_vs_fitted.png')

        # QQ plot para normalidade dos resíduos
        plt.figure(figsize=(10, 6))
        stats.probplot(residuos, dist="norm", plot=plt)
        plt.title('QQ Plot dos Resíduos')
        plt.savefig('qq_plot_residuos.png')

        # Teste de normalidade de Shapiro-Wilk
        _, p_valor_shapiro = stats.shapiro(residuos)
        print(f"\nTeste de Shapiro-Wilk para normalidade dos resíduos: p-valor = {p_valor_shapiro:.4f}")
        print(f"Conclusão: Os resíduos {'seguem' if p_valor_shapiro > 0.05 else 'não seguem'} uma distribuição normal")

        # Teste de Breusch-Pagan para homocedasticidade
        from statsmodels.stats.diagnostic import het_breuschpagan

        _, p_valor_bp, _, _ = het_breuschpagan(residuos, X_sm)
        print(f"\nTeste de Breusch-Pagan para homocedasticidade: p-valor = {p_valor_bp:.4f}")
        print(f"Conclusão: Os resíduos {'apresentam' if p_valor_bp < 0.05 else 'não apresentam'} heterocedasticidade")

        # c) Verificar colinearidade (VIF)
        vif = pd.DataFrame()
        vif["Variável"] = X.columns
        vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        print("\nVerificação de colinearidade (VIF):")
        print(vif)
        print("Valores VIF > 5 indicam possível colinearidade, VIF > 10 indicam colinearidade severa")

        # d) Comentários sobre o modelo
        print("\nComentários sobre o modelo:")
        print(f"R² ajustado: {model.rsquared_adj:.4f}")
        print(f"Erro padrão da regressão: {np.sqrt(model.mse_resid):.4f}")
        print(f"Estatística F: {model.fvalue:.4f} (p-valor: {model.f_pvalue:.4f})")

        # Significância das variáveis
        print("\nSignificância das variáveis:")
        for var, p_valor in zip(['Intercepto', 'X1', 'X2'], model.pvalues):
            print(f"{var}: p-valor = {p_valor:.4f} {'(significativo)' if p_valor < 0.05 else '(não significativo)'}")

        # Gráfico de correlação entre variáveis
        plt.figure(figsize=(8, 6))
        sns.heatmap(data_clean.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Matriz de Correlação')
        plt.savefig('matriz_correlacao.png')

    else:
        print("Não há dados suficientes para análise após remoção de valores faltantes.")
else:
    print("Não foram encontrados dados para a Alemanha com o poluente PM2.5.")

    # Gerar dados simulados para demonstração
    print("\nCriando dados simulados para demonstração...")

    # Definir semente para reprodutibilidade
    np.random.seed(42)

    # Gerar dados simulados
    n = 100
    X1 = np.random.uniform(10, 25, n)  # Nível médio de poluição entre 10 e 25 ug/m3
    X2 = np.random.uniform(100, 5000, n)  # Área entre 100 e 5000 km2

    # Gerar Y com relação linear + ruído (beta1=200, beta2=0.1, intercepto=500)
    Y = 500 + 200 * X1 + 0.1 * X2 + np.random.normal(0, 300, n)

    # Criar DataFrame
    data_clean = pd.DataFrame({'X1': X1, 'X2': X2, 'Y': Y})

    # Estatísticas descritivas
    print("\nEstatísticas descritivas (dados simulados):")
    print(data_clean.describe())

    # a) Modelo de regressão linear
    X = data_clean[['X1', 'X2']]
    y = data_clean['Y']

    # Adicionar constante para o intercepto
    X_sm = sm.add_constant(X)

    # Ajustar o modelo
    model = sm.OLS(y, X_sm).fit()

    # Resumo do modelo
    print("\nResumo do modelo de regressão (dados simulados):")
    print(model.summary())

    # b) Verificar condições sobre os resíduos
    residuos = model.resid
    fitted_values = model.fittedvalues

    # Gráfico de resíduos vs valores ajustados
    plt.figure(figsize=(10, 6))
    plt.scatter(fitted_values, residuos)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.xlabel('Valores Ajustados')
    plt.ylabel('Resíduos')
    plt.title('Resíduos vs Valores Ajustados (Dados Simulados)')
    plt.savefig('residuos_vs_fitted_simulados.png')

    # QQ plot para normalidade dos resíduos
    plt.figure(figsize=(10, 6))
    stats.probplot(residuos, dist="norm", plot=plt)
    plt.title('QQ Plot dos Resíduos (Dados Simulados)')
    plt.savefig('qq_plot_residuos_simulados.png')

    # Teste de normalidade de Shapiro-Wilk
    _, p_valor_shapiro = stats.shapiro(residuos)
    print(f"\nTeste de Shapiro-Wilk para normalidade dos resíduos: p-valor = {p_valor_shapiro:.4f}")
    print(f"Conclusão: Os resíduos {'seguem' if p_valor_shapiro > 0.05 else 'não seguem'} uma distribuição normal")

    # Teste de Breusch-Pagan para homocedasticidade
    from statsmodels.stats.diagnostic import het_breuschpagan

    _, p_valor_bp, _, _ = het_breuschpagan(residuos, X_sm)
    print(f"\nTeste de Breusch-Pagan para homocedasticidade: p-valor = {p_valor_bp:.4f}")
    print(f"Conclusão: Os resíduos {'apresentam' if p_valor_bp < 0.05 else 'não apresentam'} heterocedasticidade")

    # c) Verificar colinearidade (VIF)
    vif = pd.DataFrame()
    vif["Variável"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    print("\nVerificação de colinearidade (VIF):")
    print(vif)
    print("Valores VIF > 5 indicam possível colinearidade, VIF > 10 indicam colinearidade severa")

    # d) Comentários sobre o modelo
    print("\nComentários sobre o modelo (dados simulados):")
    print(f"R² ajustado: {model.rsquared_adj:.4f}")
    print(f"Erro padrão da regressão: {np.sqrt(model.mse_resid):.4f}")
    print(f"Estatística F: {model.fvalue:.4f} (p-valor: {model.f_pvalue:.4f})")

    # Significância das variáveis
    print("\nSignificância das variáveis:")
    for var, p_valor in zip(['Intercepto', 'X1', 'X2'], model.pvalues):
        print(f"{var}: p-valor = {p_valor:.4f} {'(significativo)' if p_valor < 0.05 else '(não significativo)'}")

    # Gráfico de correlação entre variáveis
    plt.figure(figsize=(8, 6))
    sns.heatmap(data_clean.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Matriz de Correlação (Dados Simulados)')
    plt.savefig('matriz_correlacao_simulados.png')
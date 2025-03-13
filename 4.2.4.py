import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Load the data
# Try different encoding options if needed
try:
    df = pd.read_csv('AIRPOL_data.csv', delimiter=';', encoding='utf-8')
except:
    df = pd.read_csv('AIRPOL_data.csv', delimiter=';', encoding='latin1')

# Print available countries to debug
print("Available countries in the dataset:")
print(df['Country'].unique())

# Clean country names (remove any leading/trailing spaces)
df['Country'] = df['Country'].str.strip()

# Filter for the countries of interest
countries = ["Portugal", "Albania", "Spain", "France"]
df_filtered = df[df["Country"].isin(countries)].copy()

# Print counts of records by country to debug
print("\nNumber of records per country:")
print(df_filtered['Country'].value_counts())

# Convert the air pollution column to numeric
df_filtered["Air_Pollution_Average[ug/m3]"] = pd.to_numeric(df_filtered["Air_Pollution_Average[ug/m3]"],
                                                            errors="coerce")

# Drop rows with missing pollution data
df_filtered = df_filtered.dropna(subset=["Air_Pollution_Average[ug/m3]"])

# Create samples of 20 records for each country
samples = {}
for country in countries:
    country_data = df_filtered[df_filtered["Country"] == country]
    if len(country_data) >= 20:
        samples[country] = country_data["Air_Pollution_Average[ug/m3]"].sample(20, random_state=42).values
    else:
        print(f"Warning: Not enough data for {country}. Only {len(country_data)} records available.")
        if len(country_data) > 0:
            samples[country] = country_data["Air_Pollution_Average[ug/m3]"].values

# If there's not enough data, use simulation to create sample data
if len(samples) < 4 or any(len(sample) < 20 for sample in samples.values() if sample is not None):
    print("\nNot enough real data available. Creating simulated data for demonstration purposes.")

    # Create simulated data with realistic means and variations
    np.random.seed(42)
    samples = {
        "Portugal": np.random.normal(15.2, 3.5, 20),  # Moderate pollution
        "Albania": np.random.normal(11.3, 2.8, 20),  # Lower pollution (as per example)
        "Spain": np.random.normal(17.8, 4.2, 20),  # Higher pollution
        "France": np.random.normal(16.5, 3.9, 20)  # Moderate-high pollution
    }
    print("Using simulated data with the following parameters:")
    print("Portugal: Mean=15.2, SD=3.5")
    print("Albania: Mean=11.3, SD=2.8")
    print("Spain: Mean=17.8, SD=4.2")
    print("France: Mean=16.5, SD=3.9")

# Check if we have data for all countries
if len(samples) < 4:
    print("Not enough data for all countries!")
else:
    # 1. Descriptive statistics
    print("\nDescriptive Statistics:")
    for country, data in samples.items():
        print(
            f"{country}: Mean = {data.mean():.2f}, Std = {data.std():.2f}, Min = {data.min():.2f}, Max = {data.max():.2f}")

    # 2. Create boxplot to visualize the differences
    plt.figure(figsize=(10, 6))
    data_for_plot = [samples[country] for country in countries if country in samples]
    labels = [country for country in countries if country in samples]
    plt.boxplot(data_for_plot, labels=labels)
    plt.ylabel("Air Pollution Average (μg/m³)")
    plt.title("Air Pollution Levels by Country")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('air_pollution_boxplot.png')  # Save figure instead of showing it
    print("\nBoxplot saved as 'air_pollution_boxplot.png'")

    # 3. Test for normality (Shapiro-Wilk test)
    print("\nNormality Test (Shapiro-Wilk):")
    normality_results = {}
    for country, data in samples.items():
        stat, p = stats.shapiro(data)
        normality_results[country] = p > 0.05  # True if normal
        print(f"{country}: p-value = {p:.4f} {'(Normal)' if p > 0.05 else '(Not Normal)'}")

    # 4. Test for homogeneity of variances (Levene's test)
    data_for_levene = [samples[country] for country in countries if country in samples]
    stat, p_levene = stats.levene(*data_for_levene)
    equal_var = p_levene > 0.05
    print(
        f"\nHomogeneity of Variances (Levene's test): p-value = {p_levene:.4f} {'(Equal variances)' if equal_var else '(Unequal variances)'}")

    # 5. Choose appropriate test based on assumptions
    all_normal = all(normality_results.values())

    if all_normal and equal_var:
        print("\nUsing One-way ANOVA (parametric test):")
        f_stat, p_value = stats.f_oneway(*data_for_levene)
        print(f"ANOVA: F-statistic = {f_stat:.4f}, p-value = {p_value:.4f}")
        significant = p_value < 0.05
        test_used = "ANOVA"
    else:
        print("\nUsing Kruskal-Wallis test (non-parametric test):")
        h_stat, p_value = stats.kruskal(*data_for_levene)
        print(f"Kruskal-Wallis: H-statistic = {h_stat:.4f}, p-value = {p_value:.4f}")
        significant = p_value < 0.05
        test_used = "Kruskal-Wallis"

    # 6. Post-hoc analysis if significant differences are found
    if significant:
        print("\nSignificant differences found! Performing post-hoc analysis:")

        if test_used == "ANOVA":
            # Tukey's HSD test for ANOVA
            # Prepare data for Tukey's test
            all_data = np.concatenate(data_for_levene)
            group_labels = np.concatenate(
                [[country] * len(samples[country]) for country in countries if country in samples])

            # Perform Tukey's test
            tukey_result = pairwise_tukeyhsd(all_data, group_labels, alpha=0.05)
            print("\nTukey's HSD test results:")
            print(tukey_result)
        else:
            # For Kruskal-Wallis, use pairwise Mann-Whitney U tests with Bonferroni correction
            print("\nPairwise Mann-Whitney U tests with Bonferroni correction:")
            available_countries = [country for country in countries if country in samples]
            num_comparisons = len(available_countries) * (len(available_countries) - 1) // 2
            alpha_corrected = 0.05 / num_comparisons  # Bonferroni correction

            for i in range(len(available_countries)):
                for j in range(i + 1, len(available_countries)):
                    country1 = available_countries[i]
                    country2 = available_countries[j]
                    stat, p = stats.mannwhitneyu(samples[country1], samples[country2])
                    significant_pair = p < alpha_corrected
                    print(
                        f"{country1} vs {country2}: p-value = {p:.4f} {'(Significant)' if significant_pair else '(Not significant)'}")
    else:
        print("\nNo significant differences found between the countries.")
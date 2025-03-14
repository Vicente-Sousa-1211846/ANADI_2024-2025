import pandas as pd
import matplotlib.pyplot as plt

# Try different approaches to read the CSV file
try:
    # First attempt - with semicolon delimiter
    df = pd.read_csv('AIRPOL_data.csv', delimiter=';', encoding='utf-8')

    # Print information about the Value column
    print("Using semicolon delimiter:")
    print(f"Value column data types: {df['Value'].dtype}")
    print(f"Value column first few values: {df['Value'].head().tolist()}")

    # Try to convert Value to numeric, replacing any non-numeric values with NaN
    df['Value'] = pd.to_numeric(df['Value'].str.replace(',', '.'), errors='coerce')

    # If all values are still NaN, try a different delimiter
    if df['Value'].isna().all():
        print("\nTrying comma delimiter instead...")
        df = pd.read_csv('AIRPOL_data.csv', delimiter=',', encoding='utf-8')
        print(f"Columns with comma delimiter: {df.columns.tolist()}")

        # Check if Value is in a different column or has a different name
        print(f"Column names containing 'value': {[col for col in df.columns if 'value' in col.lower()]}")

except Exception as e:
    print(f"Error reading CSV: {e}")

# If we have any valid values, proceed with the analysis
if df['Value'].notna().any():
    # Population data
    population_data = {
        "Portugal": 10300000,
        "Spain": 47400000,
        "France": 67500000,
        "Italy": 59000000
    }

    # Filter for the countries we want
    countries = ["Portugal", "Spain", "France", "Italy"]
    df_filtered = df[df["Country"].isin(countries)].copy()

    # Sum up all values for each country
    country_sums = df_filtered.groupby("Country")["Value"].sum().reset_index()
    print("\nSum of values by country:")
    print(country_sums)

    # Add population data
    country_sums["Population"] = country_sums["Country"].map(population_data)

    # Calculate deaths per 1000 inhabitants
    country_sums["Deaths_per_1000"] = (country_sums["Value"] / country_sums["Population"]) * 1000

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.bar(country_sums["Country"], country_sums["Deaths_per_1000"], color='skyblue')

    # Add labels
    plt.xlabel('Country')
    plt.ylabel('Deaths per 1000 inhabitants')
    plt.title('Number of deaths per 1000 inhabitants by country')
    plt.xticks(rotation=0)

    # Add value labels on top of each bar
    for i, v in enumerate(country_sums["Deaths_per_1000"]):
        plt.text(i, v + 0.0001, f'{v:.4f}', ha='center')

    plt.tight_layout()
    plt.show()
else:
    print("\nNo valid data found in the Value column. Please check your CSV file format.")
    print("It might help to open the file in a text editor to see the actual format.")
import pandas as pd
import numpy as np
from datetime import datetime

np.random.seed(42)

countries = ['USA', 'China', 'Japan', 'Germany', 'UK', 'France', 'Italy', 'South Korea', 
             'Canada', 'Australia', 'Brazil', 'India', 'Mexico', 'Spain', 'Russia',
             'Netherlands', 'Sweden', 'Belgium', 'Switzerland', 'Austria']

brands = ['BMW', 'Mercedes-Benz', 'Audi', 'Porsche', 'Ferrari', 'Lamborghini', 
          'McLaren', 'Aston Martin', 'Ford', 'Chevrolet', 'Dodge', 'Toyota',
          'Nissan', 'Honda', 'Hyundai', 'Kia', 'Volkswagen', 'Jaguar', 'Maserati', 'Lexus']

engine_types = ['V6', 'V8', 'V10', 'V12', 'Inline-4', 'Inline-6', 'Flat-6', 'W12', 'Hybrid', 'Electric']

horsepower_ranges = [
    (150, 250, 'Entry'),
    (250, 400, 'Mid-Range'),
    (400, 600, 'High-Performance'),
    (600, 900, 'Super'),
    (900, 1500, 'Hyper')
]

years = list(range(2015, 2026))

n_records = 5000

data = []
for _ in range(n_records):
    year = np.random.choice(years)
    country = np.random.choice(countries)
    brand = np.random.choice(brands)
    engine_type = np.random.choice(engine_types)
    
    hp_range = horsepower_ranges[np.random.randint(0, len(horsepower_ranges))]
    horsepower = np.random.randint(hp_range[0], hp_range[1])
    hp_category = hp_range[2]
    
    base_price = horsepower * np.random.uniform(100, 500)
    price = base_price * (1 + (year - 2015) * 0.03)
    
    base_sales = np.random.randint(100, 5000)
    country_factor = {'China': 2.5, 'USA': 2.0, 'Japan': 1.5, 'Germany': 1.3}.get(country, 1.0)
    brand_factor = {'Ferrari': 0.3, 'Lamborghini': 0.25, 'Porsche': 0.8, 'BMW': 1.2}.get(brand, 1.0)
    sales = int(base_sales * country_factor * brand_factor * (1 + (year - 2015) * 0.05))
    
    production = int(sales * np.random.uniform(0.9, 1.1))
    
    fuel_efficiency = max(5, 50 - horsepower * 0.02 + np.random.uniform(-5, 5))
    
    co2_emissions = horsepower * 0.5 + np.random.uniform(-20, 20)
    
    acceleration = max(1.5, 10 - horsepower * 0.005 + np.random.uniform(-0.5, 0.5))
    
    top_speed = min(350, 150 + horsepower * 0.15 + np.random.uniform(-10, 10))
    
    market_share = np.random.uniform(0.5, 15)
    
    growth_rate = np.random.uniform(-10, 25)
    
    data.append({
        'Year': year,
        'Country': country,
        'Brand': brand,
        'Engine_Type': engine_type,
        'Horsepower': horsepower,
        'HP_Category': hp_category,
        'Price_USD': round(price, 2),
        'Sales_Units': sales,
        'Production_Units': production,
        'Fuel_Efficiency_L100km': round(fuel_efficiency, 2),
        'CO2_Emissions_gkm': round(co2_emissions, 2),
        'Acceleration_0_100_s': round(acceleration, 2),
        'Top_Speed_kmh': round(top_speed, 2),
        'Market_Share_Percent': round(market_share, 2),
        'Growth_Rate_Percent': round(growth_rate, 2)
    })

df = pd.DataFrame(data)

missing_indices = np.random.choice(df.index, size=int(len(df) * 0.05), replace=False)
for idx in missing_indices:
    col = np.random.choice(['Fuel_Efficiency_L100km', 'CO2_Emissions_gkm', 'Price_USD'])
    df.loc[idx, col] = np.nan

outlier_indices = np.random.choice(df.index, size=int(len(df) * 0.02), replace=False)
for idx in outlier_indices:
    col = np.random.choice(['Price_USD', 'Sales_Units', 'Horsepower'])
    df.loc[idx, col] = df.loc[idx, col] * np.random.uniform(3, 5)

df.to_csv('fuel_performance_cars.csv', index=False)
print(f"数据集已生成: {len(df)} 条记录")
print(f"缺失值统计:\n{df.isnull().sum()}")
print(f"\n数据概览:\n{df.describe()}")

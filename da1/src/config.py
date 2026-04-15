import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

DATA_FILE = os.path.join(DATA_DIR, 'fuel_performance_cars.csv')

DPI = 300
FIGSIZE = (12, 8)
COLOR_PALETTE = 'viridis'

COUNTRY_COORDS = {
    'Germany': {'lat': 51.1657, 'lon': 10.4515, 'iso': 'DEU'},
    'Japan': {'lat': 36.2048, 'lon': 138.2529, 'iso': 'JPN'},
    'USA': {'lat': 37.0902, 'lon': -95.7129, 'iso': 'USA'},
    'Italy': {'lat': 41.8719, 'lon': 12.5674, 'iso': 'ITA'},
    'UK': {'lat': 55.3781, 'lon': -3.4360, 'iso': 'GBR'},
    'France': {'lat': 46.2276, 'lon': 2.2137, 'iso': 'FRA'},
    'China': {'lat': 35.8617, 'lon': 104.1954, 'iso': 'CHN'},
    'South Korea': {'lat': 35.9078, 'lon': 127.7669, 'iso': 'KOR'},
    'Sweden': {'lat': 60.1282, 'lon': 18.6435, 'iso': 'SWE'},
    'Spain': {'lat': 40.4637, 'lon': -3.7492, 'iso': 'ESP'}
}

HP_ORDER = ['<300hp', '300-500hp', '>500hp']

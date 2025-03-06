# data_loader.py

import pandas as pd
from sqlalchemy import create_engine

DATABASE_URL = "postgresql://windmill_owner:5xnfYcQjJFy1@ep-solitary-frost-a2al96x2.eu-central-1.aws.neon.tech/windmill?sslmode=require"

def load_data(country):
    engine = create_engine(DATABASE_URL)
    query = f"""
    SELECT order_date, city, country
    FROM sales_data
    WHERE country = '{country}'
    """
    df = pd.read_sql(query, engine)
    return df

from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / 'data'

def load_dataset():
    df_customers = pd.read_parquet(DATA_DIR / 'customers.parquet')
    df_products = pd.read_parquet(DATA_DIR / 'articles.parquet')
    df_transactions = pd.read_parquet(DATA_DIR / 'transactions_train.parquet')
    return df_customers, df_products, df_transactions

    return df_customers, df_products, df_transactions

def load_complete_dateset_filtered_date(beg_date, end_date):

    df_customers = pd.read_parquet('data/customers.parquet')
    df_products = pd.read_parquet('data/articles.parquet')
    df_transactions = pd.read_parquet('data/transactions_train.parquet')

    df_transactions['t_dat'] = pd.to_datetime(df_transactions['t_dat'])
    mask = (df_transactions['t_dat'] >= beg_date) & (df_transactions['t_dat'] <= end_date)
    df_transactions_filtered = df_transactions.loc[mask]

    return df_customers, df_products, df_transactions_filtered

def load_complete_dataset_filtered_number_customers(num_customers):

    df_customers = pd.read_parquet('data/customers.parquet')
    df_products = pd.read_parquet('data/articles.parquet')
    df_transactions = pd.read_parquet('data/transactions_train.parquet')

    unique_customer_ids = df_transactions['customer_id'].unique()

    sampled_customer_ids = pd.Series(unique_customer_ids).sample(n=num_customers, random_state=42)

    df_customers_filtered = df_customers[df_customers['customer_id'].isin(sampled_customer_ids)]
    df_transactions_filtered = df_transactions[df_transactions['customer_id'].isin(sampled_customer_ids)]

    return df_customers_filtered, df_products, df_transactions_filtered

def merge_datasets_left(df_customers, df_products, df_transactions):
    df_merged = pd.merge(df_transactions, df_customers, on='customer_id', how='left')
    df_merged = pd.merge(df_merged, df_products, on='article_id', how='left')
    return df_merged

def merge_datasets_inner(df_customers, df_products, df_transactions):
    df_merged = pd.merge(df_transactions, df_customers, on='customer_id', how='inner')
    df_merged = pd.merge(df_merged, df_products, on='article_id', how='inner')
    return df_merged
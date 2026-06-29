from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / 'data'

def load_dataset():
    df_customers = pd.read_parquet(DATA_DIR / 'customers.parquet')
    df_products = pd.read_parquet(DATA_DIR / 'articles.parquet')
    df_transactions = pd.read_parquet(DATA_DIR / 'transactions_train.parquet')
    return df_customers, df_products, df_transactions

def load_complete_dateset_filtered_date(beg_date, end_date):

    df_customers = pd.read_parquet(DATA_DIR / 'customers.parquet')
    df_products = pd.read_parquet(DATA_DIR / 'articles.parquet')
    df_transactions = pd.read_parquet(DATA_DIR / 'transactions_train.parquet')

    df_transactions['t_dat'] = pd.to_datetime(df_transactions['t_dat'])
    mask = (df_transactions['t_dat'] >= beg_date) & (df_transactions['t_dat'] <= end_date)
    df_transactions_filtered = df_transactions.loc[mask]

    return df_customers, df_products, df_transactions_filtered

def load_complete_dataset_filtered_number_customers(num_customers):

    df_customers = pd.read_parquet(DATA_DIR / 'customers.parquet')
    df_products = pd.read_parquet(DATA_DIR / 'articles.parquet')
    df_transactions = pd.read_parquet(DATA_DIR / 'transactions_train.parquet')

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

def transactions_group(df_transactions):
    groups = df_transactions.groupby(['t_dat', 'customer_id'], sort=False).ngroup()
    df_transactions = df_transactions.copy()
    df_transactions['transaction_id'] = (
        groups.astype(str) + '_' + df_transactions['customer_id'].astype(str)
    )
    return df_transactions

def filter_customers_by_min_orders(df_transactions, n):
    orders_per_customer = df_transactions.groupby('customer_id')['transaction_id'].nunique()
    active_customers = orders_per_customer[orders_per_customer > n].index
    return df_transactions[df_transactions['customer_id'].isin(active_customers)]

def compute_customer_category_affinity(
    df_products,
    df_transactions,
    category_col: str = 'product_group_name',
    pivot: bool = True,
) -> pd.DataFrame:
    """
    Calcula la afinidad de cada cliente a cada categoría de producto.

    La afinidad es la proporción de compras del cliente que pertenecen
    a cada categoría (suma 1 por cliente).

    Parámetros
    ----------
    category_col : columna de df_products a usar como categoría.
                   Por defecto 'product_group_name'. Otras opciones:
                   'index_group_name', 'garment_group_name', 'section_name'.
    pivot        : si True devuelve un DataFrame ancho (customer_id × categorías).
                   si False devuelve formato largo (customer_id, category, affinity).
    """
    df = df_transactions[['customer_id', 'article_id']].merge(
        df_products[['article_id', category_col]],
        on='article_id',
        how='left',
    )

    counts = (
        df.groupby(['customer_id', category_col])
        .size()
        .reset_index(name='n_purchases')
    )

    totals = counts.groupby('customer_id')['n_purchases'].transform('sum')
    counts['affinity'] = counts['n_purchases'] / totals

    if not pivot:
        return counts[['customer_id', category_col, 'affinity']]

    affinity_pivot = counts.pivot_table(
        index='customer_id',
        columns=category_col,
        values='affinity',
        fill_value=0.0,
    )
    affinity_pivot.columns.name = None
    return affinity_pivot


def imputar_nulos_tfm(df):
    """
    Imputa los valores nulos del DataFrame de H&M basándose en reglas de negocio.
    """
    print("Iniciando imputación de valores nulos...")
    df_clean = df.copy()
    
    # 1. Variables Binarias de Marketing (FN y Active)
    # Rellenamos con 0 y forzamos a int8 para ahorrar muchísima memoria RAM
    for col in ['FN', 'Active']:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(0).astype('int8')
            
    # 2. Variables de Fidelización (Textos/Categorías)
    if 'fashion_news_frequency' in df_clean.columns:
        # Primero unificamos si hay alguna escrita como 'None' en minúsculas, luego rellenamos los nulos
        df_clean['fashion_news_frequency'] = df_clean['fashion_news_frequency'].replace('None', 'NONE').fillna('NONE')
        
    if 'club_member_status' in df_clean.columns:
        df_clean['club_member_status'] = df_clean['club_member_status'].fillna('GUEST')
        
    # 3. La Edad (Estrategia MVP: Mediana global)
    if 'age' in df_clean.columns:
        mediana_edad = df_clean['age'].median()
        df_clean['age'] = df_clean['age'].fillna(mediana_edad)
        print(f" -> Edad imputada por la mediana: {mediana_edad} años.")
        
    # 4. Descripciones (Solo por si no las habías filtrado antes)
    if 'detail_desc' in df_clean.columns:
        df_clean['detail_desc'] = df_clean['detail_desc'].fillna('Sin descripción')

    print("¡Imputación completada! Dataset listo para el análisis.")
    return df_clean
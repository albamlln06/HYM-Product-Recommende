from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / 'data'

def load_dataset():
    """
    Carga los archivos desde el disco y aplica el formato correcto a las columnas clave.
    Todas las demás funciones deben llamar a esta para evitar repetir código.
    """
    df_customers = pd.read_parquet(DATA_DIR / 'customers.parquet')
    df_products = pd.read_parquet(DATA_DIR / 'articles.parquet')
    df_transactions = pd.read_parquet(DATA_DIR / 'transactions_train.parquet')
    #Forzar mismos tipos para el JOIN
    df_customers['customer_id'] = df_customers['customer_id'].astype(str)
    df_products['article_id'] = df_products['article_id'].astype(str)

    df_transactions['customer_id'] = df_transactions['customer_id'].astype(str)
    df_transactions['article_id'] = df_transactions['article_id'].astype(str)
    # Centralizamos aquí la conversión a datetime para que aplique a todo
    if 't_dat' in df_transactions.columns:
        df_transactions['t_dat'] = pd.to_datetime(df_transactions['t_dat'])
    
    if 'sales_channel_id' in df_transactions.columns:
        # Si es 2 (Online) pasa a ser 1. Si es 1 (Física) pasa a ser 0.
        # Al usar astype('int8') la comprimimos al tamaño mínimo en el mismo paso
        df_transactions['is_online'] = (df_transactions['sales_channel_id'] == 2).astype('int8')
        
        # Borramos la original para no duplicar datos
        df_transactions = df_transactions.drop(columns=['sales_channel_id'])
    
    df_customers = auto_optimize_categories(df_customers, exclude_cols=['customer_id'])
    df_products = auto_optimize_categories(df_products, exclude_cols=['article_id'])
        
    return df_customers, df_products, df_transactions


def load_complete_dateset_filtered_date(beg_date, end_date):
    """Filtra las transacciones por un rango de fechas."""
    # 1. Cargamos los datos usando la función base
    df_customers, df_products, df_transactions = load_dataset()

    # 2. Aplicamos el filtro (t_dat ya es datetime gracias a la función base)
    mask = (df_transactions['t_dat'] >= beg_date) & (df_transactions['t_dat'] <= end_date)
    df_transactions_filtered = df_transactions.loc[mask]

    return df_customers, df_products, df_transactions_filtered

def load_complete_dataset_filtered_number_customers(num_customers, random_state=42):
    """Filtra el dataset para quedarse con un número específico de clientes."""
    # 1. Cargamos los datos usando la función base
    df_customers, df_products, df_transactions = load_dataset()

    # 1. SAMPLEO INTELIGENTE DE CLIENTES
    # ==========================================
    # Contamos cuántas transacciones tiene cada cliente en total
    purchases_client = df_transactions.groupby("customer_id").size()

    # Sampleo ponderado: más probabilidad a los que más compran, pero manteniendo aleatoriedad
    sampled_customer_ids = purchases_client.sample(
        n=num_customers, 
        random_state=random_state, 
        weights=purchases_client.values
    ).index.tolist()
    # 3. Filtramos
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
    print(df_transactions.describe())
    print(df_transactions.info())
    orders_per_customer = df_transactions.groupby('customer_id')['transaction_id'].nunique()
    active_customers = orders_per_customer[orders_per_customer > n].index
    return df_transactions[df_transactions['customer_id'].isin(active_customers)]

def filter_customers_by_activity(
    df_transactions,
    min_purchases: int = 3,
    max_months_since_last_purchase: int = 6,
):

    df_transactions = df_transactions.copy()
    df_transactions['t_dat'] = pd.to_datetime(df_transactions['t_dat'])
    reference_date = df_transactions['t_dat'].max()
    cutoff_date = reference_date - pd.DateOffset(months=max_months_since_last_purchase)

    purchase_counts = df_transactions.groupby('customer_id')['article_id'].count()
    last_purchase = df_transactions.groupby('customer_id')['t_dat'].max()

    active = purchase_counts.index[
        (purchase_counts >= min_purchases) & (last_purchase >= cutoff_date)
    ]

    df_filtered = df_transactions[df_transactions['customer_id'].isin(active)]
    return transactions_group(df_filtered)


def compute_customer_category_affinity(
    df_products,
    df_transactions,
    category_col: str = 'product_group_name',
    pivot: bool = True,
):
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
    df_clean = df.copy()
    
    # 1. Variables Binarias de Marketing (FN y Active)
    # Rellenamos con 0 y forzamos a int8 para ahorrar muchísima memoria RAM
    for col in ['FN', 'Active']:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(0).astype('int8')
            
    # 2. Variables de Fidelización (Textos/Categorías)
    if 'fashion_news_frequency' in df_clean.columns:
        # Protegemos si es categórica
        if df_clean['fashion_news_frequency'].dtype.name == 'category':
            if 'NONE' not in df_clean['fashion_news_frequency'].cat.categories:
                df_clean['fashion_news_frequency'] = df_clean['fashion_news_frequency'].cat.add_categories('NONE')
        # Reemplazamos y rellenamos
        df_clean['fashion_news_frequency'] = df_clean['fashion_news_frequency'].replace('None', 'NONE').fillna('NONE')
        
    if 'club_member_status' in df_clean.columns:
        # Protegemos si es categórica
        if df_clean['club_member_status'].dtype.name == 'category':
            if 'GUEST' not in df_clean['club_member_status'].cat.categories:
                df_clean['club_member_status'] = df_clean['club_member_status'].cat.add_categories('GUEST')
        df_clean['club_member_status'] = df_clean['club_member_status'].fillna('GUEST')
        
    # 3. La Edad (Estrategia MVP: Mediana global)
    if 'age' in df_clean.columns:
        mediana_edad = df_clean['age'].median()
        df_clean['age'] = df_clean['age'].fillna(mediana_edad)
        
    # 4. Descripciones (Aunque la descartes luego, la protegemos por si acaso el pipeline pasa por aquí antes)
    if 'detail_desc' in df_clean.columns:
        if df_clean['detail_desc'].dtype.name == 'category':
            if 'Sin descripción' not in df_clean['detail_desc'].cat.categories:
                df_clean['detail_desc'] = df_clean['detail_desc'].cat.add_categories('Sin descripción')
        df_clean['detail_desc'] = df_clean['detail_desc'].fillna('Sin descripción')
        
    return df_clean


def auto_optimize_categories(df, max_ratio=0.05, max_categories=500, exclude_cols=None):
    """
    Convierte automáticamente columnas de texto a 'category' de forma segura.
    
    Parámetros:
    -----------
    df : pandas.DataFrame
    max_ratio : float
        Proporción máxima permitida.
    max_categories : int
        Límite máximo absoluto de categorías a crear.
    exclude_cols : list
        Lista de nombres de columnas que NO deben convertirse bajo ninguna circunstancia (ej. fechas).
    """
    # Si no nos pasan ninguna lista de exclusión, usamos una vacía por defecto
    if exclude_cols is None:
        exclude_cols = []
        
    text_columns = df.select_dtypes(include=['object', 'string']).columns
    total_rows = len(df)
    columnas_convertidas = []
    
    for col in text_columns:
        # Si la columna está en la lista de ignoradas, saltamos a la siguiente
        if col in exclude_cols:
            continue
            
        num_unique = df[col].nunique()
        ratio = num_unique / total_rows
        
        if ratio < max_ratio and num_unique <= max_categories:
            df[col] = df[col].astype('category')
            columnas_convertidas.append(col)
        
    return df
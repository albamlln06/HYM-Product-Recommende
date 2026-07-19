from datetime import datetime
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import xgboost as xgb
from utils.preprocess import auto_optimize_categories, imputar_nulos_tfm

CATEGORICAL_FEATURES_CLUSTER = [
    'product_group_name',
    'perceived_colour_master_name',
    'garment_group_name',
    'index_group_name',
    'graphical_appearance_name',
    'index_name',
    'section_name',
]

NUMERIC_FEATURES_CLUSTER = [
    'avg_buyer_age',
    'avg_price',
    'sales_volume',
    'online_ratio',
    'recency_days',
]


CATEGORICAL_FEATURES_XGBOOST = [
    'product_group_name',
    'perceived_colour_master_name',
    'garment_group_name',
    'index_group_name',
    'graphical_appearance_name',
    'index_name',
    'section_name',
]

NUMERIC_FEATURES_XGBOOST = [
    'avg_buyer_age',
    'avg_price',
    'sales_volume',
    'online_ratio',
    'recency_days',
    'sex_popularity',
    'total_sales_volume',
]

USER_NUMERIC_FEATURES = [
    'user_n_compras',
    'user_precio_medio',
    'user_precio_std',
    'user_online_ratio',
    'age',
]

USER_CATEGORICAL_FEATURES = ['club_member_status']

# Config por defecto: todas las features disponibles.
# Pasa una copia modificada a xgboost_preprocess / entrenar_modelo_xgboost
# para probar distintas combinaciones sin tocar estas listas.
FEATURE_CONFIG_DEFAULT = {
    "article_numeric":    NUMERIC_FEATURES_XGBOOST,
    "article_categorical": CATEGORICAL_FEATURES_XGBOOST,
    "user_numeric":       USER_NUMERIC_FEATURES,
    "user_categorical":   USER_CATEGORICAL_FEATURES,
}

# Pesos por categoría para el entrenamiento de XGBoost (sample_weight).
# 1.0 = sin cambios. Un valor < 1.0 hace que esas filas influyan menos en la
# función de pérdida (p.ej. bajar la importancia de la ropa interior en el
# ranking); > 1.0 las prioriza. None / dict vacío -> todas las filas pesan igual.
CATEGORY_WEIGHTS_DEFAULT = None

def compute_article_features(df_customers, df_products, df_transactions):
    """
    Calcula features agregadas por article_id a partir de transacciones.
 
    df_transactions DEBE ser ya el subconjunto de train (no pases el
    dataset completo si vas a evaluar con leave-one-out después).
    """
    avg_age = (
        df_transactions.merge(df_customers[["customer_id", "age"]], on="customer_id", how="left")
        .groupby("article_id")["age"]
        .mean()
        .reset_index()
        .rename(columns={"age": "avg_buyer_age"})
    )
 
    max_date = df_transactions["t_dat"].max()
    article_sale_features = (
        df_transactions.groupby("article_id").agg(
            avg_price=("price", "mean"),
            sales_volume=("article_id", "count"),
            online_ratio=("is_online", "mean"),
            recency_days=("t_dat", lambda x: (max_date - x.max()).days),
            first_sale_date = ("t_dat", "min")
        ).reset_index()
    )
    article_sale_features['product_age_days'] = (max_date - article_sale_features['first_sale_date']).dt.days
    article_sale_features = article_sale_features.drop(columns=['first_sale_date'])

    # Esto se pone justo antes de hacer tu df_products.merge()
    fecha_corte_30d = max_date - pd.Timedelta(days=30)
    
    # Filtramos ventas recientes y contamos
    ventas_recientes = df_transactions[df_transactions['t_dat'] >= fecha_corte_30d]
    momentum = ventas_recientes.groupby('article_id').size().reset_index(name='sales_last_30d')

    df = df_products.merge(avg_age, on="article_id", how="left")
    df = df.merge(article_sale_features, on="article_id", how="left")
    df = df.merge(momentum, on="article_id", how="left")
    # Estrategia de imputación personalizada para evitar sesgos en el modelo
    imputation_strategy = {
        'avg_buyer_age': df['avg_buyer_age'].median(),
        'avg_price': df['avg_price'].median(),
        'sales_volume': 0.0,
        'online_ratio': 0.0,
        'recency_days': 999.0,#  Valor elevado para evitar sesgos en el modelo
        'sales_last_30d':0.0,
        'product_age_days': 0.0  
    }
    # Aplicamos la lógica columna por columna
    for col, fill_value in imputation_strategy.items():
        if col in df.columns:
            df[col] = df[col].fillna(fill_value)
            
    if 'index_group_name' in df.columns:
        group_mean = df.groupby('index_group_name')['sales_volume'].transform('mean')
        df['sex_popularity'] = (df['sales_volume'] / group_mean.replace(0, np.nan)).fillna(0)
    else:
        df['sex_popularity'] = 0.0

    if 'product_group_name' in df.columns:
        df['total_sales_volume'] = df.groupby('product_group_name')['sales_volume'].transform('sum')
    else:
        df['total_sales_volume'] = df['sales_volume']

    return df

def clustering_preprocess(df_customers, df_products, df_transactions):
    """
    Preprocesado para K-Means: dummies + escalado (StandardScaler),
    porque K-Means se basa en distancias y necesita que todas las
    features estén en una escala comparable.
    """
    df = compute_article_features(df_customers, df_products, df_transactions)
 
    available_cat = [c for c in CATEGORICAL_FEATURES_CLUSTER if c in df.columns]
    encoded = pd.get_dummies(df[available_cat], drop_first=False)
    X = pd.concat([encoded, df[NUMERIC_FEATURES_CLUSTER]], axis=1).astype(float)
 
    scaler = StandardScaler()
    X_final = scaler.fit_transform(X)
    article_ids = df["article_id"].values
 
    return X_final, article_ids, scaler, df

def clustering_preprocess_old(df_customers, df_products, df_transactions):

    avg_age = (
        df_transactions.merge(df_customers[['customer_id', 'age']], on='customer_id', how='left')
        .groupby('article_id')['age']
        .mean()
        .reset_index()
        .rename(columns={'age': 'avg_buyer_age'})
    )

    max_date = df_transactions['t_dat'].max()
    tx_features = (
        df_transactions.groupby('article_id').agg(
            avg_price=('price', 'mean'),
            sales_volume=('article_id', 'count'),
            online_ratio=("is_online", "mean"),
            recency_days=('t_dat', lambda x: (max_date - x.max()).days),
        ).reset_index()
    )

    df = df_products.merge(avg_age, on='article_id', how='left')
    df = df.merge(tx_features, on='article_id', how='left')

    for col in NUMERIC_FEATURES_CLUSTER:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    available_cat = [c for c in CATEGORICAL_FEATURES_CLUSTER if c in df.columns]
    encoded = pd.get_dummies(df[available_cat], drop_first=False)
    X = pd.concat([encoded, df[NUMERIC_FEATURES_CLUSTER]], axis=1).astype(float)

    scaler = StandardScaler()
    X_final = scaler.fit_transform(X)

    article_ids = df['article_id'].values

    return X_final, article_ids, scaler, df

def compute_user_features(df_customers, df_transactions, df_products):
    #Solo para XGBoost
    """Features de usuario calculadas SOLO con transacciones de train."""
    max_date = df_transactions['t_dat'].max()

    user_tx = df_transactions.groupby("customer_id").agg(
        user_n_compras=("article_id", "count"),
        user_precio_medio=("price", "mean"),
        user_precio_std=("price", "std"),
        user_online_ratio=("is_online", "mean"),
        user_last_buy = ("t_dat", "max")
    ).reset_index()
    user_tx["user_precio_std"] = user_tx["user_precio_std"].fillna(0)
    user_tx["user_recency_days"] = (max_date - user_tx['user_last_buy']).dt.days
    user_tx = user_tx.drop(columns=['user_last_buy']) # Borramos la fecha, ya no hace falta
 
    cols_customer = [c for c in ["customer_id", "age", "club_member_status"] if c in df_customers.columns]
    user_df = user_tx.merge(df_customers[cols_customer], on="customer_id", how="left")
 
    if "age" in user_df.columns:
        user_df["age"] = user_df["age"].fillna(user_df["age"].median())
    
    tx_sections = df_transactions[['customer_id', 'article_id']].merge(
        df_products[['article_id', 'section_name']], on='article_id', how='left'
    )
    
    # 2. Contamos cuántas veces ha comprado en cada sección
    section_counts = tx_sections.groupby(['customer_id', 'section_name']).size().reset_index(name='count')
    
    # 3. Ordenamos y nos quedamos solo con la primera (la más comprada) por cliente
    favorite_section = section_counts.sort_values('count', ascending=False).drop_duplicates('customer_id')
    favorite_section = favorite_section.rename(columns={'section_name': 'user_favorite_section'})
    favorite_section = favorite_section[['customer_id', 'user_favorite_section']]
    
    # Unimos todo al df de usuario
    user_df = user_df.merge(favorite_section, on="customer_id", how="left")
    return user_df

def compute_cross_features(dataset):
    """
    Calcula variables que relacionan al cliente con el artículo específico.
    Se debe ejecutar DESPUÉS de hacer los merges de usuario y artículo.
    """
    # 1. AFINIDAD DE PRECIO
    # Diferencia absoluta
    dataset['cross_price_diff'] = (dataset['avg_price'] - dataset['user_precio_medio']).abs()
    # Ratio (Cuidado con las divisiones por cero)
    dataset['cross_price_ratio'] = dataset['avg_price'] / dataset['user_precio_medio'].replace(0, np.nan)
    # Si el usuario no tiene precio medio, el ratio es 1 (precio normal)
    dataset['cross_price_ratio'] = dataset['cross_price_ratio'].fillna(1.0)

    # 2. AFINIDAD DEMOGRÁFICA
    if 'age' in dataset.columns and 'avg_buyer_age' in dataset.columns:
        dataset['cross_age_diff'] = (dataset['age'] - dataset['avg_buyer_age']).abs()
        
    # 3. AFINIDAD DE SECCIÓN
    if 'section_name' in dataset.columns and 'user_favorite_section' in dataset.columns:
        # 1 si es su sección favorita, 0 si no
        dataset['cross_is_favorite_section'] = (dataset['section_name'] == dataset['user_favorite_section']).astype('int8')

    return dataset


def find_optimal_k(X_final, k_range=range(2, 15)):

    #Esta funcion solo proporciona una evalución para elegir el mejor valor de K, no es una parte del pipeline de recomendación.
    inertias = []
    silhouettes = []

    #Buscamos mejor inercia y mayor silhouette score para cada valor de K

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_final)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_final, labels, sample_size=5000, random_state=42))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(list(k_range), inertias, marker='o')
    axes[0].set_title('Elbow Method')
    axes[0].set_xlabel('K')
    axes[0].set_ylabel('Inertia')

    axes[1].plot(list(k_range), silhouettes, marker='o', color='orange')
    axes[1].set_title('Silhouette Score')
    axes[1].set_xlabel('K')
    axes[1].set_ylabel('Score')

    plt.tight_layout()
    plt.show()


def fit_product_clustering(X_final, K, article_ids):
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_final)

    df_clusters = pd.DataFrame({
        'article_id': article_ids,
        'cluster': labels,
    })

    return df_clusters, kmeans


def inspect_clusters(df_products, df_clusters, numeric_cols=None, category_col='product_group_name'):
    df_merged = df_products.merge(df_clusters, on='article_id', how='left')

    if numeric_cols is None:
        numeric_cols = ['avg_buyer_age']

    available_numeric = [c for c in numeric_cols if c in df_merged.columns]

    summary = df_merged.groupby('cluster').agg(
        count=('article_id', 'count'),
        top_category=(category_col, lambda x: x.value_counts().index[0]),
        **{col: (col, 'mean') for col in available_numeric}
    ).reset_index()

    return df_merged, summary


def cluster_products(K=6, datasets=None):
    numeric_cols = ['avg_buyer_age']

    X_final, article_ids, scaler, df_products = clustering_preprocess(*datasets)

    find_optimal_k(X_final, k_range=range(2, 15))

    df_article_clusters, kmeans_model = fit_product_clustering(X_final, K, article_ids)

    df_merged, summary = inspect_clusters(
        df_products=df_products,
        df_clusters=df_article_clusters,
        numeric_cols=numeric_cols,
        category_col='product_group_name',
    )

    return df_article_clusters, kmeans_model, scaler, df_merged, summary, X_final, article_ids, df_products

def get_customer_profile(customer_id, df_transactions, X_df):
    bought = df_transactions.loc[df_transactions['customer_id'] == customer_id, 'article_id'].unique()
    bought_valid = X_df.index.intersection(bought)

    if len(bought_valid) == 0:
        return pd.DataFrame()
    #Ensucia la conosola
    #print("Customer profile: ", X_df.loc[bought_valid].mean(axis=0).to_frame().T)

    return X_df.loc[bought_valid].mean(axis=0).to_frame().T

# ==========================================
# FILTRADO ESTACIONAL (recomendaciones en vivo)
# ==========================================
# Solo se filtran las categorías inequívocamente ligadas a una estación
# (abrigos, botas, bañadores...). El resto del catálogo (camisetas,
# pantalones, vestidos, accesorios...) es "todo el año" y nunca se excluye.
PRODUCT_TYPES_INVIERNO = {
    "Coat", "Jacket", "Outdoor Waistcoat", "Outdoor trousers", "Outdoor overall",
    "Gloves", "Scarf", "Beanie", "Hat/beanie", "Boots", "Bootie", "Long John",
    "Sleeping sack",
}
PRODUCT_TYPES_VERANO = {
    "Swimwear bottom", "Bikini top", "Swimsuit", "Swimwear set", "Swimwear top",
    "Sandals", "Flip flop", "Heeled sandals", "Sarong",
}


def get_current_season(reference_date=None):
    """
    Estación real (hemisferio norte) a partir de una fecha. None -> ahora mismo.

    Devuelve 'invierno' / 'verano' / None. None cubre primavera y otoño: son
    estaciones de transición, no tiene sentido excluir ni abrigos ni ropa de
    baño en esos meses.
    """
    month = (reference_date or datetime.now()).month
    if month in (12, 1, 2):
        return "invierno"
    if month in (6, 7, 8):
        return "verano"
    return None


def filter_articles_by_season(df_articles, season, type_col="product_type_name"):
    """
    Descarta del pool de candidatos los artículos fuera de estación.

    season='verano'   -> descarta PRODUCT_TYPES_INVIERNO (abrigos, botas...).
    season='invierno' -> descarta PRODUCT_TYPES_VERANO (bañadores, sandalias...).
    season=None        -> no filtra nada (primavera/otoño).
    """
    if season is None or type_col not in df_articles.columns:
        return df_articles

    off_season_types = PRODUCT_TYPES_INVIERNO if season == "verano" else PRODUCT_TYPES_VERANO
    return df_articles[~df_articles[type_col].isin(off_season_types)]


def recommend_by_cluster_similarity(
    customer_id,
    df_transactions,
    df_clusters_with_price,
    X_df,                      
    top_n=10,
    rating_col='avg_buyer_age',
):
    customer_profile = get_customer_profile(customer_id, df_transactions, X_df)
    if customer_profile.empty:
        return pd.DataFrame()

    bought = df_transactions.loc[df_transactions['customer_id'] == customer_id, 'article_id'].unique()

    clusters_bought = df_clusters_with_price.loc[
        df_clusters_with_price['article_id'].isin(bought), 'cluster'
    ].unique()

    candidates = df_clusters_with_price[
        df_clusters_with_price['cluster'].isin(clusters_bought)
        & ~df_clusters_with_price['article_id'].isin(bought)
    ].copy()
    if candidates.empty:
        return pd.DataFrame()

    candidate_vectors = X_df.loc[candidates['article_id']]
    candidates['similarity'] = cosine_similarity(customer_profile, candidate_vectors).flatten()

    recommendations = candidates.sort_values(
        by=['similarity', rating_col], ascending=[False, False]
    ).head(top_n)

    cols = ['article_id', 'cluster', 'similarity']
    if rating_col in recommendations.columns:
        cols.append(rating_col)
    return recommendations[cols]

# ==========================================
# MODELO XGBOOST
# ==========================================
def generar_negativos_cliente(
    positivos_cliente, # DataFrame con los artículos que compró el cliente
    cliente,
    compras_por_cliente,
    pool_candidatos,
    prob_muestreo,
    rng,
    mapa_articulo_cluster=None,
    articulos_por_cluster=None,
    n_negativos_faciles=4,
    n_negativos_dificiles=4
):
    # Usamos un set para las búsquedas rápidas (O(1))
    comprados = compras_por_cliente.get(cliente, set())
    articulos_positivos = positivos_cliente['article_id'].tolist()
    num_positivos = len(articulos_positivos)
    negativos = set()

    # ==========================================
    # 1. NEGATIVOS FÁCILES (Globales por popularidad)
    # ==========================================
    n_faciles_necesarios = num_positivos * n_negativos_faciles
    intentos_faciles = 0
    
    while len(negativos) < n_faciles_necesarios and intentos_faciles < intentos_faciles <15:
        candidatos_batch = rng.choice(
            pool_candidatos,
            size=n_faciles_necesarios*3,
            p=prob_muestreo,
            replace=True
        )
        for item in candidatos_batch:
            if item not in comprados and item not in negativos:
                negativos.add(item)
            if len(negativos) >= n_faciles_necesarios:
                break
        intentos_faciles += 1
             
    # ==========================================
    # 2. HARD NEGATIVES (Del mismo clúster)
    # ==========================================
    # ==========================================
    
    if mapa_articulo_cluster and articulos_por_cluster and n_negativos_dificiles > 0:
        
        for articulo in articulos_positivos:
            cluster = mapa_articulo_cluster.get(articulo)
            
            # Si el artículo no tiene clúster o el clúster está vacío, pasamos al siguiente
            if cluster is None or cluster not in articulos_por_cluster:
                continue
                
            candidatos_cluster = articulos_por_cluster[cluster]
            if len(candidatos_cluster) == 0:
                continue
            
            target_actual = n_negativos_dificiles
            agregados = 0
            intentos_dificiles = 0
            
            # Subimos también aquí un poco la seguridad
            while agregados < target_actual and intentos_dificiles < 10:
                size_muestra = min(len(candidatos_cluster), target_actual * 3)
                muestra_cluster = rng.choice(candidatos_cluster, size=size_muestra, replace=True)
                
                for item in muestra_cluster:
                    if item not in comprados and item not in negativos:
                        negativos.add(item)
                        agregados += 1
                    if agregados >= target_actual:
                        break
                intentos_dificiles += 1

    # Formatear la salida final
    return [{'customer_id': cliente, 'article_id': neg, 'label': 0} for neg in negativos]


def compute_category_sample_weights(dataset, article_df, category_weights=None, default_weight=1.0):
    """
    Calcula el sample_weight de cada fila del dataset de entrenamiento de
    XGBoost según la categoría del artículo (positivo o negativo muestreado).

    category_weights: dict {columna_categorica: {valor: peso}}, p.ej.
        {"product_group_name": {"Underwear": 0.3, "Underwear/nightwear": 0.3}}
    hace que las filas de ropa interior pesen un 30% en la función de pérdida.
    Los valores no listados (o columnas ausentes en article_df) usan
    default_weight. Si se configura más de una columna, los pesos se
    multiplican entre sí.
    """
    weights = pd.Series(default_weight, index=dataset.index, dtype=float)
    if not category_weights:
        return weights

    article_categories = article_df.set_index('article_id')
    for category_col, value_weights in category_weights.items():
        if category_col not in article_categories.columns:
            continue
        col_values = dataset['article_id'].map(article_categories[category_col])
        weights *= col_values.map(value_weights).fillna(default_weight)

    return weights


def xgboost_preprocess(df_customers, df_products, df_transactions, 
    n_negativos_faciles=4,            
    n_negativos_dificiles=4,          
    random_state=42, 
    feature_config=None, 
    category_weights=None,
    mapa_articulo_cluster=None,       
    articulos_por_cluster=None,
    candidate_pool_size = 3000        
):
    rng = np.random.default_rng(random_state)

    # 1. Features de artículo y usuario
    article_df = compute_article_features(df_customers, df_products, df_transactions)
    user_df    = compute_user_features(df_customers, df_transactions, df_products)
    top_articles = article_df.nlargest(candidate_pool_size, 'sales_volume') #generamos la lista de candidatos antes para usarla en generación de negativos
    # 2. Creamos la lista para el parámetro 'pool_candidatos'
    candidate_pool = top_articles['article_id'].tolist()
    prob_muestreo = (top_articles['sales_volume'] / top_articles['sales_volume'].sum()).tolist()

    # 3. Muestras positivas: pares únicos (cliente, artículo) realmente comprados
    positivos = df_transactions[['customer_id', 'article_id']].drop_duplicates().copy()
    positivos['label'] = 1
  
    compras_por_cliente = (
        df_transactions.groupby('customer_id')['article_id'].apply(set).to_dict()
    )

    negativos_rows = []
    for cliente, grupo in positivos.groupby('customer_id'):
        negativos_rows.extend(
            generar_negativos_cliente(
                positivos_cliente=grupo,
                cliente=cliente,
                compras_por_cliente=compras_por_cliente,
                pool_candidatos=candidate_pool,
                prob_muestreo=prob_muestreo,
                mapa_articulo_cluster=mapa_articulo_cluster,
                articulos_por_cluster=articulos_por_cluster,
                n_negativos_faciles=n_negativos_faciles,
                n_negativos_dificiles=n_negativos_dificiles,
                rng=rng,
            )
        )
    negativos = pd.DataFrame(negativos_rows)

    dataset   = pd.concat([positivos, negativos], ignore_index=True)

    # 3b. Peso de cada fila según la categoría del artículo (p.ej. bajar el
    #     peso de la ropa interior). None -> todas las filas pesan 1.0.
    dataset['sample_weight'] = compute_category_sample_weights(dataset, article_df, category_weights)

   
    # 4. Join final: dataset × features de usuario × features de artículo
    dataset = (
        dataset
        .merge(user_df,       on='customer_id', how='left')
        .merge(article_df, on='article_id',  how='left')
    )
    dataset = compute_cross_features(dataset)
    #5Imputamos los nulos
    dataset = imputar_nulos_tfm(dataset)
    #6 Convertimos los textos a categorías para ahorrar memoria RAM
    dataset = auto_optimize_categories(dataset,exclude_cols=['customer_id','article_id'])
    categorical_categories = {
        col: dataset[col].cat.categories
        for col in dataset.select_dtypes(include=['category']).columns
    }
    # 7. Separar X e y
    cols_no_feature = ['customer_id', 'article_id', 'label', 'sample_weight','detail_desc','prod_name']
    feature_cols    = [c for c in dataset.columns if c not in cols_no_feature]

    X = dataset[feature_cols].copy()
    y = dataset['label']
    sample_weight = dataset['sample_weight']
    print("\n" + "="*50)
    print("AUDITORÍA DE MATRIZ X (Entrenamiento)")
    print("="*50)
    print(f"Dimensiones de X: {X.shape[0]} filas x {X.shape[1]} columnas")
    print("\nListado de variables y sus tipos de datos:")

    print(f"Positivos: {len(positivos):,}  |  Negativos: {len(negativos):,}")
    print(f"X shape: {X.shape}  |  Features: {len(feature_cols)}")

    return X, y, sample_weight, dataset, article_df, user_df, categorical_categories,candidate_pool


def encode_xgboost_categoricals(user_df, article_df, feature_config=None):
    """
    Aplica el one-hot encoding de usuario y artículo usado por XGBoost.

    Se comparte entre xgboost_preprocess (entrenamiento) y el servido en vivo
    de recomendaciones, para que ambos generen exactamente las mismas columnas.
    Acepta feature_config para controlar qué features se incluyen.
    """
    if feature_config is None:
        feature_config = FEATURE_CONFIG_DEFAULT

    # --- Usuario ---
    user_num = [c for c in feature_config["user_numeric"] if c in user_df.columns]
    user_cat = [c for c in feature_config["user_categorical"] if c in user_df.columns]
    user_sub = user_df[["customer_id"] + user_num + user_cat].copy()
    user_encoded = (
        pd.get_dummies(user_sub, columns=user_cat, dummy_na=False) if user_cat else user_sub
    )

    # --- Artículo ---
    art_num = [c for c in feature_config["article_numeric"] if c in article_df.columns]
    art_cat = [c for c in feature_config["article_categorical"] if c in article_df.columns]
    article_sub = article_df[["article_id"] + art_num + art_cat].copy()
    article_encoded = (
        pd.get_dummies(article_sub, columns=art_cat, dummy_na=False) if art_cat else article_sub
    )

    return user_encoded, article_encoded


def recommend_xgboost_for_user(model, user_features, candidate_df, feature_cols, categorical_cols,categorical_categories, top_n=12):
    """
    Rankea un pool fijo de artículos candidatos para un usuario con un XGBClassifier ya entrenado.

    user_features    : dict o Series con las features de UN usuario sin codificar.
    candidate_df     : DataFrame con 'article_id' + features de artículo sin codificar.
    feature_cols     : columnas exactas usadas en el entrenamiento (X.columns).
    categorical_cols : lista de columnas que se convirtieron a 'category' en el train.
    categorical_categories : lista con los valores de cada categoría del train para que puedas decodificar las predicciones
    """
      
   # 1 Nueva construcción más eficiente replicamos columna a columna en vez de pd.concat()
    combined = candidate_df.copy()
    
    if isinstance(user_features, pd.DataFrame):
        user_dict = user_features.iloc[0].to_dict()
    else:
        user_dict = dict(user_features)
        
    for col, val in user_dict.items():
        combined[col] = val
    

    # 2. Calcular cross_features AHORA que usuario y artículo están en la misma fila
    combined = compute_cross_features(combined)
    
    # 3. Limpieza idéntica a la del entrenamiento
    combined = imputar_nulos_tfm(combined)

    # 4. Alinear columnas con feature_cols (Como tu original, pero SIN el .astype(float) global)
    X_infer = combined.reindex(columns=feature_cols)
 
    # 5. Forzar el tipo 'category' SOLO a las columnas que XGBoost espera
    for col in categorical_cols:
        if col in X_infer.columns:
            cats = categorical_categories.get(col)
            X_infer[col] = pd.Categorical(X_infer[col], categories=cats)

    # 6. Para el resto de columnas, nos aseguramos de que sean numéricas nativas (float/int)
    # y evitamos cualquier tipo 'object' residual que rompa el predict.
    cols_numericas = [c for c in X_infer.columns if c not in categorical_cols]
    X_infer[cols_numericas] = X_infer[cols_numericas].astype(float)

    # 7. Predicción y ranking
    scores = model.predict_proba(X_infer)[:, 1]
    # Asignamos score al article_block original para no perder IDs ni ensuciar datos
    ranked = candidate_df.assign(score=scores).sort_values('score', ascending=False)
    
    return ranked.head(top_n)[['article_id', 'score']].reset_index(drop=True)


# ==========================================
# MÉTRICAS DE EVALUACIÓN
# ==========================================
def apk(actual, predicted, k=12):
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapk(actual, predicted, k=12):
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

# ==========================================
# MODELOS BASE
# ==========================================

# ==========================================
# FILTRADO COLABORATIVO ÍTEM-ÍTEM
# ==========================================
# Imports necesarios: numpy y pandas ya están en models.py; añadir estos dos:
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize


def build_interaction_matrix(df_train, time_decay_halflife_days=None):
    """
    Construye la matriz dispersa usuario x artículo a partir de las transacciones.

    Cada celda vale 1 si el usuario compró el artículo (feedback implícito).
    Si time_decay_halflife_days no es None, en lugar de 1 se usa un peso
    temporal exp(-ln(2) * dias_desde_compra / halflife): una compra de hace
    `halflife` días pesa 0.5, de hace 2*halflife pesa 0.25, etc. En moda,
    la co-compra reciente es más informativa que la antigua.

    Devuelve:
        R          : csr_matrix (n_usuarios x n_articulos)
        user_index : pd.Index para mapear customer_id -> fila
        item_index : pd.Index para mapear article_id  -> columna
    """
    df = df_train[['customer_id', 'article_id']].copy()

    if time_decay_halflife_days is not None:
        max_date = df_train['t_dat'].max()
        dias = (max_date - df_train['t_dat']).dt.days.values
        pesos = np.exp(-np.log(2) * dias / time_decay_halflife_days)
    else:
        pesos = np.ones(len(df))

    user_codes, user_index = pd.factorize(df['customer_id'], sort=True)
    item_codes, item_index = pd.factorize(df['article_id'], sort=True)

    # Si un usuario compró el mismo artículo varias veces, los pesos se suman
    # (csr_matrix agrega duplicados automáticamente): la recompra refuerza la señal.
    R = csr_matrix(
        (pesos, (user_codes, item_codes)),
        shape=(len(user_index), len(item_index)),
    )
    return R, user_index, item_index


def fit_item_item(R, top_n_similares=100):
    """
    Calcula la matriz de similitud del coseno ítem-ítem.

    R : csr_matrix usuario x artículo.
    top_n_similares : para cada artículo, se conservan solo sus top_n vecinos
        más similares y el resto se pone a 0. Esto mantiene la matriz dispersa
        (con catálogos grandes, la matriz completa no cabe en memoria) y actúa
        como el 'k vecinos' clásico del filtrado colaborativo por vecindad.

    Devuelve S : csr_matrix (n_articulos x n_articulos) con la similitud coseno.
    """
    # Normalizamos las columnas (vectores de artículo) a norma L2 = 1;
    # así R_norm.T @ R_norm es directamente la similitud del coseno.
    R_norm = normalize(R.tocsc(), norm='l2', axis=0)
    S = (R_norm.T @ R_norm).tocsr()

    # Quitamos la diagonal (similitud de un artículo consigo mismo = 1):
    # no queremos que un artículo se recomiende "por parecerse a sí mismo".
    S.setdiag(0)
    S.eliminate_zeros()

    # Poda: nos quedamos con los top_n vecinos por fila
    if top_n_similares is not None:
        S = _keep_top_n_per_row(S, top_n_similares)

    return S


def _keep_top_n_per_row(S, n):
    """Conserva los n valores más altos de cada fila de una csr_matrix."""
    S = S.tocsr()
    data, indices, indptr = [], [], [0]
    for i in range(S.shape[0]):
        row_start, row_end = S.indptr[i], S.indptr[i + 1]
        row_data = S.data[row_start:row_end]
        row_idx = S.indices[row_start:row_end]
        if len(row_data) > n:
            top = np.argpartition(row_data, -n)[-n:]
            row_data, row_idx = row_data[top], row_idx[top]
        data.append(row_data)
        indices.append(row_idx)
        indptr.append(indptr[-1] + len(row_data))
    return csr_matrix(
        (np.concatenate(data), np.concatenate(indices), np.array(indptr)),
        shape=S.shape,
    )


def predict_item_item(
    df_train,
    users_list,
    k=12,
    top_n_similares=100,
    time_decay_halflife_days=None,
):
    """
    Filtrado colaborativo ítem-ítem sobre feedback implícito (compras).

    Para cada usuario: puntuación de cada artículo candidato = suma de
    similitudes coseno entre el candidato y los artículos que el usuario
    compró. Se excluyen los ya comprados y se devuelve el top-k.

    Mismo contrato que predict_random / predict_popular:
        (df_train, users_list, k) -> lista de listas de article_id.

    time_decay_halflife_days : None -> matriz binaria clásica.
        Un número (p.ej. 90) -> las compras recientes pesan más, tanto al
        calcular similitudes como al puntuar candidatos.
    """
    R, user_index, item_index = build_interaction_matrix(
        df_train, time_decay_halflife_days=time_decay_halflife_days
    )
    S = fit_item_item(R, top_n_similares=top_n_similares)

    # Puntuaciones de todos los artículos para todos los usuarios de una vez:
    # scores[u] = r_u @ S  (suma de similitudes con lo comprado por u)
    user_pos = {u: i for i, u in enumerate(user_index)}
    item_ids = item_index.to_numpy()

    predictions = []
    for u in users_list:
        if u not in user_pos:
            predictions.append([])          # usuario sin historial en train
            continue

        r_u = R[user_pos[u]]                # vector disperso de compras del usuario
        scores = np.asarray((r_u @ S).todense()).ravel()

        # Excluimos lo ya comprado
        comprados = r_u.indices
        scores[comprados] = -np.inf

        n_validos = int(np.sum(np.isfinite(scores) & (scores > 0)))
        if n_validos == 0:
            predictions.append([])
            continue

        top = min(k, n_validos)
        cand = np.argpartition(scores, -top)[-top:]
        cand = cand[np.argsort(scores[cand])[::-1]]
        predictions.append(item_ids[cand].tolist())

    return predictions

    
def predict_random(df_train, users_list, k=12, seed=42):
    """
    Genera k predicciones aleatorias para una lista de usuarios.
    """
    np.random.seed(seed)
    todos_los_articulos = df_train['article_id'].unique()
    
    # Generamos la matriz de predicciones
    predictions = [np.random.choice(todos_los_articulos, k, replace=False).tolist() for _ in range(len(users_list))]
    return predictions

def predict_popular(df_train, users_list, k=12):
    """
    Recomienda los k artículos más vendidos del histórico a todos los usuarios.
    """
    top_k_articulos = df_train['article_id'].value_counts().head(k).index.tolist()

    # Matriz donde todos reciben la misma recomendación top
    predictions = [top_k_articulos for _ in range(len(users_list))]
    return predictions

def predict_cluster(df_transactions, df_customers, df_products, customer_ids, K=8, top_n=12, explore_k=False):

    X_final, article_ids, _, df_products_enriched = clustering_preprocess(df_customers, df_products, df_transactions)

    X_df = pd.DataFrame(X_final, index=article_ids)

    if explore_k:
        find_optimal_k(X_final, k_range=range(2, 15))

    df_clusters, kmeans_model = fit_product_clustering(X_final, K, article_ids)

    df_merged, summary = inspect_clusters(
        df_products=df_products_enriched,
        df_clusters=df_clusters,
        category_col='product_group_name',
    )

    predictions = []
    for customer_id in customer_ids:
        recs = recommend_by_cluster_similarity(
            customer_id=customer_id,
            df_transactions=df_transactions,
            df_clusters_with_price=df_merged,
            X_df=X_df,
            top_n=top_n,
        )
        predictions.append(recs['article_id'].tolist() if not recs.empty else [])

    return predictions, df_merged, summary, kmeans_model, X_final


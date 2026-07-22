import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from implicit.bpr import BayesianPersonalizedRanking
import xgboost as xgb
from utils.preprocess import auto_optimize_categories, imputar_nulos_tfm
import os 
import json
import time

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
    # Ya se calculaban en compute_article_features pero no se usaban como
    # feature: momentum de ventas (tendencia) y antigüedad del producto,
    # ambas habituales en las soluciones de la competición H&M de Kaggle.
    'sales_last_30d',
    'product_age_days',
]

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
        'recency_days': 999.0,
        'sales_last_30d':0.0,
        'product_age_days': 0.0  # Valor elevado para evitar sesgos en el modelo
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

def compute_cross_features(dataset, historial_dict=None, cov_dict=None,cov_decay=0.7,usar_log=False):
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
        # 1 si es su sección favorita, 0 si no. Comparamos por valor (texto),
        # no por categoría: section_name y user_favorite_section son dtype
        # 'category' con conjuntos de categorías distintos (una es de
        # artículo, la otra de usuario), y pandas no deja comparar
        # Categoricals directamente si sus categorías no coinciden.
        dataset['cross_is_favorite_section'] = (
            dataset['section_name'].astype(str) == dataset['user_favorite_section'].astype(str)
        ).astype('int8')
    # 4. CO-VISITATION SCORE (Relación basada en historial)
    if historial_dict is not None and cov_dict is not None:
        dataset['covisitation_score'] = covisitation_scores_batch(
            dataset, historial_dict, cov_dict, decay_factor=cov_decay, usar_log=usar_log,
        )

    return dataset

def generar_historial_dict(df_train, n_recientes=None):
    """
    Genera un diccionario con el historial de compras por usuario.
    
    Parameters
    ----------
    df_train : DataFrame
        Datos de entrenamiento con 'customer_id', 'article_id' y 't_dat'.
    n_recientes : int, opcional
        Si se especifica, toma solo los últimos 'n' artículos comprados.
        Si es None o 0, utiliza todo el historial del usuario.
    """
    # Ordenar por cliente y fecha de manera descendente (lo más reciente primero)
    df_sorted = df_train.sort_values(by=['customer_id', 't_dat'], ascending=[True, False])
    
    # Recortar a los n más recientes si se especifica
    if n_recientes:
        df_sorted = df_sorted.groupby('customer_id').head(n_recientes)
        
    # Agrupar y convertir a diccionario
    historial_dict = (
        df_sorted
        .groupby('customer_id')['article_id']
        .apply(list)
        .to_dict()
    )
    
    return historial_dict

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
    
    #print("Customer profile: ", X_df.loc[bought_valid].mean(axis=0).to_frame().T)

    return X_df.loc[bought_valid].mean(axis=0).to_frame().T

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
    positivos_cliente,
    cliente,
    compras_por_cliente,
    todos_los_articulos,
    prob_muestreo,
    n_negativos_por_positivo,
    rng,
):
    
    comprados = compras_por_cliente.get(cliente, set())
    articulos_usados = comprados.copy() #para eviatar duplicados (muy improbable, pero por si acaso)
    n_necesarios = len(positivos_cliente) * n_negativos_por_positivo
    n_generados = 0
    intentos = 0
    batch_size = min(n_necesarios * 3, len(todos_los_articulos))

    negativos = []

    while n_generados < n_necesarios and intentos < n_necesarios * 20:
        candidatos = rng.choice(
            todos_los_articulos,
            size=batch_size,
            p=prob_muestreo,
            replace=True
        )

        intentos += len(candidatos)

        for c in candidatos:
            if c not in articulos_usados:
                negativos.append({
                    "customer_id": cliente,
                    "article_id": c,
                    "label": 0,
                })
                articulos_usados.add(c)
                n_generados += 1

                if n_generados >= n_necesarios:
                    break

    return negativos

def train_bpr_model(df_transactions, factors=32, iterations=15, learning_rate=0.05, regularization=0.01, random_state=42):
    """
    Entrena un modelo de matrix factorization BPR (Bayesian Personalized
    Ranking, vía la librería `implicit`) sobre las compras de
    df_transactions, para capturar señal puramente colaborativa
    ("clientes que compraron lo mismo que tú también compraron Z") que ni el
    clustering por atributos de producto ni las features hechas a mano
    recogen. No se usa como modelo de recomendación aparte: sus vectores
    latentes se usan como UNA feature más (bpr_score) dentro de XGBoost, ver
    xgboost_preprocess.

    Devuelve (user_factors_df, item_factors_df): un DataFrame indexado por
    customer_id / article_id con su vector latente, listo para hacer
    producto escalar contra cualquier candidato. Un customer_id o article_id
    que no aparezca en df_transactions simplemente no tiene fila (cold start:
    se trata luego como vector cero, sin señal).
    """
    interacciones = df_transactions[['customer_id', 'article_id']].drop_duplicates()
    user_codes = interacciones['customer_id'].astype('category')
    article_codes = interacciones['article_id'].astype('category')

    user_items = sp.csr_matrix(
        (np.ones(len(interacciones), dtype=np.float32), (user_codes.cat.codes, article_codes.cat.codes)),
        shape=(len(user_codes.cat.categories), len(article_codes.cat.categories)),
    )

    bpr_model = BayesianPersonalizedRanking(
        factors=factors, learning_rate=learning_rate, regularization=regularization,
        iterations=iterations, random_state=random_state,
    )
    bpr_model.fit(user_items, show_progress=False)

    # Nombres de columna en texto (no enteros): hace falta para poder guardar
    # estos factores en parquet como el resto de artefactos del pipeline.
    factor_cols = [f"bpr_f{i}" for i in range(bpr_model.user_factors.shape[1])]
    user_factors_df = pd.DataFrame(bpr_model.user_factors, index=user_codes.cat.categories, columns=factor_cols)
    item_factors_df = pd.DataFrame(bpr_model.item_factors, index=article_codes.cat.categories, columns=factor_cols)
    return user_factors_df, item_factors_df


def bpr_dot_scores(ids, id_factors_df, other_vector):
    """
    Producto escalar entre other_vector (el vector BPR de UN usuario o UN
    artículo) y el vector de cada elemento de `ids` en id_factors_df.
    ids sin vector conocido (cold start) puntúan 0.0 (neutral).
    Vectorizado: sirve igual para 1 candidato que para 50.000.
    """
    matrix = id_factors_df.reindex(ids).fillna(0.0).to_numpy()
    return matrix @ other_vector

def get_or_train_bpr(df_transactions, bpr_params, cache_dir="model_cache"):
    """
    Controlador de caché para el modelo BPR. 
    Comprueba si los parámetros o los datos han cambiado antes de reentrenar.
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    config_path = os.path.join(cache_dir, "bpr_config.json")
    user_factors_path = os.path.join(cache_dir, "user_factors.parquet")
    item_factors_path = os.path.join(cache_dir, "item_factors.parquet")
    
    # 1. Definir el estado actual (Parámetros + Huella dactilar de los datos)
    current_config = {
        # Parámetros del modelo
        "factors": bpr_params.get("factors", 32),
        "iterations": bpr_params.get("iterations", 15),
        "learning_rate": bpr_params.get("learning_rate", 0.05),
        "regularization": bpr_params.get("regularization", 0.01),
        "random_state": bpr_params.get("random_state", 42),
        
        # Huella dactilar de los datos (detecta si cambia el train_set)
        "num_transactions": len(df_transactions),
        "num_unique_users": df_transactions['customer_id'].nunique(),
        "num_unique_items": df_transactions['article_id'].nunique()
    }
    
    # 2. Comprobar si existe caché y si la configuración es idéntica
    if os.path.exists(config_path) and os.path.exists(user_factors_path) and os.path.exists(item_factors_path):
        with open(config_path, "r") as f:
            cached_config = json.load(f)
            
        if cached_config == current_config:
            print("✅ BPR Cache: Parámetros y datos intactos. Cargando matrices desde disco...")
            user_factors_df = pd.read_parquet(user_factors_path)
            item_factors_df = pd.read_parquet(item_factors_path)
            return user_factors_df, item_factors_df
        else:
            print("⚠️ BPR Cache: Detectados cambios en datos o parámetros. Reentrenando...")
    else:
        print("🔍 BPR Cache: No se encontró caché. Entrenando desde cero...")
        
    # 3. Si no hay caché válida, entrenamos
    user_factors_df, item_factors_df = train_bpr_model(
        df_transactions, 
        factors=current_config["factors"],
        iterations=current_config["iterations"],
        learning_rate=current_config["learning_rate"],
        regularization=current_config["regularization"],
        random_state=current_config["random_state"]
    )
    
    os.makedirs(cache_dir,exist_ok = True)
    # 4. Guardamos las nuevas matrices y el nuevo archivo de control
    user_factors_df.to_parquet(user_factors_path)
    item_factors_df.to_parquet(item_factors_path)
    
    with open(config_path, "w") as f:
        json.dump(current_config, f, indent=4)
        
    return user_factors_df, item_factors_df


def generar_candidatos_bpr_batch(clientes, user_factors_df, item_factors_df, compras_por_cliente, top_k=100):
    """
    Genera los top_k candidatos BPR para una lista de clientes procesando en bloques (batches)
    para maximizar la velocidad usando multiplicación de matrices pura.
    """
    batch_size=2048
    item_ids = item_factors_df.index.to_numpy()
    # Trasponemos la matriz de artículos para que la multiplicación sea directa: (B, K) @ (K, N) = (B, N)
    item_matrix_T = item_factors_df.to_numpy().T 
    
    # Diccionario rápido para saber en qué columna quedó cada artículo
    item_id_to_idx = {item_id: idx for idx, item_id in enumerate(item_ids)}
    
    resultados = []
    
    # Iterar sobre los clientes en bloques gigantes
    for i in range(0, len(clientes), batch_size):
        batch_clientes = clientes[i : i + batch_size]
        
        # Extraemos la sub-matriz de usuarios del bloque (Cold-start se rellena con 0)
        user_batch_matrix = user_factors_df.reindex(batch_clientes).fillna(0.0).to_numpy()
        
        # MAGIA AQUÍ: Calculamos todos los scores del bloque de golpe
        # scores_batch tendrá tamaño (tamaño_del_bloque, total_articulos_catalogo)
        scores_batch = user_batch_matrix @ item_matrix_T
        
        # Procesar los resultados de cada usuario dentro del bloque
        for j, cliente in enumerate(batch_clientes):
            scores_usuario = scores_batch[j]
            
            # Penalizar comprados
            comprados = compras_por_cliente.get(cliente, set())
            if comprados:
                indices_comprados = [item_id_to_idx[item] for item in comprados if item in item_id_to_idx]
                scores_usuario[indices_comprados] = -np.inf
            
            # Extraer Top-K
            if len(scores_usuario) > top_k:
                # argpartition separa los Top K súper rápido (pero sin ordenarlos internamente)
                top_k_indices = np.argpartition(scores_usuario, -top_k)[-top_k:]
                # Ordenamos solo esos K mejores de mayor a menor score
                top_k_indices = top_k_indices[np.argsort(-scores_usuario[top_k_indices])]
            else:
                top_k_indices = np.argsort(-scores_usuario)
                
            top_k_articulos = item_ids[top_k_indices]
            top_k_scores = scores_usuario[top_k_indices]
            
            # Guardar resultados
            for art_id, score in zip(top_k_articulos, top_k_scores):
                resultados.append({
                    "customer_id": cliente,
                    "article_id": art_id,
                    "bpr_score": score
                })
                       
    return pd.DataFrame(resultados)

def generar_candidatos_cluster(clientes, compras_por_cliente, df_articles, df_train, top_k=100):
    """
    Genera candidatos de artículos proporcionalmente a los clústers históricos del usuario.
    Devuelve estrictamente un DataFrame con ['customer_id', 'article_id'].
    """
    # 1. Calcular popularidad dentro de cada clúster
    ventas_cluster = df_train[['article_id']].merge(
        df_articles[['article_id', 'cluster']], on='article_id', how='inner'
    )
    populares_cluster = ventas_cluster.groupby(['cluster', 'article_id']).size().reset_index(name='ventas')
    top_por_cluster = populares_cluster.sort_values(['cluster', 'ventas'], ascending=[True, False])

    # 2. Diccionarios de acceso rápido
    items_por_cluster_dict = top_por_cluster.groupby('cluster')['article_id'].apply(list).to_dict()
    item_to_cluster = dict(zip(df_articles['article_id'], df_articles['cluster']))

    resultados = []

    for cliente in clientes:
        comprados = compras_por_cliente.get(cliente, set())
        
        if not comprados:
            continue
            
        clusters_comprados = [item_to_cluster[item] for item in comprados if item in item_to_cluster]
        
        if not clusters_comprados:
            continue
            
        freq_clusters = pd.Series(clusters_comprados).value_counts(normalize=True)
        cupos_por_cluster = (freq_clusters * top_k).round().astype(int)
        
        candidatos_usuario = []
        for cluster, cupo in cupos_por_cluster.items():
            if cupo <= 0: continue
            
            candidatos_potenciales = items_por_cluster_dict.get(cluster, [])
            candidatos_validos = [item for item in candidatos_potenciales if item not in comprados]
            
            candidatos_usuario.extend(candidatos_validos[:cupo])
            
        # Solo guardamos el par usuario-artículo
        for art_id in candidatos_usuario:
            resultados.append({
                "customer_id": cliente,
                "article_id": art_id
            })

    return pd.DataFrame(resultados)

###FUNCIONES PARA COVISITACION
def construir_o_cargar_covisitacion(
    df_train,
    cache_dir="backend/bpr_cache",
    max_pairs_per_item=100,
    min_co_purchases=1
):
    """
    Construye (o carga desde caché) la matriz de co-visitación.

    Dos artículos co-visitan si aparecen en el mismo transaction_id.

    Parameters
    ----------
    max_pairs_per_item : int
        Número máximo de vecinos que se conservan para cada artículo.

    min_co_purchases : int
        Número mínimo de compras conjuntas para conservar un par.
    """

    cache_path = os.path.join(cache_dir, "covisitation_matrix.parquet")
    config_path = os.path.join(cache_dir, "covisitation_config.json")
    os.makedirs(cache_dir, exist_ok=True)

    # Huella de parámetros + datos (mismo patrón que get_or_train_bpr): si
    # cambia cualquiera de estos, la caché queda invalidada automáticamente.
    current_config = {
        "max_pairs_per_item": max_pairs_per_item,
        "min_co_purchases": min_co_purchases,
        "num_transactions": len(df_train),
        "num_unique_customers": df_train["customer_id"].nunique(),
        "num_unique_articles": df_train["article_id"].nunique(),
    }

    if os.path.exists(cache_path) and os.path.exists(config_path):
        with open(config_path, "r") as f:
            cached_config = json.load(f)

        if cached_config == current_config:
            print("✅ Covisitación Cache: Parámetros y datos intactos. Cargando desde disco...")
            return pd.read_parquet(cache_path)
        else:
            print("⚠️ Covisitación Cache: Detectados cambios en datos o parámetros. Reconstruyendo...")
    else:
        print("🔍 Covisitación Cache: No se encontró caché. Construyendo desde cero...")

    print("Construyendo matriz de co-visitation...")

    # Cada artículo solo cuenta una vez por ticket
    df_unique = (
        df_train[
            ["transaction_id", "article_id"]
        ]
        .drop_duplicates()
    )

    # Todas las combinaciones de artículos dentro del mismo ticket
    df_pairs = df_unique.merge(
        df_unique,
        on="transaction_id",
        suffixes=("_A", "_B")
    )

    # Eliminar pares consigo mismos
    df_pairs = df_pairs[
        df_pairs.article_id_A != df_pairs.article_id_B
    ]

    # Número de veces que aparecen juntos
    co_counts = (
        df_pairs
        .groupby(
            ["article_id_A", "article_id_B"]
        )
        .size()
        .reset_index(name="co_purchases")
    )

    # Eliminar relaciones muy débiles
    co_counts = co_counts[
        co_counts.co_purchases >= min_co_purchases
    ]

    # Ordenar por frecuencia
    co_counts = co_counts.sort_values(
        ["article_id_A", "co_purchases"],
        ascending=[True, False]
    )

    # Conservar únicamente los mejores vecinos
    top_covisitation = (
        co_counts
        .groupby("article_id_A")
        .head(max_pairs_per_item)
    )

    top_covisitation.to_parquet(cache_path, index=False)
    with open(config_path, "w") as f:
        json.dump(current_config, f, indent=4)

    return top_covisitation

def construir_diccionario_covisitacion(df_covisitation):
    """
    Devuelve un diccionario:

    {
        article_A:
        {
            article_B: co_purchases,
            article_C: co_purchases,
            ...
        }
    }
    """

    return (
        df_covisitation
        .groupby("article_id_A")
        .apply(
            lambda x: dict(
                zip(
                    x["article_id_B"],
                    x["co_purchases"]
                )
            )
        )
        .to_dict()
    )

import numpy as np


def calcular_covisitation_score(
    historial_usuario,
    articulo_candidato,
    cov_dict,
    decay_factor=0.7,
    usar_log=False
):
    """
    Calcula el grado de afinidad entre un usuario
    (representado por su historial reciente)
    y un artículo candidato.
    """

    score = 0.0

    for i, articulo_historial in enumerate(historial_usuario):

        peso = decay_factor ** i

        raw_score = (
            cov_dict
            .get(articulo_historial, {})
            .get(articulo_candidato, 0)
        )

        if usar_log:
            raw_score = np.log1p(raw_score)

        score += peso * raw_score

    return score


def covisitation_score_dict(historial, cov_dict, decay_factor=0.7, usar_log=False):
    """
    Diccionario {article_id candidato: score} para UN historial de usuario
    (lista de article_id, más reciente primero), sumando con decaimiento por
    recencia los vecinos de co-visitación de cada artículo del historial.

    Es el núcleo que comparten covisitation_scores_batch (entrenamiento, un
    dataset con muchas filas por cliente), assign_covisitation_score (servido
    en vivo, un único usuario por petición) y generar_candidatos_covisitacion
    (generación de candidatos) — para no repetir la misma lógica en varios sitios.
    """
    score_dict = {}
    for i, articulo_historial in enumerate(historial):
        peso = decay_factor ** i
        for vecino, co_purchases in cov_dict.get(articulo_historial, {}).items():
            valor = np.log1p(co_purchases) if usar_log else co_purchases
            score_dict[vecino] = score_dict.get(vecino, 0.0) + peso * valor
    return score_dict


def covisitation_scores_batch(dataset, historial_dict, cov_dict, decay_factor=0.7, usar_log=False):
    """
    Versión vectorizada de calcular_covisitation_score para un dataset entero.

    calcular_covisitation_score recorre el historial completo del cliente por
    cada FILA (customer_id, article_id) del dataset — con varias filas por
    cliente (positivos + negativos en entrenamiento), repite la misma suma
    ponderada una y otra vez para el mismo cliente. Aquí se agrupa por
    customer_id único: para cada cliente se construye una sola vez su
    diccionario {article_id candidato: score} (covisitation_score_dict), y
    luego se aplica a todas sus filas con un simple .map().

    Devuelve una Series alineada con el índice de `dataset`, lista para
    asignar como dataset['covisitation_score'].
    """
    resultado = pd.Series(0.0, index=dataset.index)

    for cliente, filas in dataset.groupby('customer_id').groups.items():
        historial = historial_dict.get(cliente, [])
        if not historial:
            continue

        score_dict = covisitation_score_dict(historial, cov_dict, decay_factor, usar_log)
        if not score_dict:
            continue

        articulos = dataset.loc[filas, 'article_id']
        resultado.loc[filas] = articulos.map(score_dict).fillna(0.0)

    return resultado


def assign_covisitation_score(candidate_df, historial, cov_dict, decay_factor=0.7, usar_log=False):
    """
    Añade la columna 'covisitation_score' a candidate_df para UN usuario cuyo
    historial de compras (lista de article_id, más reciente primero) se pasa
    directamente. Pensada para el servido en vivo / bucle de evaluación (un
    usuario a la vez) — para un dataset con muchas filas por cliente conviene
    covisitation_scores_batch en su lugar.
    """
    if not historial:
        return candidate_df.assign(covisitation_score=0.0)
    score_dict = covisitation_score_dict(historial, cov_dict, decay_factor, usar_log)
    return candidate_df.assign(
        covisitation_score=candidate_df["article_id"].map(score_dict).fillna(0.0)
    )


def generar_candidatos_covisitacion(clientes, historial_dict, cov_dict, top_k=20, usar_log=True):
    """
    Genera hasta top_k candidatos por cliente a partir de la co-visitación de
    su historial reciente ("quien compró esto también compró..."). Mismo
    contrato de salida que generar_candidatos_bpr_batch/generar_candidatos_cluster:
    DataFrame con ['customer_id', 'article_id'].
    """
    resultados = []
    for cliente in clientes:
        historial = historial_dict.get(cliente, [])
        if not historial:
            continue

        score_dict = covisitation_score_dict(historial, cov_dict, decay_factor=0.7, usar_log=usar_log)
        if not score_dict:
            continue

        comprados = set(historial)
        candidatos_validos = {
            articulo: score for articulo, score in score_dict.items() if articulo not in comprados
        }
        top_articulos = sorted(candidatos_validos.items(), key=lambda x: x[1], reverse=True)[:top_k]

        for articulo, _score in top_articulos:
            resultados.append({"customer_id": cliente, "article_id": articulo})

    return pd.DataFrame(resultados, columns=["customer_id", "article_id"])


def generar_candidatos_populares(clientes, df_articles, top_k=20):
    """
    Candidatos "de tendencia" (cold-start): los top_k artículos más vendidos
    del catálogo, iguales para todos los clientes. Mismo contrato de salida
    que el resto de generar_candidatos_*: DataFrame con ['customer_id', 'article_id'].
    """
    top_populares = (
        df_articles[["article_id", "sales_volume"]]
        .sort_values("sales_volume", ascending=False)
        .head(top_k)["article_id"]
        .tolist()
    )
    resultados = [
        {"customer_id": cliente, "article_id": articulo}
        for cliente in clientes
        for articulo in top_populares
    ]
    return pd.DataFrame(resultados, columns=["customer_id", "article_id"])


def xgboost_preprocess(
    df_customers, df_products, df_transactions, n_negativos_por_positivo=4, random_state=42,
    bpr_factors=32, bpr_iterations=15, bpr_regularization=0.01,
    historial_dict=None, cov_dict=None,
):
    rng = np.random.default_rng(random_state)

    # 1. Features de artículo y usuario
    article_df = compute_article_features(df_customers, df_products, df_transactions)
    user_df    = compute_user_features(df_customers, df_transactions, df_products)

    # 2. Muestras positivas: pares únicos (cliente, artículo) realmente comprados
    positivos = df_transactions[['customer_id', 'article_id']].drop_duplicates().copy()
    positivos['label'] = 1
    t0_bpr = time.time()
    # 3. Usamos el modelo BPR antes de generar negativos por si lo usamos alli
    bpr_params = {
        "factors": bpr_factors,
        "iterations": bpr_iterations,
        "regularization": bpr_regularization,
        "random_state": random_state
    }
    print(f"Tiempo BPR: {time.time() - t0_bpr:.2f}s")
    # 3a Llamamos a nuestra función 'inteligente' con caché en lugar de train_bpr_model
    #si ya está entrenado, usamos el cache
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # Definimos la carpeta de caché dentro de backend/
    CACHE_DIR = os.path.join(BASE_DIR, "bpr_cache")
    user_factors_df, item_factors_df = get_or_train_bpr(df_transactions, bpr_params,cache_dir=CACHE_DIR)
    # 4. Negative sampling ponderado por popularidad (artículos más vendidos
    #    tienen más probabilidad de ser muestreados como negativos — más realista)
    print('Generando negativos...')
    t0_neg = time.time()
    todos_los_articulos = article_df['article_id'].values
    popularidad = (
        article_df.set_index('article_id')['sales_volume']
        .reindex(todos_los_articulos).fillna(1).values
    )
    popularidad = np.where(popularidad <= 0, 1, popularidad)
    prob_muestreo = popularidad / popularidad.sum()
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
                todos_los_articulos=todos_los_articulos,
                prob_muestreo=prob_muestreo,
                n_negativos_por_positivo=n_negativos_por_positivo,
                rng=rng,
            )
        )
    negativos = pd.DataFrame(negativos_rows)

    dataset   = pd.concat([positivos, negativos], ignore_index=True)
    print(f"Tiempo generación de negativos: {time.time() - t0_neg:.2f}s")
    # 3b. Señal colaborativa (BPR): "quién compró qué", independiente de los
    #     atributos de producto. Se añade como una feature más (bpr_score) =
    #     producto escalar entre el vector latente del cliente y el del
    #     artículo, vectorizado para todo el dataset a la vez.
    
    user_matrix = user_factors_df.reindex(dataset['customer_id']).fillna(0.0).to_numpy()
    item_matrix = item_factors_df.reindex(dataset['article_id']).fillna(0.0).to_numpy()
    dataset['bpr_score'] = (user_matrix * item_matrix).sum(axis=1)

    # 4. Encoding de categóricas de usuario y artículo (compartido con el servido en vivo)
    user_encoded, article_encoded = encode_xgboost_categoricals(user_df, article_df)
    
    # 5. Join final: dataset × features de usuario × features de artículo
    dataset = (
        dataset
        .merge(user_encoded,       on='customer_id', how='left')
        .merge(article_encoded, on='article_id',  how='left')
    )
    # 5b. Cross features usuario-artículo (afinidad de precio/edad/sección):
    #     puramente aritméticas sobre columnas ya presentes, sin depender de
    #     si ese artículo en concreto se compró (a diferencia de un flag de
    #     recompra, esto no se filtra con el label y no tiene fuga de datos).
    dataset = compute_cross_features(dataset, historial_dict=historial_dict, cov_dict=cov_dict)
    # La imputación de categóricas (p.ej. club_member_status -> 'GUEST') ya se
    # hizo en encode_xgboost_categoricals, antes del cast a 'category' (un
    # fillna posterior sobre una columna category rompe si el valor no es ya
    # una categoría existente).
    # 6. Separar X e y. Las categóricas (dtype 'category', ver
    #    encode_xgboost_categoricals) se dejan tal cual para que XGBoost las
    #    trate de forma nativa; solo las numéricas se rellenan y castean.
    cols_no_feature = ['customer_id', 'article_id', 'label']
    feature_cols    = [c for c in dataset.columns if c not in cols_no_feature]

    X = dataset[feature_cols].copy()
    num_feature_cols = [c for c in feature_cols if not isinstance(X[c].dtype, pd.CategoricalDtype)]
    X[num_feature_cols] = X[num_feature_cols].fillna(0).astype(float)
    y = dataset['label']

    print(f"Positivos: {len(positivos):,}  |  Negativos: {len(negativos):,}")
    print(f"X shape: {X.shape}  |  Features: {len(feature_cols)}")

    return X, y, dataset, article_df, user_df, user_factors_df, item_factors_df


def encode_xgboost_categoricals(user_df, article_df):
    """
    Prepara las categóricas de usuario y artículo para las categóricas
    NATIVAS de XGBoost (enable_categorical=True): en vez de one-hot
    (pd.get_dummies, que antes explotaba a cientos de columnas y ralentizaba
    entrenamiento e inferencia), se dejan como dtype 'category' y es el
    propio XGBoost quien aprende los splits sobre esas categorías.

    Se comparte entre xgboost_preprocess (entrenamiento) y el servido en vivo
    de recomendaciones, para que ambos usen exactamente las mismas categorías.

    Importante: load_dataset() ya convierte columnas de texto a 'category'
    para ahorrar memoria (auto_optimize_categories), así que club_member_status
    puede llegar aquí siendo category SIN 'GUEST' entre sus categorías. Un
    fillna con un valor que no sea ya una categoría existente falla, así que
    se vuelve a texto plano antes de imputar y se re-categoriza después.
    """
    cat_user = [c for c in ['club_member_status', 'user_favorite_section'] if c in user_df.columns]
    user_encoded = user_df.copy()
    for col in cat_user:
        if isinstance(user_encoded[col].dtype, pd.CategoricalDtype):
            user_encoded[col] = user_encoded[col].astype(object)
    user_encoded = imputar_nulos_tfm(user_encoded)
    for col in cat_user:
        user_encoded[col] = user_encoded[col].astype('category')

    available_cat = [c for c in CATEGORICAL_FEATURES_XGBOOST if c in article_df.columns]
    article_encoded = article_df[['article_id'] + available_cat + NUMERIC_FEATURES_XGBOOST].copy()
    for col in available_cat:
        if isinstance(article_encoded[col].dtype, pd.CategoricalDtype):
            article_encoded[col] = article_encoded[col].astype(object)
    article_encoded = imputar_nulos_tfm(article_encoded)
    for col in available_cat:
        article_encoded[col] = article_encoded[col].astype('category')

    return user_encoded, article_encoded


def recommend_xgboost_for_user(model, user_features, candidate_df, feature_cols, top_n=12):
    """
    Rankea un pool fijo de artículos candidatos para un usuario con un XGBRanker ya entrenado.

    user_features : DataFrame de UNA fila con las features del usuario, ya
                    codificadas (salida de encode_xgboost_categoricals para
                    ese customer_id, p.ej. user_encoded_indexed.loc[[u]]).
                    Tiene que ser un DataFrame de 1 fila, no una Series: al
                    convertir una fila a Series se pierde el dtype
                    'category' de las columnas categóricas.
    candidate_df  : DataFrame con columna 'article_id' + features de artículo codificadas
                    (el 'article_encoded' de encode_xgboost_categoricals, filtrado al pool
                    de candidatos, p.ej. los artículos más vendidos).
    feature_cols  : columnas exactas usadas en el entrenamiento (X.columns de xgboost_preprocess),
                    para alinear el orden/presencia de columnas en la inferencia.
    """
    n = len(candidate_df)
    user_block = pd.concat([user_features] * n, ignore_index=True)
    article_block = candidate_df.reset_index(drop=True)

    combined = pd.concat([user_block, article_block], axis=1)
    combined = compute_cross_features(combined)
    combined = combined.reindex(columns=feature_cols)
    num_cols = [c for c in feature_cols if not isinstance(combined[c].dtype, pd.CategoricalDtype)]
    combined[num_cols] = combined[num_cols].fillna(0).astype(float)
    X_infer = combined

    scores = model.predict(X_infer)

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


def hits_at_k(actual, predicted, k=12):
    """Nº de artículos EXACTOS acertados (SKU real) en las k primeras recos de UN usuario."""
    predicted = predicted[:k]
    return sum(1 for p in predicted if p in actual)


def total_hits(actual_list, predicted_list, k=12):
    """Suma de aciertos exactos de artículo en toda la evaluación (todos los usuarios)."""
    return int(sum(hits_at_k(a, p, k) for a, p in zip(actual_list, predicted_list)))


def hit_rate(actual_list, predicted_list, k=12):
    """Fracción de usuarios con AL MENOS 1 acierto exacto en sus k primeras recos."""
    if not actual_list:
        return 0.0
    return float(np.mean([hits_at_k(a, p, k) > 0 for a, p in zip(actual_list, predicted_list)]))


def category_hit_rate(actual_categories, predicted_article_ids, article_to_category, k=12):
    """
    Métrica más laxa que MAP@12: en vez de exigir acertar el artículo exacto,
    mide qué fracción de las k primeras recomendaciones caen en una categoría
    "correcta" para ese usuario (actual_categories = un set de categorías).

    Sirve para distinguir un modelo que recomienda cosas del estilo del
    usuario pero no el SKU exacto (razonable) de uno que recomienda categorías
    que no tienen nada que ver (malo), algo que MAP@12 por sí solo no ve.
    """
    predicted = predicted_article_ids[:k]
    if not predicted or not actual_categories:
        return 0.0
    aciertos = sum(1 for a in predicted if article_to_category.get(a) in actual_categories)
    return aciertos / len(predicted)


def mean_category_hit_rate(actual_categories_list, predicted_list, article_to_category, k=12):
    return float(np.mean([
        category_hit_rate(cats, preds, article_to_category, k)
        for cats, preds in zip(actual_categories_list, predicted_list)
    ]))

# ==========================================
# MODELOS BASE
# ==========================================
def predict_random(df_products, users_list, k=12, seed=42):
    """
    Genera k predicciones aleatorias para una lista de usuarios, muestreando
    sobre el catálogo completo (df_products) para competir en el mismo
    universo de candidatos que el resto de modelos (Cluster, XGBoost).
    """
    np.random.seed(seed)
    todos_los_articulos = df_products['article_id'].unique()

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
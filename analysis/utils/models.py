import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity


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
    'user_avg_price',
    'sales_volume',
    'online_ratio',
    'recency_days',
    'user_n_compras',
    'article_price',
    'sex_popularity',
    'total_sales_volume',
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
            online_ratio=("sales_channel_id", lambda x: (x == 2).mean()),
            recency_days=("t_dat", lambda x: (max_date - x.max()).days),
        ).reset_index()
    )
 
    df = df_products.merge(avg_age, on="article_id", how="left")
    df = df.merge(article_sale_features, on="article_id", how="left")
 
    for col in NUMERIC_FEATURES_CLUSTER:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
 
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
            online_ratio=('sales_channel_id', lambda x: (x == 2).mean()),
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

def compute_user_features(df_customers, df_transactions):
    #Solo para XGBoost
    """Features de usuario calculadas SOLO con transacciones de train."""
    user_tx = df_transactions.groupby("customer_id").agg(
        user_n_compras=("article_id", "count"),
        user_precio_medio=("price", "mean"),
        user_precio_std=("price", "std"),
        user_online_ratio=("sales_channel_id", lambda x: (x == 2).mean()),
    ).reset_index()
    user_tx["user_precio_std"] = user_tx["user_precio_std"].fillna(0)
 
    cols_customer = [c for c in ["customer_id", "age", "club_member_status"] if c in df_customers.columns]
    user_df = user_tx.merge(df_customers[cols_customer], on="customer_id", how="left")
 
    if "age" in user_df.columns:
        user_df["age"] = user_df["age"].fillna(user_df["age"].median())
 
    return user_df

def xgboost_preprocess(df_customers, df_products, df_transactions, n_negativos_por_positivo=4, random_state=42):
    """
    Construye el dataset (X, y) listo para entrenar un XGBClassifier.
 
    Reutiliza compute_article_features() para no duplicar lógica con
    clustering_preprocess(). Añade:
      - features de usuario,
      - negative sampling ponderado por popularidad (sales_volume),
      - el join final usuario x artículo con label 0/1.
 
    df_transactions DEBE ser solo train (mismo cuidado que en clustering_preprocess).
 
    Devuelve:
      X (DataFrame de features, con nombres de columna -> útil para
         feature_importances_), y (Serie de labels), dataset (DataFrame
         completo con customer_id/article_id/label + features, por si
         quieres inspeccionarlo), article_df (salida de
         compute_article_features, reutilizable para generar candidatos).
    """
    rng = np.random.default_rng(random_state)
 
    # --- Features de artículo (compartidas con clustering) ---
    article_df = compute_article_features(df_customers, df_products, df_transactions)
 
    available_cat = [c for c in CATEGORICAL_FEATURES_CLUSTER if c in article_df.columns]
    article_encoded = pd.get_dummies(article_df[available_cat], drop_first=False, dummy_na=True)
    article_features = pd.concat(
        [article_df[["article_id"] + NUMERIC_FEATURES_CLUSTER], article_encoded],
        axis=1,
    )
 
    # --- Features de usuario ---
    user_features = compute_user_features(df_customers, df_transactions)
    cat_user = [c for c in ["club_member_status"] if c in user_features.columns]
    if cat_user:
        user_features = pd.get_dummies(user_features, columns=cat_user, dummy_na=True)
 
    # --- Negative sampling ponderado por popularidad ---
    todos_los_articulos = article_features["article_id"].values
    popularidad = article_df.set_index("article_id").loc[todos_los_articulos, "sales_volume"].values
    popularidad = np.where(popularidad <= 0, 1, popularidad)  # evita prob=0 para todos
    prob_muestreo = popularidad / popularidad.sum()
 
    compras_por_cliente = df_transactions.groupby("customer_id")["article_id"].apply(set).to_dict()
 
    positivos = df_transactions[["customer_id", "article_id"]].drop_duplicates().copy()
    positivos["label"] = 1
 
    negativos_rows = []
    for cliente, grupo in positivos.groupby("customer_id"):
        comprados = compras_por_cliente.get(cliente, set())
        n_necesarios = len(grupo) * n_negativos_por_positivo
        intentos = 0
        n_generados = 0
        while n_generados < n_necesarios and intentos < n_necesarios * 5:
            candidatos = rng.choice(todos_los_articulos, size=min(50, n_necesarios), p=prob_muestreo)
            intentos += len(candidatos)
            for c in candidatos:
                if c not in comprados:
                    negativos_rows.append({"customer_id": cliente, "article_id": c, "label": 0})
                    n_generados += 1
                    if n_generados >= n_necesarios:
                        break
 
    negativos = pd.DataFrame(negativos_rows)
 
    dataset = pd.concat([positivos, negativos], ignore_index=True)
    dataset = dataset.merge(user_features, on="customer_id", how="left")
    dataset = dataset.merge(article_features, on="article_id", how="left")
 
    cols_no_feature = ["customer_id", "article_id", "label"]
    feature_cols = [c for c in dataset.columns if c not in cols_no_feature]
 
    X = dataset[feature_cols]
    y = dataset["label"]
 
    return X, y, dataset, article_features, user_features


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
    
    print("Customer profile: ", X_df.loc[bought_valid].mean(axis=0).to_frame().T)

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
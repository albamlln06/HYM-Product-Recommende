import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity


CATEGORICAL_FEATURES = [
    'product_group_name',
    'perceived_colour_master_name',
    'garment_group_name',
    'index_group_name',
    'graphical_appearance_name',
    'index_name',
    'section_name',
]

NUMERIC_FEATURES = [
    'avg_buyer_age',
    'avg_price',
    'sales_volume',
    'online_ratio',
    'recency_days',
]


def clustering_preprocess(df_customers, df_products, df_transactions):

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

    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    available_cat = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    encoded = pd.get_dummies(df[available_cat], drop_first=False)
    X = pd.concat([encoded, df[NUMERIC_FEATURES]], axis=1).astype(float)

    scaler = StandardScaler()
    X_final = scaler.fit_transform(X)

    article_ids = df['article_id'].values

    return X_final, article_ids, scaler, df


def find_optimal_k(X_final, k_range=range(2, 15)):
    inertias = []
    silhouettes = []

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


def recommend_by_cluster_similarity(
    customer_id,
    df_transactions,
    df_clusters_with_price,
    X_final,
    article_ids,
    top_n=10,
    rating_col='avg_buyer_age',
):
    id_to_idx = {aid: i for i, aid in enumerate(article_ids)}

    bought = df_transactions[df_transactions['customer_id'] == customer_id]['article_id'].unique()
    bought = [a for a in bought if a in id_to_idx]

    if len(bought) == 0:
        return pd.DataFrame()

    bought_idx = [id_to_idx[a] for a in bought]
    customer_profile = X_final[bought_idx].mean(axis=0).reshape(1, -1)

    clusters_bought = df_clusters_with_price[
        df_clusters_with_price['article_id'].isin(bought)
    ]['cluster'].unique()

    candidates = df_clusters_with_price[
        (df_clusters_with_price['cluster'].isin(clusters_bought)) &
        (~df_clusters_with_price['article_id'].isin(bought))
    ].copy()

    if candidates.empty:
        return pd.DataFrame()

    candidate_idx = [id_to_idx[a] for a in candidates['article_id'] if a in id_to_idx]
    candidates = candidates[candidates['article_id'].isin(article_ids[candidate_idx])]

    candidate_vectors = X_final[candidate_idx]
    similarities = cosine_similarity(customer_profile, candidate_vectors).flatten()
    candidates = candidates.copy()
    candidates['similarity'] = similarities

    recommendations = candidates.sort_values(
        by=['similarity', rating_col],
        ascending=[False, False],
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


# ==========================================
# PIPELINE COMPLETO DE RECOMENDACIÓN
# ==========================================
def predict_cluster(df_transactions, df_customers, df_products, customer_ids, K=8, top_n=12, explore_k=False):

    X_final, article_ids, _, df_products_enriched = clustering_preprocess(df_customers, df_products, df_transactions)

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
            X_final=X_final,
            article_ids=article_ids,
            top_n=top_n,
        )
        predictions.append(recs['article_id'].tolist() if not recs.empty else [])

    return predictions, df_merged, summary, kmeans_model
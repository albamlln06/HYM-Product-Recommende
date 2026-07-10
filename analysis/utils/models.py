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
            replace=False
        )

        intentos += len(candidatos)

        for c in candidatos:
            if c not in comprados:
                negativos.append({
                    "customer_id": cliente,
                    "article_id": c,
                    "label": 0,
                })

                n_generados += 1

                if n_generados >= n_necesarios:
                    break

    return negativos

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


def xgboost_preprocess(df_customers, df_products, df_transactions, n_negativos_por_positivo=4, random_state=42, feature_config=None, category_weights=None):
    rng = np.random.default_rng(random_state)

    # 1. Features de artículo y usuario
    article_df = compute_article_features(df_customers, df_products, df_transactions)
    user_df    = compute_user_features(df_customers, df_transactions, df_products)

    # 2. Muestras positivas: pares únicos (cliente, artículo) realmente comprados
    positivos = df_transactions[['customer_id', 'article_id']].drop_duplicates().copy()
    positivos['label'] = 1

    # 3. Negative sampling ponderado por popularidad (artículos más vendidos
    #    tienen más probabilidad de ser muestreados como negativos — más realista)
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

    # 3b. Peso de cada fila según la categoría del artículo (p.ej. bajar el
    #     peso de la ropa interior). None -> todas las filas pesan 1.0.
    dataset['sample_weight'] = compute_category_sample_weights(dataset, article_df, category_weights)

    # 4. Encoding de categóricas de usuario y artículo (compartido con el servido en vivo)
    user_encoded, article_encoded = encode_xgboost_categoricals(user_df, article_df, feature_config)

    # 5. Join final: dataset × features de usuario × features de artículo
    dataset = (
        dataset
        .merge(user_encoded,       on='customer_id', how='left')
        .merge(article_encoded, on='article_id',  how='left')
    )
    #imputamos los nulos
    dataset = imputar_nulos_tfm(dataset)
    #Convertimos los textos a categorías para ahorrar memoria RAM
    # 6. Separar X e y
    cols_no_feature = ['customer_id', 'article_id', 'label', 'sample_weight']
    feature_cols    = [c for c in dataset.columns if c not in cols_no_feature]

    X = dataset[feature_cols].fillna(0).astype(float)
    y = dataset['label']
    sample_weight = dataset['sample_weight']

    print(f"Positivos: {len(positivos):,}  |  Negativos: {len(negativos):,}")
    print(f"X shape: {X.shape}  |  Features: {len(feature_cols)}")

    return X, y, sample_weight, dataset, article_df, user_df


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


def recommend_xgboost_for_user(model, user_features, candidate_df, feature_cols, top_n=12):
    """
    Rankea un pool fijo de artículos candidatos para un usuario con un XGBClassifier ya entrenado.

    user_features : dict o Series con las features de UN usuario, ya codificadas
                    (salida de encode_xgboost_categoricals para ese customer_id).
    candidate_df  : DataFrame con columna 'article_id' + features de artículo codificadas
                    (el 'article_encoded' de encode_xgboost_categoricals, filtrado al pool
                    de candidatos, p.ej. los artículos más vendidos).
    feature_cols  : columnas exactas usadas en el entrenamiento (X.columns de xgboost_preprocess),
                    para alinear el orden/presencia de columnas en la inferencia.
    """
    n = len(candidate_df)
    user_block = pd.DataFrame([user_features] * n).reset_index(drop=True)
    article_block = candidate_df.reset_index(drop=True)

    combined = pd.concat([user_block, article_block], axis=1)
    X_infer = combined.reindex(columns=feature_cols, fill_value=0).fillna(0).astype(float)

    scores = model.predict_proba(X_infer)[:, 1]

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
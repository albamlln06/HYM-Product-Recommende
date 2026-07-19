"""
Entrena/exporta los artefactos que necesita el backend en vivo (app.py).

Reutiliza exactamente la misma lógica que los notebooks de análisis
(analysis/03_train_clustering.ipynb, analysis/03_train_xgboost.ipynb):
misma muestra de clientes activos, mismo split leave-one-out, mismos
hiperparámetros. Se ejecuta una sola vez (offline) y deja todo lo
necesario para servir recomendaciones y métricas en models/.

Cada modelo se entrena en su propia función (entrenar_modelo_xxx) y
queda registrado en MLflow: hiperparámetros, métrica MAP@12 y el
modelo entrenado. Así se pueden comparar runs entre sí en la UI de
MLflow (`mlflow ui`) antes de decidir qué hiperparámetros usar.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import time
import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBRanker
import warnings

# Ocultar específicamente los FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
sys.path.insert(0, str(BASE_DIR / "analysis"))

from utils import preprocess, models

N_CUSTOMERS = 600
MIN_PURCHASES = 6
MAX_MONTHS_SINCE_LAST_PURCHASE = 12
#Productos que se van a recomendar (nº)
K_EVAL = 12
# Nº de compras más recientes de cada cliente que se dejan como test (resto
# a train). 1 = leave-one-out clásico; con más margen (p.ej. 2) cada usuario
# tiene más de un acierto posible y el MAP@12 varía menos entre ejecuciones.
N_TEST_PURCHASES = 1
#Semilla
RANDOM_STATE = 42

# --- Hiperparámetros del modelo de clustering ---
K_CLUSTERS = 10

# --- Hiperparámetros del modelo XGBoost ---
CANDIDATE_POOL_SIZE = 50000  # nº de artículos usados para el entrenamiento
N_NEGATIVOS_POR_POSITIVO = 8
XGB_N_ESTIMATORS = 500
XGB_MAX_DEPTH = 4
XGB_LEARNING_RATE = 0.06

# --- Hiperparámetros del modelo BPR (feature bpr_score de XGBoost) ---
# Barrido rápido (BPR evaluado solo, sin XGBoost) con N_CUSTOMERS=600: subir
# iterations y regularization ayuda algo (MAP@12 standalone 0.0002->0.0010),
# pero sigue muy por debajo de Popular (0.0037) a este tamaño de muestra —
# hay pocos datos de interacción para que BPR aprenda señal real.
BPR_FACTORS = 32
BPR_ITERATIONS = 50
BPR_REGULARIZATION = 0.05

# --- MLflow ---
MLFLOW_EXPERIMENT_NAME = "hym-recomendator2"
MLFLOW_TRACKING_URI = f"sqlite:///{Path(__file__).resolve().parent / 'mlflow.db'}"


def leave_one_out_split(df_transactions, n_test=1):
    """
    Deja como test las n_test compras más recientes de cada cliente (por
    fecha) y el resto como train. n_test=1 es el leave-one-out clásico: cada
    usuario "acierta o falla" contra un único artículo, lo que da un MAP@12
    con mucha varianza. Subir n_test da más margen (más de un acierto
    posible por usuario) a costa de dejar menos historial en train.
    """
    df_sorted = df_transactions.sort_values("t_dat", ascending=False, kind="mergesort")
    rank_reciente = df_sorted.groupby("customer_id").cumcount()
    df_test = df_sorted[rank_reciente < n_test]
    df_train = df_sorted[rank_reciente >= n_test]

    users_with_train = set(df_train["customer_id"].unique())
    eval_users = [u for u in df_test["customer_id"].unique() if u in users_with_train]

    ground_truth = (
        df_test[df_test["customer_id"].isin(eval_users)]
        .groupby("customer_id")["article_id"]
        .apply(list)
        .to_dict()
    )
    actual = [ground_truth[u] for u in eval_users]
    return df_train, df_test, eval_users, actual


def build_category_ground_truth(df_products, df_train, eval_users, actual, category_col="product_group_name"):
    """
    Construye la info de categoría que necesita registrar_metricas_categoria
    para evaluar los 4 modelos: la categoría del artículo que cada usuario
    realmente compró en el hold-out (test) y el conjunto de categorías que ha
    comprado en general (histórico de train). Usa product_group_name, la
    misma columna que ya se usa como "categoría" en el modelo de clustering.
    """
    article_to_category = df_products.set_index("article_id")[category_col].to_dict()

    actual_categories_test = [
        {article_to_category[a] for a in articulos if a in article_to_category}
        for articulos in actual
    ]

    categorias_por_usuario = (
        df_train.assign(category=df_train["article_id"].map(article_to_category))
        .dropna(subset=["category"])
        .groupby("customer_id")["category"]
        .apply(set)
        .to_dict()
    )
    categorias_compradas_general = [categorias_por_usuario.get(u, set()) for u in eval_users]

    return article_to_category, actual_categories_test, categorias_compradas_general


def registrar_metricas_categoria(
    predicciones, article_to_category, actual_categories_test, categorias_compradas_general, k_eval,
):
    """
    Calcula el % de aciertos por categoría de un modelo y lo registra en el
    run de MLflow activo: category_hit_rate_test (contra la categoría del
    artículo que el usuario realmente compró en el hold-out) y
    category_hit_rate_general (contra todas las categorías que ha comprado
    alguna vez en train). Complementa a MAP@12: un modelo puede fallar el SKU
    exacto pero seguir recomendando dentro de la categoría correcta.
    """
    cat_hit_test = models.mean_category_hit_rate(
        actual_categories_test, predicciones, article_to_category, k=k_eval
    )
    cat_hit_general = models.mean_category_hit_rate(
        categorias_compradas_general, predicciones, article_to_category, k=k_eval
    )
    mlflow.log_metric("category_hit_rate_test", cat_hit_test)
    mlflow.log_metric("category_hit_rate_general", cat_hit_general)
    return cat_hit_test, cat_hit_general


def registrar_metricas_aciertos(actual, predicciones, k_eval):
    """
    Aciertos EXACTOS de artículo (no por categoría): cuántos SKU concretos de
    actual aparecen en las k primeras recomendaciones. Se registra en el run
    de MLflow activo. total_hits es la cuenta bruta (útil para ver "cuántos
    aciertos ha tenido" en números absolutos, no solo el MAP@12 normalizado);
    hit_rate es el % de usuarios con al menos 1 acierto exacto.
    """
    total = models.total_hits(actual, predicciones, k=k_eval)
    rate = models.hit_rate(actual, predicciones, k=k_eval)
    mlflow.log_metric("total_hits", total)
    mlflow.log_metric("hit_rate", rate)
    return total, rate


def entrenar_modelo_random(
    df_products, eval_users, actual, article_to_category, actual_categories_test, categorias_compradas_general,
    k_eval=12, seed=42, run_name="baseline_random", extra_params=None,
):
    """
    Baseline: recomienda artículos al azar sobre el catálogo completo.
    Sirve para tener un suelo de referencia comparable al resto de modelos,
    que también buscan sobre todo el catálogo (no solo lo vendido en la muestra de train).
    """
    with mlflow.start_run(run_name=run_name):
        params = {"k_eval": k_eval, "seed": seed}
        if extra_params:
            params.update(extra_params)
        mlflow.log_params(params)

        predicciones = models.predict_random(df_products, eval_users, k=k_eval, seed=seed)
        map12 = models.mapk(actual, predicciones, k=k_eval)
        mlflow.log_metric("map12", map12)

        cat_hit_test, cat_hit_general = registrar_metricas_categoria(
            predicciones, article_to_category, actual_categories_test, categorias_compradas_general, k_eval,
        )
        total_hits, hit_rate = registrar_metricas_aciertos(actual, predicciones, k_eval)

        print(
            f"[Random]  MAP@12 = {map12:.4f}  |  CatTest = {cat_hit_test:.4f}  |  CatGeneral = {cat_hit_general:.4f}  |  "
            f"Aciertos = {total_hits}  |  HitRate = {hit_rate:.4f}"
        )

    return predicciones, map12, cat_hit_test, cat_hit_general, total_hits, hit_rate


def entrenar_modelo_popular(
    df_train, eval_users, actual, article_to_category, actual_categories_test, categorias_compradas_general,
    k_eval=12, run_name="baseline_popular", extra_params=None,
):
    """Baseline: recomienda a todo el mundo los artículos más vendidos."""
    with mlflow.start_run(run_name=run_name):
        params = {"k_eval": k_eval}
        if extra_params:
            params.update(extra_params)
        mlflow.log_params(params)

        predicciones = models.predict_popular(df_train, eval_users, k=k_eval)
        map12 = models.mapk(actual, predicciones, k=k_eval)
        mlflow.log_metric("map12", map12)

        cat_hit_test, cat_hit_general = registrar_metricas_categoria(
            predicciones, article_to_category, actual_categories_test, categorias_compradas_general, k_eval,
        )
        total_hits, hit_rate = registrar_metricas_aciertos(actual, predicciones, k_eval)

        print(
            f"[Popular] MAP@12 = {map12:.4f}  |  CatTest = {cat_hit_test:.4f}  |  CatGeneral = {cat_hit_general:.4f}  |  "
            f"Aciertos = {total_hits}  |  HitRate = {hit_rate:.4f}"
        )

    return predicciones, map12, cat_hit_test, cat_hit_general, total_hits, hit_rate


def entrenar_modelo_cluster(
    df_customers, df_products, df_train, eval_users, actual,
    article_to_category, actual_categories_test, categorias_compradas_general,
    k_clusters=8, k_eval=12, random_state=42,
    run_name="cluster_kmeans", extra_params=None,
):
    """Entrena el modelo de clustering (KMeans + similitud coseno) y lo registra en MLflow."""
    with mlflow.start_run(run_name=run_name):
        params = {"k_clusters": k_clusters, "random_state": random_state}
        if extra_params:
            params.update(extra_params)
        mlflow.log_params(params)

        X_final, article_ids, scaler, df_products_enriched = models.clustering_preprocess(
            df_customers, df_products, df_train
        )
        df_clusters, kmeans_model = models.fit_product_clustering(X_final, k_clusters, article_ids)
        df_merged, cluster_summary = models.inspect_clusters(
            df_products=df_products_enriched, df_clusters=df_clusters, category_col="product_group_name"
        )
        X_df = pd.DataFrame(X_final, index=article_ids)

        predicciones = []
        for u in eval_users:
            recs = models.recommend_by_cluster_similarity(u, df_train, df_merged, X_df, top_n=k_eval)
            predicciones.append(recs["article_id"].tolist() if not recs.empty else [])

        map12 = models.mapk(actual, predicciones, k=k_eval)
        mlflow.log_metric("map12", map12)
        mlflow.sklearn.log_model(kmeans_model, name="kmeans_model")

        cat_hit_test, cat_hit_general = registrar_metricas_categoria(
            predicciones, article_to_category, actual_categories_test, categorias_compradas_general, k_eval,
        )
        total_hits, hit_rate = registrar_metricas_aciertos(actual, predicciones, k_eval)

        print(
            f"[Cluster] MAP@12 = {map12:.4f}  |  CatTest = {cat_hit_test:.4f}  |  CatGeneral = {cat_hit_general:.4f}  |  "
            f"Aciertos = {total_hits}  |  HitRate = {hit_rate:.4f}"
        )

    resultado = {
        "kmeans_model": kmeans_model,
        "scaler": scaler,
        "X_final": X_final,
        "article_ids": article_ids,
        "df_merged": df_merged,
        "predicciones": predicciones,
        "map12": map12,
        "category_hit_rate_test": cat_hit_test,
        "category_hit_rate_general": cat_hit_general,
        "total_hits": total_hits,
        "hit_rate": hit_rate,
    }
    return resultado


def entrenar_modelo_xgboost(
    df_customers, df_products, df_train, eval_users, actual,
    article_to_category, actual_categories_test, categorias_compradas_general,
    n_estimators=300, max_depth=6, learning_rate=0.05,
    n_negativos_por_positivo=8, candidate_pool_size=120000,
    bpr_factors=32, bpr_iterations=15, bpr_regularization=0.01,
    k_eval=12, random_state=42,
    run_name="xgboost", extra_params=None,
):
    """
    Entrena el modelo de ranking XGBoost y lo registra en MLflow.

    run_name    : nombre del run en MLflow. Al hacer pruebas conviene poner
                  algo descriptivo (p.ej. "xgb_n300_d6_lr0.05") para
                  distinguir cada combinación de un vistazo en la UI.
    extra_params: dict opcional con parámetros que NO afectan a esta función
                  pero que quieres dejar registrados en MLflow para saber
                  con qué datos se entrenó (p.ej. nº de clientes usados).
    bpr_factors, bpr_iterations, bpr_regularization: hiperparámetros del
                  modelo BPR (matrix factorization colaborativa, ver
                  models.train_bpr_model) que aporta la feature bpr_score a
                  XGBoost.
    """
    with mlflow.start_run(run_name=run_name):
        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "n_negativos_por_positivo": n_negativos_por_positivo,
            "candidate_pool_size": candidate_pool_size,
            "random_state": random_state,
            # Antes de esto era one-hot (pd.get_dummies, X con ~230 columnas);
            # ahora son categóricas nativas de XGBoost (X con ~20 columnas).
            # Runs sin este param en MLflow son de la versión one-hot y no son
            # directamente comparables en tiempo de entrenamiento/inferencia.
            "xgboost_categorical_encoding": "native",
            # Runs sin esto no tenían la feature bpr_score (señal colaborativa
            # vía matrix factorization BPR), tampoco directamente comparables.
            "bpr_factors": bpr_factors,
            "bpr_iterations": bpr_iterations,
            "bpr_regularization": bpr_regularization,
        }
        if extra_params:
            params.update(extra_params)
        mlflow.log_params(params)
        t0_xgb_preprocess = time.time()
        print('Inicciando preprocessing XGBoost...')
        X, y, dataset, article_df, user_df, user_factors_df, item_factors_df = models.xgboost_preprocess(
            df_customers, df_products, df_train,
            n_negativos_por_positivo=n_negativos_por_positivo, random_state=random_state,
            bpr_factors=bpr_factors, bpr_iterations=bpr_iterations, bpr_regularization=bpr_regularization,
        )
        feature_cols = list(X.columns)
        
        # XGBRanker necesita las filas agrupadas (contiguas) por query group;
        # aquí el grupo es el cliente, cuyas filas positivas/negativas se rankean entre sí.
        order = dataset["customer_id"].sort_values(kind="stable").index
        X_sorted = X.loc[order].reset_index(drop=True)
        y_sorted = y.loc[order].reset_index(drop=True)
        # qid debe ser numérico: codificamos customer_id (string) a enteros correlativos.
        qid_codes, _ = pd.factorize(dataset.loc[order, "customer_id"], sort=False)
        qid_sorted = pd.Series(qid_codes)
        print(f"Preprocesado XGBoost: {time.time() - t0_xgb_preprocess:.2f}s")
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
        train_idx, val_idx = next(gss.split(X_sorted, y_sorted, groups=qid_sorted))
        train_idx, val_idx = np.sort(train_idx), np.sort(val_idx)
        t0_xgb_train = time.time()
        X_train, y_train, qid_train = X_sorted.iloc[train_idx], y_sorted.iloc[train_idx], qid_sorted.iloc[train_idx]
        X_val, y_val, qid_val = X_sorted.iloc[val_idx], y_sorted.iloc[val_idx], qid_sorted.iloc[val_idx]

        xgb_model = XGBRanker(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective="rank:map",
            eval_metric="map",
            random_state=random_state,
            enable_categorical=True,
            tree_method="hist",
        )
        xgb_model.fit(
            X_train, y_train, qid=qid_train,
            eval_set=[(X_val, y_val)], eval_qid=[qid_val],
            verbose=False,
        )
        print(f"Entrenamiento XGBoost: {time.time() - t0_xgb_train:.2f}s")
        t0_infer = time.time()
        user_encoded, article_encoded = models.encode_xgboost_categoricals(user_df, article_df)
        candidate_pool = article_encoded.sort_values("sales_volume", ascending=False).head(candidate_pool_size)
        user_encoded_indexed = user_encoded.set_index("customer_id")

        # Si un cliente solo ha comprado artículos de un único género
        # (index_group_name: Ladieswear/Menswear/Baby-Children/Divided/Sport),
        # se le acotan las recomendaciones a ese género. Si ha comprado de
        # varios (o no hay datos), no se filtra y se usa el pool completo.
        article_to_gender = df_products.set_index("article_id")["index_group_name"].to_dict()
        generos_por_usuario = (
            df_train.assign(genero=df_train["article_id"].map(article_to_gender))
            .dropna(subset=["genero"])
            .groupby("customer_id")["genero"]
            .apply(set)
            .to_dict()
        )
        candidate_genero = candidate_pool["article_id"].map(article_to_gender)
        print('Iniciando de predicciones')
        predicciones = []
        for u in eval_users:
            if u not in user_encoded_indexed.index:
                predicciones.append([])
                continue

            candidate_pool_u = candidate_pool
            generos_usuario = generos_por_usuario.get(u, set())
            if len(generos_usuario) == 1:
                genero_usuario = next(iter(generos_usuario))
                pool_filtrado = candidate_pool[candidate_genero == genero_usuario]
                if not pool_filtrado.empty:
                    candidate_pool_u = pool_filtrado

            # bpr_score es una feature CRUZADA (depende del usuario Y del
            # artículo), así que no se puede precalcular una vez en
            # candidate_pool como el resto de columnas: se recalcula para
            # cada usuario contra su propio candidate_pool_u (vectorizado).
            user_vector = user_factors_df.reindex([u]).fillna(0.0).to_numpy()[0]
            bpr_scores = models.bpr_dot_scores(candidate_pool_u["article_id"], item_factors_df, user_vector)
            candidate_pool_u = candidate_pool_u.assign(bpr_score=bpr_scores)
            user_row = user_encoded_indexed.loc[[u]]
            recs = models.recommend_xgboost_for_user(xgb_model, user_row, candidate_pool_u, feature_cols, top_n=k_eval)
            predicciones.append(recs["article_id"].tolist())

        map12 = models.mapk(actual, predicciones, k=k_eval)
        mlflow.log_metric("map12", map12)
        mlflow.xgboost.log_model(xgb_model, name="xgboost_model")

        cat_hit_test, cat_hit_general = registrar_metricas_categoria(
            predicciones, article_to_category, actual_categories_test, categorias_compradas_general, k_eval,
        )
        total_hits, hit_rate = registrar_metricas_aciertos(actual, predicciones, k_eval)

        print(
            f"[XGBoost] {run_name} -> MAP@12 = {map12:.4f}  |  CatTest = {cat_hit_test:.4f}  |  "
            f"CatGeneral = {cat_hit_general:.4f}  |  Aciertos = {total_hits}  |  HitRate = {hit_rate:.4f}  |  params={params}"
        )
        print(f"Inferencia XGBoost: {time.time() - t0_infer:.2f}s")
    resultado = {
        "xgb_model": xgb_model,
        "feature_cols": feature_cols,
        "user_df": user_df,
        "user_encoded": user_encoded,
        "candidate_pool": candidate_pool,
        "user_factors_df": user_factors_df,
        "item_factors_df": item_factors_df,
        "predicciones": predicciones,
        "map12": map12,
        "category_hit_rate_test": cat_hit_test,
        "category_hit_rate_general": cat_hit_general,
        "total_hits": total_hits,
        "hit_rate": hit_rate,
    }
    return resultado


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    t0 = time.time()
    print(f"Cargando muestra de {N_CUSTOMERS} clientes...")
    df_customers, df_products, df_transactions = preprocess.load_complete_dataset_filtered_number_customers(
        N_CUSTOMERS, random_state=RANDOM_STATE
    )
    df_transactions = preprocess.filter_customers_by_activity(
        df_transactions, min_purchases=MIN_PURCHASES, max_months_since_last_purchase=MAX_MONTHS_SINCE_LAST_PURCHASE
    )
    print(f"Transacciones tras filtro de actividad: {len(df_transactions):,}")

    df_train, df_test, eval_users, actual = leave_one_out_split(df_transactions, n_test=N_TEST_PURCHASES)
    print(f"Usuarios de evaluación: {len(eval_users):,}  |  train: {len(df_train):,}  |  test: {len(df_test):,}")

    # Info de categoría (product_group_name) para medir, además de MAP@12, si
    # cada modelo acierta al menos la categoría del producto (más laxo que el
    # SKU exacto): contra lo que el usuario compró en el hold-out y contra
    # todo lo que ha comprado alguna vez en train.
    article_to_category, actual_categories_test, categorias_compradas_general = build_category_ground_truth(
        df_products, df_train, eval_users, actual,
    )

    # --- Entrenamiento de cada modelo (cada uno queda registrado en MLflow) ---
    # data_config identifica la muestra (nº de clientes + filtro de actividad +
    # nº de compras dejadas como test) para poder comparar los 4 modelos entre
    # sí en igualdad de condiciones (ver experiment_xgboost.py, que reentrena
    # estos mismos baselines por cada combinación de datos que prueba).
    # n_test_purchases va aparte también como su propio param: cambia qué mide
    # el split (leave-1-out clásico vs. más margen), así que un run con
    # n_test_purchases=1 y otro con =2 NO son comparables en MAP@12/CatTest
    # aunque compartan el resto de la config de datos.
    extra_params_datos = {
        "n_customers": N_CUSTOMERS,
        "min_purchases": MIN_PURCHASES,
        "max_months_since_last_purchase": MAX_MONTHS_SINCE_LAST_PURCHASE,
        "n_test_purchases": N_TEST_PURCHASES,
        "data_config": (
            f"cust{N_CUSTOMERS}_minp{MIN_PURCHASES}_maxm{MAX_MONTHS_SINCE_LAST_PURCHASE}"
            f"_ntest{N_TEST_PURCHASES}"
        ),
    }

    _, map_random, cat_test_random, cat_general_random, hits_random, hit_rate_random = entrenar_modelo_random(
        df_products, eval_users, actual, article_to_category, actual_categories_test, categorias_compradas_general,
        k_eval=K_EVAL, seed=RANDOM_STATE, extra_params=extra_params_datos,
    )
    _, map_popular, cat_test_popular, cat_general_popular, hits_popular, hit_rate_popular = entrenar_modelo_popular(
        df_train, eval_users, actual, article_to_category, actual_categories_test, categorias_compradas_general,
        k_eval=K_EVAL, extra_params=extra_params_datos,
    )
    to_cluster = time.time()
    print("Entrenando modelo de clustering...")
    resultado_cluster = entrenar_modelo_cluster(
        df_customers, df_products, df_train, eval_users, actual,
        article_to_category, actual_categories_test, categorias_compradas_general,
        k_clusters=K_CLUSTERS, k_eval=K_EVAL, random_state=RANDOM_STATE,
        extra_params=extra_params_datos,
    )
    print(f"Entrenamiento de clustering: {to_cluster - t0:.2f}s")
    print("Entrenando modelo XGBoost...")
    resultado_xgboost = entrenar_modelo_xgboost(
        df_customers, df_products, df_train, eval_users, actual,
        article_to_category, actual_categories_test, categorias_compradas_general,
        n_estimators=XGB_N_ESTIMATORS, max_depth=XGB_MAX_DEPTH, learning_rate=XGB_LEARNING_RATE,
        n_negativos_por_positivo=N_NEGATIVOS_POR_POSITIVO, candidate_pool_size=CANDIDATE_POOL_SIZE,
        bpr_factors=BPR_FACTORS, bpr_iterations=BPR_ITERATIONS, bpr_regularization=BPR_REGULARIZATION,
        k_eval=K_EVAL, random_state=RANDOM_STATE, extra_params=extra_params_datos,
    )

    # --- Métricas ---
    raw_metrics = {
        "Random": {
            "map12": map_random,
            "category_hit_rate_test": cat_test_random,
            "category_hit_rate_general": cat_general_random,
            "total_hits": hits_random,
            "hit_rate": hit_rate_random,
        },
        "Popular": {
            "map12": map_popular,
            "category_hit_rate_test": cat_test_popular,
            "category_hit_rate_general": cat_general_popular,
            "total_hits": hits_popular,
            "hit_rate": hit_rate_popular,
        },
        "Cluster": {
            "map12": resultado_cluster["map12"],
            "category_hit_rate_test": resultado_cluster["category_hit_rate_test"],
            "category_hit_rate_general": resultado_cluster["category_hit_rate_general"],
            "total_hits": resultado_cluster["total_hits"],
            "hit_rate": resultado_cluster["hit_rate"],
        },
        "XGBoost": {
            "map12": resultado_xgboost["map12"],
            "category_hit_rate_test": resultado_xgboost["category_hit_rate_test"],
            "category_hit_rate_general": resultado_xgboost["category_hit_rate_general"],
            "total_hits": resultado_xgboost["total_hits"],
            "hit_rate": resultado_xgboost["hit_rate"],
        },
    }
    print(f"MAP@12 / Acierto categoría / Aciertos exactos  (n_eval_users={len(eval_users)}):")
    for name, m in sorted(raw_metrics.items(), key=lambda x: -x[1]["map12"]):
        print(
            f"  {name:<10} MAP@12={m['map12']:.4f}  "
            f"CatTest={m['category_hit_rate_test']:.4f}  CatGeneral={m['category_hit_rate_general']:.4f}  "
            f"Aciertos={m['total_hits']}  HitRate={m['hit_rate']:.4f}"
        )

    # --- Guardado de artefactos para el backend en vivo ---
    MODELS_DIR.mkdir(exist_ok=True)

    metrics_payload = {
        "metrics": [
            {"model": name, **m}
            for name, m in sorted(raw_metrics.items(), key=lambda x: -x[1]["map12"])
        ],
        "meta": {
            "n_customers_sampled": N_CUSTOMERS,
            "n_eval_users": len(eval_users),
            "k": K_EVAL,
            "k_clusters": K_CLUSTERS,
            "trained_at": datetime.now(timezone.utc).isoformat(),
        },
    }
    (MODELS_DIR / "metrics.json").write_text(json.dumps(metrics_payload, indent=2))

    joblib.dump(resultado_cluster["kmeans_model"], MODELS_DIR / "kmeans.joblib")
    joblib.dump(resultado_cluster["scaler"], MODELS_DIR / "scaler.joblib")
    joblib.dump(resultado_xgboost["xgb_model"], MODELS_DIR / "xgboost_model.joblib")
    (MODELS_DIR / "xgboost_feature_cols.json").write_text(json.dumps(resultado_xgboost["feature_cols"]))

    np.save(MODELS_DIR / "cluster_X_final.npy", resultado_cluster["X_final"])
    np.save(MODELS_DIR / "cluster_article_ids.npy", resultado_cluster["article_ids"])

    resultado_cluster["df_merged"].to_parquet(MODELS_DIR / "article_features.parquet", index=False)
    resultado_xgboost["candidate_pool"].to_parquet(MODELS_DIR / "candidate_articles.parquet", index=False)
    resultado_xgboost["user_df"].to_parquet(MODELS_DIR / "customers.parquet", index=False)
    resultado_xgboost["user_encoded"].to_parquet(MODELS_DIR / "customers_xgb_features.parquet", index=False)
    df_train[["customer_id", "article_id", "t_dat", "price", "is_online"]].to_parquet(
        MODELS_DIR / "train_transactions.parquet", index=False
    )
    # Factores BPR (señal colaborativa para la feature bpr_score de XGBoost),
    # para que el servido en vivo pueda calcular el mismo bpr_score por cliente.
    resultado_xgboost["user_factors_df"].rename_axis("customer_id").reset_index().to_parquet(
        MODELS_DIR / "bpr_user_factors.parquet", index=False
    )
    resultado_xgboost["item_factors_df"].rename_axis("article_id").reset_index().to_parquet(
        MODELS_DIR / "bpr_item_factors.parquet", index=False
    )

    print(f"\nArtefactos guardados en {MODELS_DIR}")


if __name__ == "__main__":
    main()

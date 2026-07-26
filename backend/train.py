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
TRAINING_CACHE_DIR = MODELS_DIR / "training_cache"
sys.path.insert(0, str(BASE_DIR / "analysis"))

from utils import preprocess, models

N_CUSTOMERS = 1000
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
# Valores del mejor trial encontrado por optuna_search.py (estudio
# "xgboost_hpo", trial 179 de 180, MAP@12=0.024188 sobre N_CUSTOMERS=1000 —
# misma muestra que usa este script, así que es comparable). candidate_pool_size,
# BPR y candidate_top_k_* NO se tocan: en la búsqueda se dejaron fijos, no los
# exploró Optuna.
CANDIDATE_POOL_SIZE = 50000  # nº de artículos usados para el entrenamiento
N_NEGATIVOS_POR_POSITIVO = 10
XGB_N_ESTIMATORS = 550
XGB_MAX_DEPTH = 7
XGB_LEARNING_RATE = 0.09057400301148144
XGB_REG_LAMBDA = 0.0068033325275598956
XGB_REG_ALPHA = 0.0017096436589603633
XGB_SUBSAMPLE = 0.7474002782840928
XGB_COLSAMPLE_BYTREE = 0.6006997087236939
XGB_MIN_CHILD_WEIGHT = 8
XGB_GAMMA = 0.7588223352482466

# --- Hiperparámetros del modelo BPR (feature bpr_score de XGBoost) ---
# Barrido rápido (BPR evaluado solo, sin XGBoost) con N_CUSTOMERS=600: subir
# iterations y regularization ayuda algo (MAP@12 standalone 0.0002->0.0010),
# pero sigue muy por debajo de Popular (0.0037) a este tamaño de muestra —
# hay pocos datos de interacción para que BPR aprenda señal real.
BPR_FACTORS = 32
BPR_ITERATIONS = 50
BPR_REGULARIZATION = 0.05

# --- Interruptores modulares del pipeline ---
USE_CACHED_CLUSTERING = True
USE_HYBRID_NEGATIVES = True
# Puesto a False temporalmente: los hiperparámetros XGB_* actuales vienen del
# mejor trial de optuna_search.py, que tuneó SIN candidatos híbridos
# (use_hybrid_candidates nunca se pasaba ahí, así que caía en su default
# False). Evaluarlos aquí con True mezclaba dos regímenes de candidate pool
# distintos y el MAP@12 se desplomó (0.0131 -> 0.0041). Este run sirve para
# confirmar que sí funcionan bien en el régimen para el que se tunearon.
USE_HYBRID_CANDIDATES = False

NEGATIVE_PROPORTIONS = {
    "popular": 0.38881769647337966,
    "cluster": 0.2867407826021783,
    "bpr": 0.15499201744478697,
    "cov": 0.1694495034796551,
}
BPR_NEIGHBORS_TOP_K = 50

TOP_K_CANDIDATES_BPR = 100
TOP_K_CANDIDATES_CLUSTER = 30
TOP_K_CANDIDATES_COV = 20
TOP_K_CANDIDATES_POPULAR = 20

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

def calcular_recall_candidatos(df_inferencia, actual_dict):
    """
    Calcula el Recall del pool de candidatos: 
    ¿Qué porcentaje de los artículos comprados realmente estaban en el pool antes de pasar por XGBoost?
    
    - df_inferencia: DataFrame que contiene al menos 'customer_id' y 'article_id' de los candidatos.
    - actual_dict: Diccionario {customer_id: [lista_de_articulos_comprados_en_test]}
    """
    hits_candidatos = 0
    total_comprados_test = 0

    # Agrupamos los candidatos por cliente en Sets para que la búsqueda sea instantánea
    candidatos_por_cliente = df_inferencia.groupby('customer_id')['article_id'].apply(set).to_dict()

    for customer, comprados_reales in actual_dict.items():
        comprados_set = set(comprados_reales)
        if not comprados_set:
            continue

        candidatos_cliente = candidatos_por_cliente.get(customer, set())

        # Intersección: ¿cuántos de los comprados ESTABAN en la lista de candidatos?
        hits_candidatos += len(comprados_set.intersection(candidatos_cliente))
        total_comprados_test += len(comprados_set)

    # Calculamos el porcentaje (Recall)
    recall_candidatos = hits_candidatos / total_comprados_test if total_comprados_test > 0 else 0.0
    
    return recall_candidatos, hits_candidatos, total_comprados_test

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
    use_cache=True, cache_dir=None,
):
    """Entrena el modelo de clustering (KMeans + similitud coseno) y lo registra en MLflow."""
    with mlflow.start_run(run_name=run_name):
        params = {"k_clusters": k_clusters, "random_state": random_state, "use_cache": use_cache}
        if extra_params:
            params.update(extra_params)
        mlflow.log_params(params)

        if use_cache:
            cluster_artifacts = models.construir_o_cargar_clustering(
                df_customers, df_products, df_train,
                k_clusters=k_clusters,
                cache_dir=str(cache_dir or TRAINING_CACHE_DIR),
            )
            X_final = cluster_artifacts["X_final"]
            article_ids = cluster_artifacts["article_ids"]
            scaler = cluster_artifacts["scaler"]
            kmeans_model = cluster_artifacts["kmeans_model"]
            df_merged = cluster_artifacts["df_merged"]
        else:
            X_final, article_ids, scaler, df_products_enriched = models.clustering_preprocess(
                df_customers, df_products, df_train
            )
            df_clusters, kmeans_model = models.fit_product_clustering(X_final, k_clusters, article_ids)
            df_merged, _cluster_summary = models.inspect_clusters(
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
    reg_lambda=1.0, reg_alpha=0.0, subsample=1.0, colsample_bytree=1.0,
    min_child_weight=1, gamma=0.0, early_stopping_rounds=30,
    n_negativos_por_positivo=8, candidate_pool_size=120000,
    bpr_factors=32, bpr_iterations=15, bpr_regularization=0.01,
    k_eval=12, random_state=42,
    run_name="xgboost", extra_params=None,
    historial_dict=None, cov_dict=None,
    inference_batch_size=100,
    use_hybrid_negatives=False, negative_proportions=None,
    article_to_cluster=None, cluster_to_articles_sorted=None,
    bpr_neighbors_top_k=50, cache_dir=None,
    use_hybrid_candidates=False,
    candidate_top_k_bpr=100, candidate_top_k_cluster=30,
    candidate_top_k_cov=20, candidate_top_k_pop=20,
    df_articles_for_candidates=None,
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
    reg_lambda, reg_alpha, subsample, colsample_bytree, min_child_weight,
    gamma: hiperparámetros de regularización/muestreo estándar de XGBoost,
                  expuestos aquí (con sus valores por defecto de XGBoost) para
                  que optuna_search.py los pueda explorar sin tocar esta
                  función.
    early_stopping_rounds: corta el entrenamiento si el MAP de validación no
                  mejora en N rondas, en vez de gastar las n_estimators
                  completas siempre. Acelera tanto los runs normales como
                  cada trial de Optuna.
    inference_batch_size: nº de usuarios de evaluación que se agrupan en cada
                  llamada a models.recommend_xgboost_batch. Antes se llamaba
                  a model.predict una vez POR USUARIO (coste fijo de esa
                  llamada multiplicado por nº de usuarios); agrupando en
                  lotes se paga ese coste fijo una vez por lote en vez de una
                  vez por usuario. Subirlo acelera la inferencia pero sube el
                  pico de memoria (candidate_pool_size × inference_batch_size
                  filas en memoria a la vez); bájalo si el proceso se queda
                  sin memoria.
    """
    with mlflow.start_run(run_name=run_name):
        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "reg_lambda": reg_lambda,
            "reg_alpha": reg_alpha,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "min_child_weight": min_child_weight,
            "gamma": gamma,
            "early_stopping_rounds": early_stopping_rounds,
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
            "use_hybrid_negatives": use_hybrid_negatives,
            "negative_proportions": json.dumps(negative_proportions or {}, sort_keys=True),
            "bpr_neighbors_top_k": bpr_neighbors_top_k,
            "use_hybrid_candidates": use_hybrid_candidates,
            "candidate_top_k_bpr": candidate_top_k_bpr,
            "candidate_top_k_cluster": candidate_top_k_cluster,
            "candidate_top_k_cov": candidate_top_k_cov,
            "candidate_top_k_pop": candidate_top_k_pop,
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
            historial_dict=historial_dict, cov_dict=cov_dict,
            use_hybrid_negatives=use_hybrid_negatives,
            negative_proportions=negative_proportions,
            article_to_cluster=article_to_cluster,
            cluster_to_articles_sorted=cluster_to_articles_sorted,
            bpr_neighbors_top_k=bpr_neighbors_top_k,
            cache_dir=str(cache_dir or TRAINING_CACHE_DIR),
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
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            gamma=gamma,
            objective="rank:map",
            eval_metric="map",
            random_state=random_state,
            enable_categorical=True,
            tree_method="hist",
            early_stopping_rounds=early_stopping_rounds,
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
        candidate_pairs = None
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
        if use_hybrid_candidates:
            df_articles_candidates = (
                df_articles_for_candidates
                if df_articles_for_candidates is not None
                else article_df
            )
            compras_por_cliente = (
                df_train.groupby("customer_id")["article_id"].apply(set).to_dict()
            )
            candidate_pairs = models.generar_candidatos_hibridos(
                eval_users,
                user_factors_df,
                item_factors_df,
                compras_por_cliente,
                df_articles_candidates,
                df_train,
                historial_dict,
                cov_dict,
                top_k_bpr=candidate_top_k_bpr,
                top_k_cluster=candidate_top_k_cluster,
                top_k_cov=candidate_top_k_cov,
                top_k_pop=candidate_top_k_pop,
            )
            candidate_pool = article_encoded
            candidate_genero = candidate_pool["article_id"].map(article_to_gender)

        actual_dict = dict(zip(eval_users, actual))
        recall_cand, hits_cand, total_reales = calcular_recall_candidatos(candidate_pairs, actual_dict)
    
        # Imprimes el resultado para verlo en consola al instante
        print(f"Recall de Candidatos: {recall_cand:.4f} ({hits_cand} encontrados de {total_reales} compras reales)")

        print('Iniciando de predicciones')
        #Hay que aplicar estrategias: Los que tengan mayor ranking bpr, los más vendidos del último mes, más populares en general, recompras
        #artículos dentro del cluster de los que compraron, etc
        #Por ahora esta creada la bpr
        predicciones_dict = models.recommend_xgboost_batch(
            xgb_model, eval_users, user_encoded_indexed, candidate_pool, candidate_genero,
            generos_por_usuario, user_factors_df, item_factors_df, feature_cols,
            historial_dict=historial_dict, cov_dict=cov_dict, top_n=k_eval,
            batch_size=inference_batch_size,
            candidate_pairs=candidate_pairs,
        )
        predicciones = [predicciones_dict.get(u, []) for u in eval_users]

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
        "recall_cand": recall_cand,
    }
    return resultado


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    TRAINING_CACHE_DIR.mkdir(parents=True, exist_ok=True)
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

    historial_dict = models.generar_historial_dict(df_train) #historial para luego generar el score de co-visitación
    # 2.2 Matriz de Co-visitación (busca en caché primero)
    df_cov = models.construir_o_cargar_covisitacion(
    df_train, 
    cache_dir=str(TRAINING_CACHE_DIR), 
    max_pairs_per_item=100
    )
    # 2.3 Conversión a diccionario anidado O(1) para máxima velocidad
    cov_dict = models.construir_diccionario_covisitacion(df_cov)
    # (Opcional) Ya puedes liberar la memoria del DataFrame de co-visitación
    del df_cov
    import gc; gc.collect()
    #Con esto ya podemos calcular el score de co-visitación para usarlo como feature, generar negativos o candidatos.

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
        use_cache=USE_CACHED_CLUSTERING,
        cache_dir=TRAINING_CACHE_DIR,
    )
    #Generamos diccionarios de clusters para usarlos en varias funciones (negativos, candidatos)
    article_to_cluster, cluster_to_articles_sorted = models.build_cluster_negative_artifacts(
        resultado_cluster["df_merged"]
    )
    print(f"Entrenamiento de clustering: {time.time() - to_cluster:.2f}s")
    print("Entrenando modelo XGBoost...")
    resultado_xgboost = entrenar_modelo_xgboost(
        df_customers, df_products, df_train, eval_users, actual,
        article_to_category, actual_categories_test, categorias_compradas_general,
        n_estimators=XGB_N_ESTIMATORS, max_depth=XGB_MAX_DEPTH, learning_rate=XGB_LEARNING_RATE,
        reg_lambda=XGB_REG_LAMBDA, reg_alpha=XGB_REG_ALPHA, subsample=XGB_SUBSAMPLE,
        colsample_bytree=XGB_COLSAMPLE_BYTREE, min_child_weight=XGB_MIN_CHILD_WEIGHT, gamma=XGB_GAMMA,
        n_negativos_por_positivo=N_NEGATIVOS_POR_POSITIVO, candidate_pool_size=CANDIDATE_POOL_SIZE,
        bpr_factors=BPR_FACTORS, bpr_iterations=BPR_ITERATIONS, bpr_regularization=BPR_REGULARIZATION,
        k_eval=K_EVAL, random_state=RANDOM_STATE, extra_params=extra_params_datos,
        historial_dict=historial_dict, cov_dict=cov_dict,
        use_hybrid_negatives=USE_HYBRID_NEGATIVES,
        negative_proportions=NEGATIVE_PROPORTIONS,
        article_to_cluster=article_to_cluster,
        cluster_to_articles_sorted=cluster_to_articles_sorted,
        bpr_neighbors_top_k=BPR_NEIGHBORS_TOP_K,
        cache_dir=TRAINING_CACHE_DIR,
        use_hybrid_candidates=USE_HYBRID_CANDIDATES,
        candidate_top_k_bpr=TOP_K_CANDIDATES_BPR,
        candidate_top_k_cluster=TOP_K_CANDIDATES_CLUSTER,
        candidate_top_k_cov=TOP_K_CANDIDATES_COV,
        candidate_top_k_pop=TOP_K_CANDIDATES_POPULAR,
        df_articles_for_candidates=resultado_cluster["df_merged"],
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

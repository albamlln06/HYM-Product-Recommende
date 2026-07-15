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

import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
sys.path.insert(0, str(BASE_DIR / "analysis"))

from utils import preprocess, models

N_CUSTOMERS = 500
MIN_PURCHASES = 3
MAX_MONTHS_SINCE_LAST_PURCHASE = 12
#Productos que se van a recomendar (nº)
K_EVAL = 12
#Semilla                 
RANDOM_STATE = 42

# --- Hiperparámetros del modelo de clustering ---
K_CLUSTERS = 8

# --- Hiperparámetros del modelo XGBoost ---
# None -> sin límite, se usa el catálogo completo como candidatos a recomendar.
# Ponle un número solo si necesitas acotar el coste de cómputo al escalar
# N_CUSTOMERS; con un número bajo, los artículos poco vendidos quedan fuera
# del pool y el modelo nunca puede recomendarlos, tunees lo que tunees.
CANDIDATE_POOL_SIZE = None
N_NEGATIVOS_POR_POSITIVO = 8
XGB_N_ESTIMATORS = 300
XGB_MAX_DEPTH = 8
XGB_LEARNING_RATE = 0.02
XGB_REG_LAMBDA = 20
XGB_SCALE_POS_WEIGHT = 5
# Por defecto usan el valor por defecto de XGBoost (= sin efecto); ajústalos
# aquí para probar otros valores en la siguiente iteración.
XGB_REG_ALPHA = 0
XGB_SUBSAMPLE = 1.0
XGB_COLSAMPLE_BYTREE = 0.7
XGB_MIN_CHILD_WEIGHT = 5
XGB_GAMMA = 1

# Estos son los valores que usa el ÚNICO modelo XGBoost que entrena main().
# Edita este dict a mano en cada iteración: es la manera de controlar tú
# mismo qué se prueba, en vez de lanzar una batería de experimentos fijos.
XGB_BASE_HYPERPARAMS = {
    "n_estimators": XGB_N_ESTIMATORS,
    "max_depth": XGB_MAX_DEPTH,
    "learning_rate": XGB_LEARNING_RATE,
    "reg_lambda": XGB_REG_LAMBDA,
    "reg_alpha": XGB_REG_ALPHA,
    "scale_pos_weight": XGB_SCALE_POS_WEIGHT,
    "subsample": XGB_SUBSAMPLE,
    "colsample_bytree": XGB_COLSAMPLE_BYTREE,
    "min_child_weight": XGB_MIN_CHILD_WEIGHT,
    "gamma": XGB_GAMMA,
    "n_negativos_por_positivo": N_NEGATIVOS_POR_POSITIVO,
}

# Nombre del run en MLflow para esta iteración, qué features usar (None ->
# FEATURE_CONFIG_DEFAULT, todas) y pesos por categoría (None -> sin cambios).
# Cambia estos tres valores junto con XGB_BASE_HYPERPARAMS entre ejecuciones.
XGB_RUN_NAME = "xgboost"
XGB_FEATURE_CONFIG = None
XGB_CATEGORY_WEIGHTS = None

# --- MLflow ---
MLFLOW_EXPERIMENT_NAME = "hym-recomendator"
MLFLOW_TRACKING_URI = f"sqlite:///{Path(__file__).resolve().parent / 'mlflow.db'}"


def leave_one_out_split(df_transactions):
    last_purchase_idx = df_transactions.groupby("customer_id")["t_dat"].idxmax()
    df_test = df_transactions.loc[last_purchase_idx]
    df_train = df_transactions.drop(index=last_purchase_idx)

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


def entrenar_modelo_random(df_train, eval_users, actual, k_eval=12, seed=42, run_name="baseline_random", extra_params=None):
    """Baseline: recomienda artículos al azar. Sirve para tener un suelo de referencia."""
    with mlflow.start_run(run_name=run_name):
        params = {"k_eval": k_eval, "seed": seed}
        if extra_params:
            params.update(extra_params)
        mlflow.log_params(params)

        predicciones = models.predict_random(df_train, eval_users, k=k_eval, seed=seed)
        map12 = models.mapk(actual, predicciones, k=k_eval)

        mlflow.log_metric("map12", map12)
        print(f"[Random]  MAP@12 = {map12:.4f}")

    return predicciones, map12


def entrenar_modelo_popular(df_train, eval_users, actual, k_eval=12, run_name="baseline_popular", extra_params=None):
    """Baseline: recomienda a todo el mundo los artículos más vendidos."""
    with mlflow.start_run(run_name=run_name):
        params = {"k_eval": k_eval}
        if extra_params:
            params.update(extra_params)
        mlflow.log_params(params)

        predicciones = models.predict_popular(df_train, eval_users, k=k_eval)
        map12 = models.mapk(actual, predicciones, k=k_eval)

        mlflow.log_metric("map12", map12)
        print(f"[Popular] MAP@12 = {map12:.4f}")

    return predicciones, map12


def entrenar_modelo_cluster(
    df_customers, df_products, df_train, eval_users, actual,
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

        print(f"[Cluster] MAP@12 = {map12:.4f}")

    resultado = {
        "kmeans_model": kmeans_model,
        "scaler": scaler,
        "X_final": X_final,
        "article_ids": article_ids,
        "df_merged": df_merged,
        "predicciones": predicciones,
        "map12": map12,
    }
    return resultado


def entrenar_modelo_xgboost(
    df_customers, df_products, df_train, eval_users, actual,
    n_estimators=XGB_N_ESTIMATORS, max_depth=XGB_MAX_DEPTH, learning_rate=XGB_LEARNING_RATE,
    reg_lambda=XGB_REG_LAMBDA, reg_alpha=XGB_REG_ALPHA, scale_pos_weight=XGB_SCALE_POS_WEIGHT,
    subsample=XGB_SUBSAMPLE, colsample_bytree=XGB_COLSAMPLE_BYTREE,
    min_child_weight=XGB_MIN_CHILD_WEIGHT, gamma=XGB_GAMMA,
    n_negativos_por_positivo=8, candidate_pool_size=CANDIDATE_POOL_SIZE,
    k_eval=12, random_state=42,
    run_name="xgboost", feature_config=None, category_weights=None, extra_params=None,
):
    """
    Entrena el modelo de ranking XGBoost y lo registra en MLflow.

    run_name        : nombre del run. Usa algo descriptivo para comparar en la UI.
    feature_config  : dict con claves article_numeric, article_categorical,
                      user_numeric, user_categorical. None → FEATURE_CONFIG_DEFAULT.
    category_weights: dict {columna_categorica: {valor: peso}} para bajar/subir
                      el peso (sample_weight) de ciertas categorías de artículo
                      en el entrenamiento, p.ej.
                      {"product_group_name": {"Underwear": 0.3}}.
                      None → todas las filas pesan igual (1.0).
    reg_lambda      : regularización L2 de XGBoost (parámetro `reg_lambda`).
    reg_alpha       : regularización L1 de XGBoost (parámetro `reg_alpha`).
    scale_pos_weight: peso de la clase positiva de XGBoost (parámetro `scale_pos_weight`).
    subsample       : fracción de filas muestreadas por árbol (parámetro `subsample`).
    colsample_bytree: fracción de columnas muestreadas por árbol (parámetro `colsample_bytree`).
    min_child_weight: peso mínimo de un nodo hijo para permitir un split (parámetro `min_child_weight`).
    gamma           : reducción mínima de pérdida exigida para hacer un split (parámetro `gamma`).
    candidate_pool_size: nº máximo de artículos candidatos a recomendar (los más
                      vendidos primero). None → sin límite, se usa el catálogo
                      completo (recomendado: si se acota, los artículos poco
                      vendidos quedan fuera y el modelo nunca puede recomendarlos).
    extra_params    : params adicionales para MLflow (p.ej. config de datos).
    """
    with mlflow.start_run(run_name=run_name):
        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "reg_lambda": reg_lambda,
            "reg_alpha": reg_alpha,
            "scale_pos_weight": scale_pos_weight,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "min_child_weight": min_child_weight,
            "gamma": gamma,
            "n_negativos_por_positivo": n_negativos_por_positivo,
            "candidate_pool_size": candidate_pool_size,
            "random_state": random_state,
        }
        if extra_params:
            params.update(extra_params)
        mlflow.log_params(params)

        # Loguear qué features se usan en este experimento
        cfg = feature_config or models.FEATURE_CONFIG_DEFAULT
        mlflow.log_param("features_article_numeric",    cfg["article_numeric"])
        mlflow.log_param("features_article_categorical", cfg["article_categorical"])
        mlflow.log_param("features_user_numeric",       cfg["user_numeric"])
        mlflow.log_param("features_user_categorical",   cfg["user_categorical"])
        mlflow.log_param("category_weights", category_weights)

        X, y, sample_weight, dataset, article_df, user_df = models.xgboost_preprocess(
            df_customers, df_products, df_train,
            n_negativos_por_positivo=n_negativos_por_positivo,
            random_state=random_state,
            feature_config=feature_config,
            category_weights=category_weights,
        )
        feature_cols = list(X.columns)

        X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
            X, y, sample_weight, test_size=0.2, random_state=random_state, stratify=y
        )
        xgb_model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            eval_metric="logloss",
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            scale_pos_weight=scale_pos_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            gamma=gamma,
            random_state=random_state,
        )
        xgb_model.fit(
            X_train, y_train,
            sample_weight=w_train,
            eval_set=[(X_val, y_val)],
            sample_weight_eval_set=[w_val],
            verbose=False,
        )

        user_encoded, article_encoded = models.encode_xgboost_categoricals(user_df, article_df, feature_config)
        candidate_pool = article_encoded.sort_values("sales_volume", ascending=False)
        if candidate_pool_size is not None:
            candidate_pool = candidate_pool.head(candidate_pool_size)
        user_encoded_indexed = user_encoded.set_index("customer_id")

        predicciones = []
        for u in eval_users:
            if u not in user_encoded_indexed.index:
                predicciones.append([])
                continue
            user_row = user_encoded_indexed.loc[u]
            recs = models.recommend_xgboost_for_user(xgb_model, user_row, candidate_pool, feature_cols, top_n=k_eval)
            predicciones.append(recs["article_id"].tolist())

        map12 = models.mapk(actual, predicciones, k=k_eval)
        mlflow.log_metric("map12", map12)
        mlflow.xgboost.log_model(xgb_model, name="xgboost_model")

        print(f"[XGBoost] {run_name} -> MAP@12 = {map12:.4f}  |  params={params}")

    resultado = {
        "xgb_model": xgb_model,
        "feature_cols": feature_cols,
        "user_df": user_df,
        "user_encoded": user_encoded,
        "candidate_pool": candidate_pool,
        "predicciones": predicciones,
        "map12": map12,
    }
    return resultado


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    print(f"Cargando muestra de {N_CUSTOMERS} clientes...")
    df_customers, df_products, df_transactions = preprocess.load_complete_dataset_filtered_number_customers(
        N_CUSTOMERS, random_state=RANDOM_STATE
    )
    df_transactions = preprocess.filter_customers_by_activity(
        df_transactions, min_purchases=MIN_PURCHASES, max_months_since_last_purchase=MAX_MONTHS_SINCE_LAST_PURCHASE
    )
    print(f"Transacciones tras filtro de actividad: {len(df_transactions):,}")

    df_train, df_test, eval_users, actual = leave_one_out_split(df_transactions)
    print(f"Usuarios de evaluación: {len(eval_users):,}  |  train: {len(df_train):,}  |  test: {len(df_test):,}")

    # --- Entrenamiento de cada modelo (cada uno queda registrado en MLflow) ---
    # data_config identifica la muestra (nº de clientes + filtro de actividad) para
    # poder comparar los 4 modelos entre sí en igualdad de condiciones (ver
    # experiment_xgboost.py, que reentrena estos mismos baselines por cada
    # combinación de datos que prueba).
    extra_params_datos = {
        "n_customers": N_CUSTOMERS,
        "min_purchases": MIN_PURCHASES,
        "max_months_since_last_purchase": MAX_MONTHS_SINCE_LAST_PURCHASE,
        "data_config": f"cust{N_CUSTOMERS}_minp{MIN_PURCHASES}_maxm{MAX_MONTHS_SINCE_LAST_PURCHASE}",
    }

    _, map_random = entrenar_modelo_random(
        df_train, eval_users, actual, k_eval=K_EVAL, seed=RANDOM_STATE, extra_params=extra_params_datos,
    )
    _, map_popular = entrenar_modelo_popular(
        df_train, eval_users, actual, k_eval=K_EVAL, extra_params=extra_params_datos,
    )

    print("Entrenando modelo de clustering...")
    resultado_cluster = entrenar_modelo_cluster(
        df_customers, df_products, df_train, eval_users, actual,
        k_clusters=K_CLUSTERS, k_eval=K_EVAL, random_state=RANDOM_STATE,
        extra_params=extra_params_datos,
    )

    print("Entrenando modelo XGBoost...")
    resultado_xgboost = entrenar_modelo_xgboost(
        df_customers, df_products, df_train, eval_users, actual,
        **XGB_BASE_HYPERPARAMS,
        candidate_pool_size=CANDIDATE_POOL_SIZE,
        k_eval=K_EVAL, random_state=RANDOM_STATE,
        run_name=XGB_RUN_NAME,
        feature_config=XGB_FEATURE_CONFIG,
        category_weights=XGB_CATEGORY_WEIGHTS,
        extra_params=extra_params_datos,
    )

    # --- Métricas ---
    raw_metrics = {
        "Random": map_random,
        "Popular": map_popular,
        "Cluster": resultado_cluster["map12"],
        "XGBoost": resultado_xgboost["map12"],
    }
    print("MAP@12:")
    for name, score in sorted(raw_metrics.items(), key=lambda x: -x[1]):
        print(f"  {name:<10} {score:.4f}")

    # --- Guardado de artefactos para el backend en vivo ---
    MODELS_DIR.mkdir(exist_ok=True)

    metrics_payload = {
        "metrics": [
            {"model": name, "map12": score}
            for name, score in sorted(raw_metrics.items(), key=lambda x: -x[1])
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

    print(f"\nArtefactos guardados en {MODELS_DIR}")


if __name__ == "__main__":
    main()

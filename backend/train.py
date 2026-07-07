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

from utils import preprocess, models  # noqa: E402

# --- Configuración de la muestra (mismo espíritu que los notebooks) ---
N_CUSTOMERS = 4000
MIN_PURCHASES = 2
MAX_MONTHS_SINCE_LAST_PURCHASE = 12
K_EVAL = 12                 # top-K para MAP@12 y para las recomendaciones servidas
RANDOM_STATE = 42

# --- Hiperparámetros del modelo de clustering ---
K_CLUSTERS = 8

# --- Hiperparámetros del modelo XGBoost ---
CANDIDATE_POOL_SIZE = 120000  # nº de artículos más vendidos usados como pool de candidatos
N_NEGATIVOS_POR_POSITIVO = 8
XGB_N_ESTIMATORS = 300
XGB_MAX_DEPTH = 6
XGB_LEARNING_RATE = 0.05

# --- MLflow ---
MLFLOW_EXPERIMENT_NAME = "hym-recomendador"


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


def entrenar_modelo_random(df_train, eval_users, actual, k_eval=12, seed=42):
    """Baseline: recomienda artículos al azar. Sirve para tener un suelo de referencia."""
    with mlflow.start_run(run_name="baseline_random"):
        mlflow.log_param("k_eval", k_eval)
        mlflow.log_param("seed", seed)

        predicciones = models.predict_random(df_train, eval_users, k=k_eval, seed=seed)
        map12 = models.mapk(actual, predicciones, k=k_eval)

        mlflow.log_metric("map12", map12)
        print(f"[Random]  MAP@12 = {map12:.4f}")

    return predicciones, map12


def entrenar_modelo_popular(df_train, eval_users, actual, k_eval=12):
    """Baseline: recomienda a todo el mundo los artículos más vendidos."""
    with mlflow.start_run(run_name="baseline_popular"):
        mlflow.log_param("k_eval", k_eval)

        predicciones = models.predict_popular(df_train, eval_users, k=k_eval)
        map12 = models.mapk(actual, predicciones, k=k_eval)

        mlflow.log_metric("map12", map12)
        print(f"[Popular] MAP@12 = {map12:.4f}")

    return predicciones, map12


def entrenar_modelo_cluster(
    df_customers, df_products, df_train, eval_users, actual,
    k_clusters=8, k_eval=12, random_state=42,
):
    """Entrena el modelo de clustering (KMeans + similitud coseno) y lo registra en MLflow."""
    with mlflow.start_run(run_name="cluster_kmeans"):
        mlflow.log_param("k_clusters", k_clusters)
        mlflow.log_param("random_state", random_state)

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
    n_estimators=300, max_depth=6, learning_rate=0.05,
    n_negativos_por_positivo=8, candidate_pool_size=120000,
    k_eval=12, random_state=42,
):
    """Entrena el modelo de ranking XGBoost y lo registra en MLflow."""
    with mlflow.start_run(run_name="xgboost"):
        mlflow.log_params({
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "n_negativos_por_positivo": n_negativos_por_positivo,
            "candidate_pool_size": candidate_pool_size,
            "random_state": random_state,
        })

        X, y, dataset, article_df, user_df = models.xgboost_preprocess(
            df_customers, df_products, df_train,
            n_negativos_por_positivo=n_negativos_por_positivo, random_state=random_state,
        )
        feature_cols = list(X.columns)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=random_state, stratify=y
        )
        xgb_model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            eval_metric="logloss",
            random_state=random_state,
        )
        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        user_encoded, article_encoded = models.encode_xgboost_categoricals(user_df, article_df)
        candidate_pool = article_encoded.sort_values("sales_volume", ascending=False).head(candidate_pool_size)
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

        print(f"[XGBoost] MAP@12 = {map12:.4f}")

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
    mlflow.set_tracking_uri(f"file:{BASE_DIR / 'mlruns'}")
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
    _, map_random = entrenar_modelo_random(df_train, eval_users, actual, k_eval=K_EVAL, seed=RANDOM_STATE)
    _, map_popular = entrenar_modelo_popular(df_train, eval_users, actual, k_eval=K_EVAL)

    print("Entrenando modelo de clustering...")
    resultado_cluster = entrenar_modelo_cluster(
        df_customers, df_products, df_train, eval_users, actual,
        k_clusters=K_CLUSTERS, k_eval=K_EVAL, random_state=RANDOM_STATE,
    )

    print("Entrenando modelo XGBoost...")
    resultado_xgboost = entrenar_modelo_xgboost(
        df_customers, df_products, df_train, eval_users, actual,
        n_estimators=XGB_N_ESTIMATORS, max_depth=XGB_MAX_DEPTH, learning_rate=XGB_LEARNING_RATE,
        n_negativos_por_positivo=N_NEGATIVOS_POR_POSITIVO, candidate_pool_size=CANDIDATE_POOL_SIZE,
        k_eval=K_EVAL, random_state=RANDOM_STATE,
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

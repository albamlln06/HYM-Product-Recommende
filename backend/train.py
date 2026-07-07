"""
Entrena/exporta los artefactos que necesita el backend en vivo (app.py).

Reutiliza exactamente la misma lógica que los notebooks de análisis
(analysis/03_train_clustering.ipynb, analysis/03_train_xgboost.ipynb):
misma muestra de clientes activos, mismo split leave-one-out, mismos
hiperparámetros. Se ejecuta una sola vez (offline) y deja todo lo
necesario para servir recomendaciones y métricas en models/.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
sys.path.insert(0, str(BASE_DIR / "analysis"))

from utils import preprocess, models  # noqa: E402

# --- Configuración de la muestra / modelos (mismo espíritu que los notebooks) ---
N_CUSTOMERS = 3000
MIN_PURCHASES = 3
MAX_MONTHS_SINCE_LAST_PURCHASE = 6
K_EVAL = 12                 # top-K para MAP@12 y para las recomendaciones servidas
K_CLUSTERS = 8
CANDIDATE_POOL_SIZE = 3000  # nº de artículos más vendidos usados como pool de candidatos para XGBoost
N_NEGATIVOS_POR_POSITIVO = 4
RANDOM_STATE = 42


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


def main():
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

    # --- Baselines ---
    pred_random = models.predict_random(df_train, eval_users, k=K_EVAL)
    pred_popular = models.predict_popular(df_train, eval_users, k=K_EVAL)

    # --- Modelo Cluster (KMeans + similitud coseno) ---
    print("Entrenando modelo de clustering...")
    X_final, article_ids, scaler, df_products_enriched = models.clustering_preprocess(
        df_customers, df_products, df_train
    )
    df_clusters, kmeans_model = models.fit_product_clustering(X_final, K_CLUSTERS, article_ids)
    df_merged, cluster_summary = models.inspect_clusters(
        df_products=df_products_enriched, df_clusters=df_clusters, category_col="product_group_name"
    )
    X_df = pd.DataFrame(X_final, index=article_ids)

    pred_cluster = []
    for u in eval_users:
        recs = models.recommend_by_cluster_similarity(u, df_train, df_merged, X_df, top_n=K_EVAL)
        pred_cluster.append(recs["article_id"].tolist() if not recs.empty else [])

    # --- Modelo XGBoost ---
    print("Entrenando modelo XGBoost...")
    X, y, dataset, article_df, user_df = models.xgboost_preprocess(
        df_customers, df_products, df_train,
        n_negativos_por_positivo=N_NEGATIVOS_POR_POSITIVO, random_state=RANDOM_STATE,
    )
    feature_cols = list(X.columns)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    xgb_model = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        eval_metric="logloss", random_state=RANDOM_STATE,
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    user_encoded, article_encoded = models.encode_xgboost_categoricals(user_df, article_df)
    candidate_pool = article_encoded.sort_values("sales_volume", ascending=False).head(CANDIDATE_POOL_SIZE)
    user_encoded_indexed = user_encoded.set_index("customer_id")

    print("Rankeando recomendaciones XGBoost para evaluación...")
    pred_xgboost = []
    for u in eval_users:
        if u not in user_encoded_indexed.index:
            pred_xgboost.append([])
            continue
        user_row = user_encoded_indexed.loc[u]
        recs = models.recommend_xgboost_for_user(xgb_model, user_row, candidate_pool, feature_cols, top_n=K_EVAL)
        pred_xgboost.append(recs["article_id"].tolist())

    # --- Métricas ---
    raw_metrics = {
        "Random": models.mapk(actual, pred_random, k=K_EVAL),
        "Popular": models.mapk(actual, pred_popular, k=K_EVAL),
        "Cluster": models.mapk(actual, pred_cluster, k=K_EVAL),
        "XGBoost": models.mapk(actual, pred_xgboost, k=K_EVAL),
    }
    print("MAP@12:")
    for name, score in sorted(raw_metrics.items(), key=lambda x: -x[1]):
        print(f"  {name:<10} {score:.4f}")

    # --- Guardado de artefactos ---
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

    joblib.dump(kmeans_model, MODELS_DIR / "kmeans.joblib")
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    joblib.dump(xgb_model, MODELS_DIR / "xgboost_model.joblib")
    (MODELS_DIR / "xgboost_feature_cols.json").write_text(json.dumps(feature_cols))

    np.save(MODELS_DIR / "cluster_X_final.npy", X_final)
    np.save(MODELS_DIR / "cluster_article_ids.npy", article_ids)

    df_merged.to_parquet(MODELS_DIR / "article_features.parquet", index=False)
    candidate_pool.to_parquet(MODELS_DIR / "candidate_articles.parquet", index=False)
    user_df.to_parquet(MODELS_DIR / "customers.parquet", index=False)
    user_encoded.to_parquet(MODELS_DIR / "customers_xgb_features.parquet", index=False)
    df_train[["customer_id", "article_id", "t_dat", "price", "is_online"]].to_parquet(
        MODELS_DIR / "train_transactions.parquet", index=False
    )

    print(f"\nArtefactos guardados en {MODELS_DIR}")


if __name__ == "__main__":
    main()

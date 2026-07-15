"""
API FastAPI que sirve, en vivo, los resultados de los modelos entrenados por
train.py (artefactos en models/): comparativa de métricas y recomendaciones
por cliente (Cluster vs XGBoost) sobre la muestra de clientes activos usada
en el entrenamiento.
"""
import json
import sys
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from mlflow.exceptions import MlflowException

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
sys.path.insert(0, str(BASE_DIR / "analysis"))

from utils import models  # noqa: E402

TOP_N = 12
MLFLOW_EXPERIMENT_NAME = "hym-recomendator"
MLFLOW_DB_PATH = Path(__file__).resolve().parent / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")

# Prefijo de run_name -> familia de modelo, para colorear/agrupar la evolución
RUN_NAME_FAMILIES = [
    ("baseline_random", "Random"),
    ("baseline_popular", "Popular"),
    ("cluster_kmeans", "Cluster"),
    ("xgb", "XGBoost"),
    ("itemitem", "ItemItem"),

]


def run_family(run_name: str) -> str:
    name = (run_name or "").lower()
    for prefix, family in RUN_NAME_FAMILIES:
        if name.startswith(prefix):
            return family
    return "Otro"

app = FastAPI(title="Panel de recomendación de productos")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5174"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# --- Carga de artefactos (una vez, al arrancar) ---
metrics_payload = json.loads((MODELS_DIR / "metrics.json").read_text())
feature_cols = json.loads((MODELS_DIR / "xgboost_feature_cols.json").read_text())

kmeans_model = joblib.load(MODELS_DIR / "kmeans.joblib")
xgb_model = joblib.load(MODELS_DIR / "xgboost_model.joblib")

article_features = pd.read_parquet(MODELS_DIR / "article_features.parquet")
candidate_articles = pd.read_parquet(MODELS_DIR / "candidate_articles.parquet")
customers = pd.read_parquet(MODELS_DIR / "customers.parquet")
customers_xgb_features = pd.read_parquet(MODELS_DIR / "customers_xgb_features.parquet")
train_transactions = pd.read_parquet(MODELS_DIR / "train_transactions.parquet")

cluster_X_final = np.load(MODELS_DIR / "cluster_X_final.npy")
cluster_article_ids = np.load(MODELS_DIR / "cluster_article_ids.npy")
X_df = pd.DataFrame(cluster_X_final, index=cluster_article_ids)

customers_indexed = customers.set_index("customer_id")
customers_xgb_indexed = customers_xgb_features.set_index("customer_id")

article_display_cols = [
    "article_id", "prod_name", "product_type_name", "product_group_name",
    "section_name", "avg_price", "cluster",
]
article_display = article_features[article_display_cols].set_index("article_id")


def article_cards(article_ids):
    """Adjunta nombre/categoría/precio a una lista de article_id, en el mismo orden."""
    cards = []
    for aid in article_ids:
        if aid not in article_display.index:
            continue
        row = article_display.loc[aid]
        cards.append({
            "article_id": int(aid),
            "prod_name": row["prod_name"],
            "product_type_name": row["product_type_name"],
            "product_group_name": row["product_group_name"],
            "section_name": row["section_name"],
            "avg_price": round(float(row["avg_price"]), 4),
        })
    return cards


@app.get("/api/metrics")
def get_metrics():
    return metrics_payload


@app.get("/api/mlflow/runs")
def get_mlflow_runs():
    try:
        runs_df = mlflow.search_runs(experiment_names=[MLFLOW_EXPERIMENT_NAME], order_by=["start_time ASC"])
    except MlflowException:
        return {"runs": []}
    if runs_df.empty:
        return {"runs": []}

    param_cols = [c for c in runs_df.columns if c.startswith("params.")]
    runs = []
    for _, row in runs_df.iterrows():
        run_name = row.get("tags.mlflow.runName") or row["run_id"][:8]
        map12 = row.get("metrics.map12")
        start_time = row.get("start_time")
        runs.append({
            "run_id": row["run_id"],
            "run_name": run_name,
            "family": run_family(run_name),
            "status": row["status"],
            "start_time": start_time.isoformat() if pd.notna(start_time) else None,
            "map12": float(map12) if pd.notna(map12) else None,
            "params": {
                c.removeprefix("params."): row[c]
                for c in param_cols
                if pd.notna(row[c])
            },
        })
    return {"runs": runs}


@app.get("/api/customers")
def search_customers(query: str = "", limit: int = 50):
    if query:
        matches = customers[customers["customer_id"].str.startswith(query)]
    else:
        matches = customers
    matches = matches.head(limit)
    return [
        {
            "customer_id": row.customer_id,
            "n_compras": int(row.user_n_compras),
            "seccion_favorita": row.user_favorite_section,
        }
        for row in matches.itertuples()
    ]


@app.get("/api/customers/{customer_id}")
def get_customer_recommendations(customer_id: str):
    if customer_id not in customers_indexed.index:
        raise HTTPException(status_code=404, detail="Cliente no encontrado en la muestra de evaluación")

    history_tx = (
        train_transactions[train_transactions["customer_id"] == customer_id]
        .sort_values("t_dat", ascending=False)
    )
    history = article_cards(history_tx["article_id"].tolist())

    cluster_recs_df = models.recommend_by_cluster_similarity(
        customer_id, train_transactions, article_features, X_df, top_n=TOP_N,
    )
    cluster_recs = article_cards(cluster_recs_df["article_id"].tolist() if not cluster_recs_df.empty else [])

    user_row = customers_xgb_indexed.loc[customer_id]
    xgb_recs_df = models.recommend_xgboost_for_user(
        xgb_model, user_row, candidate_articles, feature_cols, top_n=TOP_N,
    )
    xgb_recs = article_cards(xgb_recs_df["article_id"].tolist())

    return {
        "customer_id": customer_id,
        "n_compras": int(customers_indexed.loc[customer_id, "user_n_compras"]),
        "seccion_favorita": customers_indexed.loc[customer_id, "user_favorite_section"],
        "historial": history,
        "recomendaciones_cluster": cluster_recs,
        "recomendaciones_xgboost": xgb_recs,
    }

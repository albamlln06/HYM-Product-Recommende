"""
Búsqueda de hiperparámetros de XGBoost con Optuna.

Reutiliza exactamente la misma lógica de datos/features/evaluación que
train.py (leave_one_out_split, entrenar_modelo_xgboost, etc.) pero:

  - Carga una muestra de clientes MÁS PEQUEÑA (SEARCH_N_CUSTOMERS) y un
    candidate_pool_size MÁS PEQUEÑO (SEARCH_CANDIDATE_POOL_SIZE) que los de
    train.py, solo durante la búsqueda. El cuello de botella real de
    entrenar_modelo_xgboost es el bucle de inferencia por usuario sobre el
    candidate pool (no el fit() de XGBoost en sí), así que achicar estos dos
    valores es lo que más acorta cada trial.
  - Los datos, BPR y co-visitación se cargan UNA vez y se reutilizan en
    todos los trials (BPR/co-visitación además tienen su propia caché en
    disco, ver models.get_or_train_bpr / construir_o_cargar_covisitacion).
  - Solo se exploran los hiperparámetros del propio XGBRanker
    (n_estimators, max_depth, learning_rate, regularización L1/L2,
    subsample/colsample, min_child_weight, gamma) y
    n_negativos_por_positivo. BPR y candidate_pool_size se dejan fijos.
  - Cada trial queda registrado como su propio run de MLflow (mismo
    experimento que train.py, con params optuna_study/optuna_trial) para
    poder comparar todos los trials en `mlflow ui`.

--use-hybrid-candidates (activado por defecto): si el candidate pool de
    evaluación coincide o no con el de train.py (USE_HYBRID_CANDIDATES=True,
    candidatos personalizados por usuario vía BPR/cluster/covisitación/
    populares) importa MUCHO para qué hiperparámetros salen "mejores" — un
    trial tuneado contra un pool genérico de más-vendidos (--no-hybrid-
    candidates) puede rankear fatal contra el pool personalizado real de
    train.py, y viceversa. Por eso train.py, EvolutionPanel (frontend) y este
    flag están todos alineados: usa el mismo valor aquí que el
    USE_HYBRID_CANDIDATES con el que vayas a correr train.py después.

Como usa una muestra reducida, el MAP@12 de cada trial NO es directamente
comparable al de un run completo de train.py: sirve para RANKEAR
combinaciones de hiperparámetros entre sí, no como métrica final. Una vez
tienes los mejores hiperparámetros (impresos al final y guardados en
optuna_best_params.json), pégalos en las constantes XGB_* de train.py (o en
COMBINACIONES de experiment_xgboost.py) y vuelve a correr
`python backend/train.py` con la muestra completa para validar y regenerar
los artefactos finales.

Uso:
    python backend/optuna_search.py
    python backend/optuna_search.py --n-trials 50
    python backend/optuna_search.py --study-name mi_busqueda --n-trials 20
    python backend/optuna_search.py --study-name xgboost_hpo_no_hybrid --no-hybrid-candidates
"""
import argparse
import json
from pathlib import Path

import mlflow
import optuna

from train import (
    BASE_DIR,
    MIN_PURCHASES,
    MAX_MONTHS_SINCE_LAST_PURCHASE,
    N_TEST_PURCHASES,
    K_EVAL,
    RANDOM_STATE,
    BPR_FACTORS,
    BPR_ITERATIONS,
    BPR_REGULARIZATION,
    BPR_NEIGHBORS_TOP_K,
    K_CLUSTERS,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    preprocess,
    models,
    leave_one_out_split,
    build_category_ground_truth,
    entrenar_modelo_xgboost,
)

# --- Muestra reducida solo para la búsqueda (más rápida que la de train.py) ---
SEARCH_N_CUSTOMERS = 3000
SEARCH_CANDIDATE_POOL_SIZE = 40000
# Caché propia (distinta de backend/bpr_cache que usa train.py) para no
# invalidar entre sí las cachés de co-visitación al alternar entre este
# script y train.py con tamaños de muestra distintos.
SEARCH_COVISITATION_CACHE_DIR = str(BASE_DIR / "backend" / "bpr_cache_search")
# Idem para la caché de clustering (usada para los negativos por cluster).
SEARCH_CLUSTERING_CACHE_DIR = str(BASE_DIR / "backend" / "cluster_cache_search")

OPTUNA_STORAGE = f"sqlite:///{Path(__file__).resolve().parent / 'optuna_study.db'}"


def cargar_datos_busqueda():
    print(f"Cargando muestra de búsqueda: {SEARCH_N_CUSTOMERS} clientes...")
    df_customers, df_products, df_transactions = preprocess.load_complete_dataset_filtered_number_customers(
        SEARCH_N_CUSTOMERS, random_state=RANDOM_STATE
    )
    df_transactions = preprocess.filter_customers_by_activity(
        df_transactions, min_purchases=MIN_PURCHASES, max_months_since_last_purchase=MAX_MONTHS_SINCE_LAST_PURCHASE
    )
    df_train, df_test, eval_users, actual = leave_one_out_split(df_transactions, n_test=N_TEST_PURCHASES)
    print(f"Usuarios de evaluación: {len(eval_users):,}  |  train: {len(df_train):,}  |  test: {len(df_test):,}")

    historial_dict = models.generar_historial_dict(df_train)
    df_cov = models.construir_o_cargar_covisitacion(
        df_train, cache_dir=SEARCH_COVISITATION_CACHE_DIR, max_pairs_per_item=100
    )
    cov_dict = models.construir_diccionario_covisitacion(df_cov)
    del df_cov

    # Clustering (KMeans sobre atributos de producto), necesario para poder
    # generar negativos "duros" por cluster (ver generar_negativos_cluster en
    # models.py): un negativo del mismo cluster que un artículo comprado es
    # mucho más informativo para XGBoost que uno elegido al azar.
    cluster_artifacts = models.construir_o_cargar_clustering(
        df_customers, df_products, df_train,
        k_clusters=K_CLUSTERS,
        cache_dir=SEARCH_CLUSTERING_CACHE_DIR,
    )
    article_to_cluster, cluster_to_articles_sorted = models.build_cluster_negative_artifacts(
        cluster_artifacts["df_merged"]
    )

    article_to_category, actual_categories_test, categorias_compradas_general = build_category_ground_truth(
        df_products, df_train, eval_users, actual,
    )
    return dict(
        df_customers=df_customers, df_products=df_products, df_train=df_train,
        eval_users=eval_users, actual=actual, historial_dict=historial_dict, cov_dict=cov_dict,
        article_to_cluster=article_to_cluster, cluster_to_articles_sorted=cluster_to_articles_sorted,
        # df_merged (article_id + cluster + features) también sirve como
        # df_articles_for_candidates para generar_candidatos_hibridos cuando
        # --use-hybrid-candidates está activo.
        df_articles_for_candidates=cluster_artifacts["df_merged"],
        article_to_category=article_to_category, actual_categories_test=actual_categories_test,
        categorias_compradas_general=categorias_compradas_general,
    )


def make_objective(datos, study_name, use_hybrid_candidates):
    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 600, step=50),
            max_depth=trial.suggest_int("max_depth", 3, 8),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
            gamma=trial.suggest_float("gamma", 0.0, 5.0),
            n_negativos_por_positivo=trial.suggest_int("n_negativos_por_positivo", 4, 12),
        )

        # Proporciones de negativos "duros" (cluster/bpr/cov): lo que sobra
        # hasta 1.0 se rellena con negativos "fáciles" por popularidad. Deja
        # que Optuna decida cuánto pesa cada estrategia de hard negatives en
        # vez de usar el reparto fijo de NEGATIVE_PROPORTIONS de train.py.
        prop_cluster = trial.suggest_float("neg_prop_cluster", 0.0, 0.5)
        prop_bpr = trial.suggest_float("neg_prop_bpr", 0.0, 0.5)
        prop_cov = trial.suggest_float("neg_prop_cov", 0.0, 0.3)
        negative_proportions = {
            "popular": max(0.0, 1.0 - prop_cluster - prop_bpr - prop_cov),
            "cluster": prop_cluster,
            "bpr": prop_bpr,
            "cov": prop_cov,
        }
        cand_params = {}
        if use_hybrid_candidates:
            # Ampliamos los rangos según la naturaleza de cada técnica
            cand_params["candidate_top_k_bpr"] = trial.suggest_int("cand_top_k_bpr", 100, 400, step=10)
            cand_params["candidate_top_k_cluster"] = trial.suggest_int("cand_top_k_cluster", 50, 200, step=5)
            cand_params["candidate_top_k_cov"] = trial.suggest_int("cand_top_k_cov", 50, 150, step=5)
            #cand_params["candidate_top_k_pop"] = trial.suggest_int("cand_top_k_pop", 100, 500, step=5)
            #cand_params["candidate_top_k_seccion"] = trial.suggest_int("cand_top_k_seccion", 50, 300, step=5)

        resultado = entrenar_modelo_xgboost(
            datos["df_customers"], datos["df_products"], datos["df_train"],
            datos["eval_users"], datos["actual"],
            datos["article_to_category"], datos["actual_categories_test"], datos["categorias_compradas_general"],
            candidate_pool_size=SEARCH_CANDIDATE_POOL_SIZE,
            bpr_factors=BPR_FACTORS, bpr_iterations=BPR_ITERATIONS, bpr_regularization=BPR_REGULARIZATION,
            k_eval=K_EVAL, random_state=RANDOM_STATE,
            run_name=f"optuna_{study_name}_trial{trial.number}",
            extra_params={
                "optuna_study": study_name,
                "optuna_trial": trial.number,
                "n_customers": SEARCH_N_CUSTOMERS,
                "search_run": True,
            },
            historial_dict=datos["historial_dict"], cov_dict=datos["cov_dict"],
            use_hybrid_negatives=True,
            negative_proportions=negative_proportions,
            article_to_cluster=datos["article_to_cluster"],
            cluster_to_articles_sorted=datos["cluster_to_articles_sorted"],
            bpr_neighbors_top_k=BPR_NEIGHBORS_TOP_K,
            # Debe coincidir con el USE_HYBRID_CANDIDATES con el que se vaya a
            # correr train.py después: si no, los hiperparámetros se tunean
            # para un candidate pool en la búsqueda y se aplican sobre otro
            # distinto en producción (ver comentario del módulo).
            use_hybrid_candidates=use_hybrid_candidates,
            df_articles_for_candidates=datos["df_articles_for_candidates"],
            **params,
            **cand_params,
        )

        trial.set_user_attr("recall_candidatos", resultado["recall_cand"])
        trial.set_user_attr("hit_rate_final", resultado["hit_rate"])
        trial.set_user_attr("total_hits", resultado["total_hits"])
        trial.set_user_attr("cat_hit_test", resultado["category_hit_rate_test"])
        trial.set_user_attr("cat_hit_general", resultado["category_hit_rate_general"])
        trial.set_user_attr("hit_rate_rel", resultado["hit_rate_rel"])
            
        return resultado["map12"]

    return objective


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-trials", type=int, default=100, help="Número de combinaciones a probar")
    parser.add_argument("--study-name", type=str, default="xgboost_hpo", help="Nombre del estudio de Optuna")
    parser.add_argument(
        "--use-hybrid-candidates", dest="use_hybrid_candidates", action="store_true", default=True,
        help="Candidatos personalizados por usuario (BPR/cluster/covisitación/populares), igual que train.py. Activado por defecto.",
    )
    parser.add_argument(
        "--no-hybrid-candidates", dest="use_hybrid_candidates", action="store_false",
        help="Evalúa contra el candidate pool genérico (top-N por popularidad). Úsalo solo si vas a correr train.py también con USE_HYBRID_CANDIDATES=False.",
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    datos = cargar_datos_busqueda()

    study = optuna.create_study(
        study_name=args.study_name,
        storage=OPTUNA_STORAGE,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        load_if_exists=True,
    )
    study.optimize(
        make_objective(datos, args.study_name, args.use_hybrid_candidates),
        n_trials=args.n_trials,
    )

    print(f"\nMejor MAP@12 (muestra de búsqueda, {SEARCH_N_CUSTOMERS} clientes): {study.best_value:.4f}")
    print(f"Mejores hiperparámetros:\n{json.dumps(study.best_params, indent=2)}")

    best_path = Path(__file__).resolve().parent / "optuna_best_params.json"
    best_path.write_text(json.dumps({"best_value": study.best_value, "best_params": study.best_params}, indent=2))
    print(f"\nGuardado en {best_path}")
    print(
        "Siguiente paso: copia estos valores en las constantes XGB_* de "
        "train.py (o pásalos a entrenar_modelo_xgboost) y vuelve a correr "
        "`python backend/train.py` con la muestra completa para validar y "
        "regenerar los artefactos finales."
    )
    print(f"Compara todos los trials en MLflow con: mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")
    print(f"O inspecciona el estudio de Optuna con: optuna-dashboard {OPTUNA_STORAGE}")


if __name__ == "__main__":
    main()

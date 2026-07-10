"""
Script para iterar rápido sobre hiperparámetros de XGBoost.

Cada elemento de COMBINACIONES puede tocar dos tipos de parámetros:

  - Parámetros de DATOS (n_customers, min_purchases, max_months_since_last_purchase):
    cambian la muestra sobre la que se entrena. Recargar los datos es lento,
    así que el script los cachea y solo vuelve a cargar si cambian.

  - Parámetros del MODELO (n_estimators, max_depth, learning_rate,
    reg_lambda, reg_alpha, scale_pos_weight, subsample, colsample_bytree,
    min_child_weight, gamma, n_negativos_por_positivo, candidate_pool_size):
    se pasan directos a entrenar_modelo_xgboost.

Lo que no indiques en una combinación se rellena con el valor por defecto
de train.py. Cada combinación queda como un run de MLflow con un nombre
descriptivo (con sus hiperparámetros en el propio nombre) y todos sus
parámetros registrados, para poder compararlos luego en `mlflow ui` sin
tener que entrar run por run.

Además, la PRIMERA vez que se usa una combinación de datos nueva (nueva
n_customers/min_purchases/max_months_since_last_purchase), se entrenan
también los 3 modelos de referencia (Random, Popular, Cluster) sobre esa
misma muestra y split, todos con el mismo tag `data_config`. Así, para
cualquier combinación de datos, se puede comparar en igualdad de
condiciones qué modelo rinde mejor (panel "Evolución de modelos" del
frontend, sección de comparativa).

Uso:
    python backend/experiment_xgboost.py
"""
import mlflow

from train import (
    BASE_DIR,
    N_CUSTOMERS,
    MIN_PURCHASES,
    MAX_MONTHS_SINCE_LAST_PURCHASE,
    N_NEGATIVOS_POR_POSITIVO,
    CANDIDATE_POOL_SIZE,
    XGB_REG_LAMBDA,
    XGB_REG_ALPHA,
    XGB_SCALE_POS_WEIGHT,
    XGB_SUBSAMPLE,
    XGB_COLSAMPLE_BYTREE,
    XGB_MIN_CHILD_WEIGHT,
    XGB_GAMMA,
    RANDOM_STATE,
    K_EVAL,
    K_CLUSTERS,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    preprocess,
    leave_one_out_split,
    entrenar_modelo_random,
    entrenar_modelo_popular,
    entrenar_modelo_cluster,
    entrenar_modelo_xgboost,
)

# --- Combinaciones a probar ---
COMBINACIONES = [
    dict(n_estimators=300, max_depth=6, learning_rate=0.05),
    dict(n_estimators=500, max_depth=4, learning_rate=0.10),
    dict(n_customers=2000, min_purchases=3, n_estimators=300, max_depth=6, learning_rate=0.05),
    dict(n_customers=6000, max_months_since_last_purchase=6, n_estimators=300, max_depth=6, learning_rate=0.05),
]


def data_config_id(n_customers, min_purchases, max_months_since_last_purchase):
    return f"cust{n_customers}_minp{min_purchases}_maxm{max_months_since_last_purchase}"


def entrenar_baselines_para_combinacion(
    df_customers, df_products, df_train, eval_users, actual,
    n_customers, min_purchases, max_months_since_last_purchase,
):
    """
    Entrena Random, Popular y Cluster sobre la MISMA muestra/split que se va a
    usar para XGBoost, y los etiqueta con el mismo `data_config` para poder
    compararlos entre sí más adelante (mismos eval_users, mismo df_train).
    """
    config_id = data_config_id(n_customers, min_purchases, max_months_since_last_purchase)
    extra_params = {
        "n_customers": n_customers,
        "min_purchases": min_purchases,
        "max_months_since_last_purchase": max_months_since_last_purchase,
        "data_config": config_id,
    }

    entrenar_modelo_random(
        df_train, eval_users, actual, k_eval=K_EVAL, seed=RANDOM_STATE,
        run_name=f"baseline_random_{config_id}", extra_params=extra_params,
    )
    entrenar_modelo_popular(
        df_train, eval_users, actual, k_eval=K_EVAL,
        run_name=f"baseline_popular_{config_id}", extra_params=extra_params,
    )
    entrenar_modelo_cluster(
        df_customers, df_products, df_train, eval_users, actual,
        k_clusters=K_CLUSTERS, k_eval=K_EVAL, random_state=RANDOM_STATE,
        run_name=f"cluster_kmeans_{config_id}", extra_params=extra_params,
    )


def cargar_datos(n_customers, min_purchases, max_months_since_last_purchase):
    print(
        f"Cargando datos: n_customers={n_customers}, min_purchases={min_purchases}, "
        f"max_months_since_last_purchase={max_months_since_last_purchase}"
    )
    df_customers, df_products, df_transactions = preprocess.load_complete_dataset_filtered_number_customers(
        n_customers, random_state=RANDOM_STATE
    )
    df_transactions = preprocess.filter_customers_by_activity(
        df_transactions,
        min_purchases=min_purchases,
        max_months_since_last_purchase=max_months_since_last_purchase,
    )
    df_train, df_test, eval_users, actual = leave_one_out_split(df_transactions)
    print(f"Usuarios de evaluación: {len(eval_users):,}  |  train: {len(df_train):,}")
    return df_customers, df_products, df_train, eval_users, actual


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    cache_datos = {}  # (n_customers, min_purchases, max_months) -> datos ya cargados

    for i, combinacion in enumerate(COMBINACIONES, start=1):
        # Parámetros de datos: si no se indican, se usan los de train.py
        n_customers = combinacion.get("n_customers", N_CUSTOMERS)
        min_purchases = combinacion.get("min_purchases", MIN_PURCHASES)
        max_months = combinacion.get("max_months_since_last_purchase", MAX_MONTHS_SINCE_LAST_PURCHASE)

        clave_datos = (n_customers, min_purchases, max_months)
        if clave_datos not in cache_datos:
            cache_datos[clave_datos] = cargar_datos(n_customers, min_purchases, max_months)
            df_customers, df_products, df_train, eval_users, actual = cache_datos[clave_datos]
            print(f"Entrenando baselines de referencia (Random/Popular/Cluster) para {clave_datos}...")
            entrenar_baselines_para_combinacion(
                df_customers, df_products, df_train, eval_users, actual,
                n_customers, min_purchases, max_months,
            )
        df_customers, df_products, df_train, eval_users, actual = cache_datos[clave_datos]

        # Parámetros del modelo: si no se indican, se usan los de train.py
        n_estimators = combinacion.get("n_estimators", 300)
        max_depth = combinacion.get("max_depth", 6)
        learning_rate = combinacion.get("learning_rate", 0.05)
        reg_lambda = combinacion.get("reg_lambda", XGB_REG_LAMBDA)
        reg_alpha = combinacion.get("reg_alpha", XGB_REG_ALPHA)
        scale_pos_weight = combinacion.get("scale_pos_weight", XGB_SCALE_POS_WEIGHT)
        subsample = combinacion.get("subsample", XGB_SUBSAMPLE)
        colsample_bytree = combinacion.get("colsample_bytree", XGB_COLSAMPLE_BYTREE)
        min_child_weight = combinacion.get("min_child_weight", XGB_MIN_CHILD_WEIGHT)
        gamma = combinacion.get("gamma", XGB_GAMMA)
        n_negativos_por_positivo = combinacion.get("n_negativos_por_positivo", N_NEGATIVOS_POR_POSITIVO)
        candidate_pool_size = combinacion.get("candidate_pool_size", CANDIDATE_POOL_SIZE)

        run_name = f"xgb_n{n_estimators}_d{max_depth}_lr{learning_rate}_cust{n_customers}_minp{min_purchases}"
        print(f"\n--- Combinación {i}/{len(COMBINACIONES)}: {combinacion} ---")

        entrenar_modelo_xgboost(
            df_customers, df_products, df_train, eval_users, actual,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            scale_pos_weight=scale_pos_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            gamma=gamma,
            n_negativos_por_positivo=n_negativos_por_positivo,
            candidate_pool_size=candidate_pool_size,
            random_state=RANDOM_STATE,
            k_eval=K_EVAL,
            run_name=run_name,
            extra_params={
                "n_customers": n_customers,
                "min_purchases": min_purchases,
                "max_months_since_last_purchase": max_months,
                "data_config": data_config_id(n_customers, min_purchases, max_months),
            },
        )

    print("\nListo. Compara los runs con: mlflow ui --backend-store-uri file:./mlruns")


if __name__ == "__main__":
    main()

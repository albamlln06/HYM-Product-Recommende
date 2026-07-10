import os
import tempfile

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ── Palette tokens (reference palette, light mode) ──────────────────────────
_C = {
    "blue":    "#2a78d6",
    "aqua":    "#1baf7a",
    "yellow":  "#eda100",
    "green":   "#008300",
    "violet":  "#4a3aa7",
    "red":     "#e34948",
    "muted":   "#898781",
    "surface": "#fcfcfb",
    "ink1":    "#0b0b0b",
    "ink2":    "#52514e",
    "grid":    "#e1e0d9",
}
_CATEGORICAL = [_C["blue"], _C["aqua"], _C["yellow"], _C["green"], _C["violet"], _C["red"]]


def _base_ax(ax):
    ax.set_facecolor(_C["surface"])
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(_C["grid"])
    ax.tick_params(colors=_C["ink2"])
    ax.yaxis.label.set_color(_C["ink2"])
    ax.xaxis.label.set_color(_C["ink2"])
    ax.title.set_color(_C["ink1"])
    ax.xaxis.grid(True, color=_C["grid"], linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def plot_model_comparison(raw_metrics, figsize=(7, 3.5)):
    """
    Horizontal bar chart comparando MAP@12 de todos los modelos.
    El mejor queda en azul (énfasis); el resto en gris muted.
    Devuelve la figura — guárdala con fig.savefig() o pásala a mlflow.log_figure().
    """
    pairs = sorted(raw_metrics.items(), key=lambda x: x[1])
    names, scores = zip(*pairs)
    best_score = max(scores)

    colors = [_C["blue"] if s == best_score else _C["muted"] for s in scores]

    fig, ax = plt.subplots(figsize=figsize, facecolor=_C["surface"])
    bars = ax.barh(range(len(names)), scores, color=colors, height=0.55)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, color=_C["ink1"], fontsize=9)
    ax.set_xlabel("MAP@12", color=_C["ink2"])
    ax.set_title("Comparativa de modelos — MAP@12", color=_C["ink1"], fontsize=11, pad=8)
    _base_ax(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)

    for bar, score in zip(bars, scores):
        ax.text(
            score + best_score * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.4f}",
            va="center", fontsize=8, color=_C["ink2"],
        )

    plt.tight_layout()
    return fig


def plot_feature_importance(xgb_model, feature_cols, top_n=15, figsize=None):
    """
    Horizontal bar chart con las top_n features más importantes del modelo XGBoost.
    Usa una rampa secuencial (un solo tono azul) porque el job es magnitud, no identidad.
    """
    importances = xgb_model.feature_importances_
    pairs = sorted(zip(importances, feature_cols), reverse=True)[:top_n]
    pairs = list(reversed(pairs))
    scores, names = zip(*pairs)

    figsize = figsize or (7, max(3, top_n * 0.38 + 1))
    fig, ax = plt.subplots(figsize=figsize, facecolor=_C["surface"])

    max_s = max(scores)
    # Sequential blue: lighter = menor importancia, oscuro = mayor
    bar_colors = [plt.cm.Blues(0.3 + 0.65 * (s / max_s)) for s in scores]

    ax.barh(range(len(names)), scores, color=bar_colors, height=0.65)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, color=_C["ink1"], fontsize=8)
    ax.set_xlabel("Importancia (gain)", color=_C["ink2"])
    ax.set_title(f"Feature Importance — Top {top_n}", color=_C["ink1"], fontsize=11, pad=8)
    _base_ax(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)

    plt.tight_layout()
    return fig


def plot_map_at_k(actual, predictions_dict, max_k=12, figsize=(7, 4)):
    """
    Líneas MAP@K (K de 1 a max_k) para cada modelo en predictions_dict.
    predictions_dict = {"nombre": lista_de_listas_de_predicciones}
    """
    from models import mapk  # importación local para evitar circular

    ks = list(range(1, max_k + 1))
    fig, ax = plt.subplots(figsize=figsize, facecolor=_C["surface"])

    for i, (name, preds) in enumerate(predictions_dict.items()):
        scores = [mapk(actual, preds, k=k) for k in ks]
        color = _CATEGORICAL[i % len(_CATEGORICAL)]
        ax.plot(ks, scores, marker="o", markersize=4, linewidth=2, color=color, label=name)
        # Direct label al final de la línea
        ax.text(max_k + 0.1, scores[-1], name, va="center", fontsize=7.5, color=color)

    ax.set_xlabel("K (nº de recomendaciones)", color=_C["ink2"])
    ax.set_ylabel("MAP@K", color=_C["ink2"])
    ax.set_title("Curva de precisión MAP@K", color=_C["ink1"], fontsize=11, pad=8)
    ax.set_xticks(ks)
    ax.yaxis.grid(True, color=_C["grid"], linewidth=0.5, zorder=0)
    _base_ax(ax)
    ax.spines["right"].set_visible(False)
    # sin legend box: nombres directamente en la línea
    plt.tight_layout()
    return fig


def log_figures_to_mlflow(figures_dict):
    """
    Guarda un dict {nombre: figura_matplotlib} como artefactos en el run activo de MLflow.
    Llama dentro de un mlflow.start_run() context.
    """
    import mlflow

    with tempfile.TemporaryDirectory() as tmp:
        for name, fig in figures_dict.items():
            path = os.path.join(tmp, f"{name}.png")
            fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
            mlflow.log_artifact(path, artifact_path="plots")
            plt.close(fig)


def plot_clusters_by_k(X_final, k_values, sample_size=3000, random_state=42):
    """
    Pinta los clusters en 2D (PCA) para cada valor de K en k_values.
    Útil para validar visualmente si el K elegido produce clusters coherentes.

    Args:
        X_final:      array escalado de features (salida de clustering_preprocess).
        k_values:     lista de enteros K a comparar, ej. [4, 6, 8, 10].
        sample_size:  nº de puntos a mostrar (submuestreo para velocidad).
        random_state: semilla de reproducibilidad.
    """
    rng = np.random.default_rng(random_state)

    if len(X_final) > sample_size:
        idx = rng.choice(len(X_final), size=sample_size, replace=False)
        X_plot = X_final[idx]
    else:
        X_plot = X_final

    X_2d = PCA(n_components=2, random_state=random_state).fit_transform(X_plot)

    n_cols = min(3, len(k_values))
    n_rows = (len(k_values) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = np.array(axes).flatten()

    for i, k in enumerate(k_values):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_plot)

        axes[i].scatter(
            X_2d[:, 0], X_2d[:, 1],
            c=labels, cmap="tab20", s=8, alpha=0.6,
        )
        centroids_2d = PCA(n_components=2, random_state=random_state).fit(X_plot).transform(km.cluster_centers_)
        axes[i].scatter(
            centroids_2d[:, 0], centroids_2d[:, 1],
            c="black", marker="X", s=120, zorder=5, label="Centroids",
        )
        x_lo, x_hi = np.percentile(X_2d[:, 0], [1, 99])
        y_lo, y_hi = np.percentile(X_2d[:, 1], [1, 99])
        axes[i].set_xlim(x_lo, x_hi)
        axes[i].set_ylim(y_lo, y_hi)
        axes[i].set_title(f"K = {k}", fontsize=13)
        axes[i].set_xlabel("PC1")
        axes[i].set_ylabel("PC2")
        axes[i].legend(fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Clusters por valor de K (proyección PCA 2D)", fontsize=15, y=1.02)
    plt.tight_layout()
    plt.show()

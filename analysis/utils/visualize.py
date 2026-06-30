import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


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

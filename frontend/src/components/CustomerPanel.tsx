import { useEffect, useState } from "react";
import {
  getCustomerRecommendations,
  searchCustomers,
  type CustomerRecommendations,
  type CustomerSummary,
} from "../api";
import { ProductList } from "./ProductList";

export default function CustomerPanel() {
  const [query, setQuery] = useState("");
  const [matches, setMatches] = useState<CustomerSummary[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [details, setDetails] = useState<CustomerRecommendations | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const handle = setTimeout(() => {
      searchCustomers(query).then(setMatches).catch(() => setMatches([]));
    }, 250);
    return () => clearTimeout(handle);
  }, [query]);

  useEffect(() => {
    if (!selectedId) return;
    setLoading(true);
    setError(null);
    getCustomerRecommendations(selectedId)
      .then((data) => {
        setDetails(data);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setDetails(null);
        setLoading(false);
      });
  }, [selectedId]);

  return (
    <section>
      <h2>Recomendaciones por cliente</h2>
      <p className="muted">
        Búsqueda dentro de la muestra de clientes activos usada en la evaluación (no el
        catálogo completo de clientes).
      </p>

      <input
        className="search-input"
        type="text"
        placeholder="Buscar por customer_id…"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
      />

      {matches.length > 0 && (
        <ul className="customer-matches">
          {matches.map((c) => (
            <li key={c.customer_id}>
              <button className="customer-option" onClick={() => setSelectedId(c.customer_id)}>
                <span className="tabular">{c.customer_id.slice(0, 16)}…</span>
                <span className="muted">{c.n_compras} compras · fav: {c.seccion_favorita}</span>
              </button>
            </li>
          ))}
        </ul>
      )}

      {loading && <p className="muted">Cargando recomendaciones…</p>}
      {error && <p className="error">{error}</p>}

      {details && (
        <div className="customer-detail">
          <h3 className="tabular">{details.customer_id}</h3>
          <p className="muted">
            {details.n_compras} compras · sección favorita: {details.seccion_favorita}
          </p>

          <ProductList title={`Histórico de compras (${details.historial.length})`} articles={details.historial} />

          <div className="recommendation-columns">
            <ProductList title="Modelo Cluster (KMeans)" articles={details.recomendaciones_cluster} />
            <ProductList title="Modelo XGBoost" articles={details.recomendaciones_xgboost} />
          </div>
        </div>
      )}
    </section>
  );
}

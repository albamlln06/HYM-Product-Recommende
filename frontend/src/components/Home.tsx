import { useState } from "react";
import type { ArticleCard } from "../api";
import { ProductList } from "./ProductList";

// Datos de ejemplo fijos: esta pantalla es solo la maqueta visual todavía,
// no llama a la API real (ver conversación — se conectará más adelante).
const MOCK_CUSTOMER = {
  customer_id: "0123abc4d56ef789012345abcdef67890123456789abcdef0123456789abcd",
  n_compras: 27,
  seccion_favorita: "Womens Everyday Collection",
};

const MOCK_CLUSTER_RECS: ArticleCard[] = [
  { article_id: 1, prod_name: "Milk RW slack", product_type_name: "Trousers", product_group_name: "Garment Lower body", section_name: "Womens Everyday Collection", avg_price: 0.0318 },
  { article_id: 2, prod_name: "Space 5 pkt RW tregging", product_type_name: "Trousers", product_group_name: "Garment Lower body", section_name: "Womens Everyday Collection", avg_price: 0.0247 },
  { article_id: 3, prod_name: "Felice tapered", product_type_name: "Trousers", product_group_name: "Garment Lower body", section_name: "Womens Everyday Collection", avg_price: 0.0371 },
  { article_id: 4, prod_name: "Tea 3-pack socks", product_type_name: "Socks", product_group_name: "Socks & Tights", section_name: "Ladies H&M Sport", avg_price: 0.0115 },
];

const MOCK_XGBOOST_RECS: ArticleCard[] = [
  { article_id: 5, prod_name: "Tulip jumper", product_type_name: "Sweater", product_group_name: "Garment Upper body", section_name: "Womens Everyday Collection", avg_price: 0.0236 },
  { article_id: 6, prod_name: "Tulip", product_type_name: "Sweater", product_group_name: "Garment Upper body", section_name: "Womens Everyday Collection", avg_price: 0.0215 },
  { article_id: 7, prod_name: "Amelie", product_type_name: "Blazer", product_group_name: "Garment Upper body", section_name: "Womens Everyday Collection", avg_price: 0.0432 },
  { article_id: 8, prod_name: "Brody", product_type_name: "T-shirt", product_group_name: "Garment Upper body", section_name: "Womens Everyday Basics", avg_price: 0.0085 },
];

export default function Home() {
  const [customerId, setCustomerId] = useState("");
  const [loading, setLoading] = useState(false);
  const [showResults, setShowResults] = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!customerId.trim()) return;
    setLoading(true);
    setShowResults(false);
    setTimeout(() => {
      setLoading(false);
      setShowResults(true);
    }, 500);
  };

  return (
    <section className="home-hero">
      <div className="home-intro">
        <h2>Recomendaciones al instante</h2>
        <p className="muted">
          Introduce el ID de un cliente para ver qué le recomienda cada modelo. Para el resto
          de análisis, usa las pestañas de arriba.
        </p>
      </div>

      <form className="home-search" onSubmit={handleSubmit}>
        <input
          className="search-input home-search-input"
          type="text"
          placeholder="customer_id…"
          value={customerId}
          onChange={(e) => setCustomerId(e.target.value)}
        />
        <button type="submit" className="refresh-button home-search-button">
          Ver recomendaciones
        </button>
      </form>

      {loading && <p className="muted">Buscando recomendaciones…</p>}

      {showResults && (
        <div className="customer-detail">
          <p className="muted">
            Vista previa con datos de ejemplo — todavía no conectada a la API real.
          </p>
          <h3 className="tabular">{MOCK_CUSTOMER.customer_id}</h3>
          <p className="muted">
            {MOCK_CUSTOMER.n_compras} compras · sección favorita: {MOCK_CUSTOMER.seccion_favorita}
          </p>

          <div className="recommendation-columns">
            <ProductList title="Modelo Cluster (KMeans)" articles={MOCK_CLUSTER_RECS} />
            <ProductList title="Modelo XGBoost" articles={MOCK_XGBOOST_RECS} />
          </div>
        </div>
      )}
    </section>
  );
}

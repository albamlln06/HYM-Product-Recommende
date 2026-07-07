const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export interface MetricEntry {
  model: string;
  map12: number;
}

export interface MetricsResponse {
  metrics: MetricEntry[];
  meta: {
    n_customers_sampled: number;
    n_eval_users: number;
    k: number;
    k_clusters: number;
    trained_at: string;
  };
}

export interface CustomerSummary {
  customer_id: string;
  n_compras: number;
  seccion_favorita: string;
}

export interface ArticleCard {
  article_id: number;
  prod_name: string;
  product_type_name: string;
  product_group_name: string;
  section_name: string;
  avg_price: number;
}

export interface CustomerRecommendations {
  customer_id: string;
  n_compras: number;
  seccion_favorita: string;
  historial: ArticleCard[];
  recomendaciones_cluster: ArticleCard[];
  recomendaciones_xgboost: ArticleCard[];
}

async function fetchJson<T>(path: string): Promise<T> {
  const res = await fetch(`${API_URL}${path}`);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `Error ${res.status} al llamar a ${path}`);
  }
  return res.json();
}

export function getMetrics(): Promise<MetricsResponse> {
  return fetchJson("/api/metrics");
}

export function searchCustomers(query: string): Promise<CustomerSummary[]> {
  const params = new URLSearchParams();
  if (query) params.set("query", query);
  return fetchJson(`/api/customers?${params.toString()}`);
}

export function getCustomerRecommendations(customerId: string): Promise<CustomerRecommendations> {
  return fetchJson(`/api/customers/${encodeURIComponent(customerId)}`);
}

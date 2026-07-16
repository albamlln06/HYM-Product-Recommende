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

export interface MlflowRun {
  run_id: string;
  run_name: string;
  family: string;
  status: string;
  start_time: string | null;
  map12: number | null;
  params: Record<string, string>;
}

export interface MlflowRunsResponse {
  runs: MlflowRun[];
}

export interface Profile {
  customer_id: string;
  display_name: string;
  age: number | null;
  club_member_status: string | null;
  historial: ArticleCard[];
}

export interface ProfileRecommendations {
  customer_id: string;
  personalized: boolean;
  historial: ArticleCard[];
  recomendaciones_cluster?: ArticleCard[];
  recomendaciones_xgboost?: ArticleCard[];
  recomendaciones_populares?: ArticleCard[];
}

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, init);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `Error ${res.status} al llamar a ${path}`);
  }
  return res.json();
}

function postJson<T>(path: string, body?: unknown): Promise<T> {
  return fetchJson(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
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

export function getMlflowRuns(): Promise<MlflowRunsResponse> {
  return fetchJson("/api/mlflow/runs");
}

export function searchArticles(query: string, limit = 30): Promise<ArticleCard[]> {
  const params = new URLSearchParams();
  if (query) params.set("query", query);
  params.set("limit", String(limit));
  return fetchJson(`/api/articles?${params.toString()}`);
}

export function createProfile(payload: {
  display_name: string;
  age?: number | null;
  club_member_status?: string | null;
}): Promise<Profile> {
  return postJson("/api/profile", payload);
}

export function addPurchase(customerId: string, articleId: number): Promise<Profile> {
  return postJson(`/api/profile/${encodeURIComponent(customerId)}/purchases`, { article_id: articleId });
}

export function resetProfile(customerId: string): Promise<Profile> {
  return postJson(`/api/profile/${encodeURIComponent(customerId)}/reset`);
}

export function getProfileRecommendations(customerId: string): Promise<ProfileRecommendations> {
  return fetchJson(`/api/profile/${encodeURIComponent(customerId)}/recommendations`);
}

import { useEffect, useState } from "react";
import {
  addPurchase,
  createProfile,
  getProfileRecommendations,
  resetProfile,
  searchArticles,
  type ArticleCard,
  type Profile,
  type ProfileRecommendations,
} from "../api";
import { ProductList } from "./ProductList";

const STORAGE_KEY = "hym_profile";

export default function ProfilePanel() {
  const [profile, setProfile] = useState<Profile | null>(null);
  const [displayName, setDisplayName] = useState("");
  const [age, setAge] = useState("");
  const [clubStatus, setClubStatus] = useState("");
  const [formError, setFormError] = useState<string | null>(null);

  const [query, setQuery] = useState("");
  const [catalog, setCatalog] = useState<ArticleCard[]>([]);
  const [recommendations, setRecommendations] = useState<ProfileRecommendations | null>(null);
  const [loadingRecs, setLoadingRecs] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      try {
        setProfile(JSON.parse(stored));
      } catch {
        localStorage.removeItem(STORAGE_KEY);
      }
    }
  }, []);

  useEffect(() => {
    if (!profile) return;
    const handle = setTimeout(() => {
      searchArticles(query).then(setCatalog).catch(() => setCatalog([]));
    }, 250);
    return () => clearTimeout(handle);
  }, [query, profile]);

  const persistProfile = (p: Profile) => {
    setProfile(p);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(p));
  };

  const handleRegister = (e: React.FormEvent) => {
    e.preventDefault();
    if (!displayName.trim()) return;
    setFormError(null);
    createProfile({
      display_name: displayName.trim(),
      age: age ? Number(age) : null,
      club_member_status: clubStatus || null,
    })
      .then(persistProfile)
      .catch((err) => setFormError(err.message));
  };

  const handleLogout = () => {
    localStorage.removeItem(STORAGE_KEY);
    setProfile(null);
    setRecommendations(null);
    setCatalog([]);
    setQuery("");
  };

  const handleBuy = (article: ArticleCard) => {
    if (!profile) return;
    addPurchase(profile.customer_id, article.article_id)
      .then(persistProfile)
      .catch((err) => setError(err.message));
  };

  const handleReset = () => {
    if (!profile) return;
    resetProfile(profile.customer_id).then(persistProfile).catch((err) => setError(err.message));
  };

  const handleShowRecommendations = () => {
    if (!profile) return;
    setLoadingRecs(true);
    setError(null);
    getProfileRecommendations(profile.customer_id)
      .then((data) => {
        setRecommendations(data);
        setLoadingRecs(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoadingRecs(false);
      });
  };

  if (!profile) {
    return (
      <section>
        <h2>Crear mi perfil</h2>
        <p className="muted">
          Regístrate con un nombre cualquiera (no hay contraseña, es una demo). Empezarás con un
          perfil vacío: simula compras en el catálogo para recibir recomendaciones personalizadas.
        </p>
        <form className="profile-form" onSubmit={handleRegister}>
          <input
            className="search-input"
            type="text"
            placeholder="Tu nombre…"
            value={displayName}
            onChange={(e) => setDisplayName(e.target.value)}
          />
          <input
            className="search-input"
            type="number"
            placeholder="Edad (opcional)"
            value={age}
            onChange={(e) => setAge(e.target.value)}
          />
          <select className="search-input" value={clubStatus} onChange={(e) => setClubStatus(e.target.value)}>
            <option value="">Estado de socio (opcional)</option>
            <option value="ACTIVE">ACTIVE</option>
            <option value="PRE-CREATE">PRE-CREATE</option>
            <option value="LEFT CLUB">LEFT CLUB</option>
          </select>
          <button type="submit" className="refresh-button home-search-button">
            Iniciar sesión
          </button>
        </form>
        {formError && <p className="error">{formError}</p>}
      </section>
    );
  }

  return (
    <section>
      <div className="panel-toolbar">
        <h2>Hola, {profile.display_name}</h2>
        <button className="refresh-button" onClick={handleLogout}>
          Cambiar de perfil
        </button>
      </div>

      <input
        className="search-input"
        type="text"
        placeholder="Buscar en el catálogo…"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
      />

      <ProductList
        title="Catálogo"
        articles={catalog}
        actionLabel="Añadir a mis compras"
        onAction={handleBuy}
      />

      <div className="customer-detail">
        <div className="panel-toolbar">
          <h3>Mis compras simuladas ({profile.historial.length})</h3>
          {profile.historial.length > 0 && (
            <button className="refresh-button" onClick={handleReset}>
              Vaciar
            </button>
          )}
        </div>
        <ProductList title="" articles={profile.historial} />

        <div className="panel-toolbar">
          <button className="refresh-button home-search-button" onClick={handleShowRecommendations}>
            Ver mis recomendaciones
          </button>
        </div>
        {profile.historial.length === 0 && (
          <p className="muted">
            Aún no has añadido compras: si pides recomendaciones ahora verás una lista genérica de
            los artículos más vendidos, no personalizada.
          </p>
        )}

        {loadingRecs && <p className="muted">Calculando recomendaciones…</p>}
        {error && <p className="error">{error}</p>}

        {recommendations && !loadingRecs && (
          recommendations.personalized ? (
            <div className="recommendation-columns">
              <ProductList title="Modelo Cluster (KMeans)" articles={recommendations.recomendaciones_cluster ?? []} />
              <ProductList title="Modelo XGBoost" articles={recommendations.recomendaciones_xgboost ?? []} />
            </div>
          ) : (
            <ProductList title="Más vendidos (recomendación genérica)" articles={recommendations.recomendaciones_populares ?? []} />
          )
        )}
      </div>
    </section>
  );
}

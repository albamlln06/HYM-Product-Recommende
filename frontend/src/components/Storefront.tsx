import { useEffect, useMemo, useState } from "react";
import {
  addPurchase,
  createProfile,
  getProfileRecommendations,
  searchArticles,
  type ArticleCard,
  type Profile,
  type ProfileRecommendations,
} from "../api";
import { StoreNav } from "./StoreNav";
import { StoreProductCard, groupArticlesByProduct } from "./StoreProductCard";
import { ProductList } from "./ProductList";

const PROFILE_KEY = "hym_profile";
const CART_KEY = "hym_cart";

interface CartLine {
  article: ArticleCard;
  qty: number;
}

function loadFromStorage<T>(key: string, fallback: T): T {
  const raw = localStorage.getItem(key);
  if (!raw) return fallback;
  try {
    return JSON.parse(raw) as T;
  } catch {
    localStorage.removeItem(key);
    return fallback;
  }
}

export default function Storefront({ onSwitchToPanel }: { onSwitchToPanel: () => void }) {
  const [profile, setProfile] = useState<Profile | null>(() => loadFromStorage(PROFILE_KEY, null));
  const [cart, setCart] = useState<CartLine[]>(() => loadFromStorage(CART_KEY, []));
  const [cartOpen, setCartOpen] = useState(false);

  const [allCategories, setAllCategories] = useState<string[]>([]);
  const [category, setCategory] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [catalog, setCatalog] = useState<ArticleCard[]>([]);

  const [displayName, setDisplayName] = useState("");
  const [age, setAge] = useState("");
  const [clubStatus, setClubStatus] = useState("");
  const [checkoutError, setCheckoutError] = useState<string | null>(null);
  const [checkoutLoading, setCheckoutLoading] = useState(false);
  const [recommendations, setRecommendations] = useState<ProfileRecommendations | null>(null);

  useEffect(() => {
    searchArticles("", 60).then((articles) => {
      const cats = [...new Set(articles.map((a) => a.category))].sort();
      setAllCategories(cats);
    });
  }, []);

  useEffect(() => {
    const handle = setTimeout(() => {
      searchArticles(query, 60).then(setCatalog).catch(() => setCatalog([]));
    }, 250);
    return () => clearTimeout(handle);
  }, [query]);

  useEffect(() => {
    localStorage.setItem(CART_KEY, JSON.stringify(cart));
  }, [cart]);

  const filteredCatalog = useMemo(
    () => (category ? catalog.filter((a) => a.category === category) : catalog),
    [catalog, category],
  );
  const productGroups = useMemo(() => groupArticlesByProduct(filteredCatalog), [filteredCatalog]);

  const cartCount = cart.reduce((sum, line) => sum + line.qty, 0);
  const cartTotal = cart.reduce((sum, line) => sum + line.article.avg_price * line.qty, 0);

  const persistProfile = (p: Profile) => {
    setProfile(p);
    localStorage.setItem(PROFILE_KEY, JSON.stringify(p));
  };

  const handleAddToCart = (article: ArticleCard) => {
    setCart((prev) => {
      const existing = prev.find((line) => line.article.article_id === article.article_id);
      if (existing) {
        return prev.map((line) =>
          line.article.article_id === article.article_id ? { ...line, qty: line.qty + 1 } : line,
        );
      }
      return [...prev, { article, qty: 1 }];
    });
    setCartOpen(true);
  };

  const changeQty = (articleId: number, delta: number) => {
    setCart((prev) =>
      prev
        .map((line) => (line.article.article_id === articleId ? { ...line, qty: line.qty + delta } : line))
        .filter((line) => line.qty > 0),
    );
  };

  const removeLine = (articleId: number) => {
    setCart((prev) => prev.filter((line) => line.article.article_id !== articleId));
  };

  const ensureProfile = async (): Promise<Profile> => {
    if (profile) return profile;
    if (!displayName.trim()) {
      throw new Error("Introduce un nombre para continuar");
    }
    const created = await createProfile({
      display_name: displayName.trim(),
      age: age ? Number(age) : null,
      club_member_status: clubStatus || null,
    });
    persistProfile(created);
    return created;
  };

  const handleCheckout = async () => {
    setCheckoutError(null);
    setCheckoutLoading(true);
    try {
      const currentProfile = await ensureProfile();
      let latestProfile = currentProfile;
      for (const line of cart) {
        for (let i = 0; i < line.qty; i++) {
          latestProfile = await addPurchase(currentProfile.customer_id, line.article.article_id);
        }
      }
      persistProfile(latestProfile);
      setCart([]);
      const recs = await getProfileRecommendations(currentProfile.customer_id);
      setRecommendations(recs);
    } catch (err) {
      setCheckoutError(err instanceof Error ? err.message : String(err));
    } finally {
      setCheckoutLoading(false);
    }
  };

  const handleSkipToRecommendations = async () => {
    setCheckoutError(null);
    setCheckoutLoading(true);
    try {
      const currentProfile = await ensureProfile();
      const recs = await getProfileRecommendations(currentProfile.customer_id);
      setRecommendations(recs);
    } catch (err) {
      setCheckoutError(err instanceof Error ? err.message : String(err));
    } finally {
      setCheckoutLoading(false);
    }
  };

  const closeCart = () => {
    setCartOpen(false);
    setRecommendations(null);
    setCheckoutError(null);
  };

  return (
    <div className="storefront">
      <StoreNav
        categories={allCategories}
        activeCategory={category}
        onCategoryChange={setCategory}
        query={query}
        onQueryChange={setQuery}
        profileName={profile?.display_name ?? null}
        cartCount={cartCount}
        onCartClick={() => setCartOpen(true)}
        onSwitchToPanel={onSwitchToPanel}
      />

      <main className="store-body">
        <h2>{category ?? "Todos los productos"}</h2>
        {productGroups.length === 0 ? (
          <p className="muted">Sin resultados para este catálogo.</p>
        ) : (
          <ul className="store-product-grid">
            {productGroups.map((group) => (
              <StoreProductCard key={group.product_code} group={group} onAddToCart={handleAddToCart} />
            ))}
          </ul>
        )}
      </main>

      {cartOpen && (
        <div className="cart-overlay" onClick={closeCart}>
          <aside className="cart-drawer" onClick={(e) => e.stopPropagation()}>
            <div className="panel-toolbar">
              <h3>Tu carrito</h3>
              <button className="refresh-button" onClick={closeCart}>Cerrar</button>
            </div>

            {recommendations ? (
              <div className="cart-recommendations">
                <p className="muted">
                  {recommendations.personalized
                    ? "¡Gracias por tu compra! Estas son tus recomendaciones:"
                    : "Aún no tienes compras: aquí tienes los artículos más vendidos."}
                </p>
                {recommendations.personalized ? (
                  <ProductList title="Recomendado para ti" articles={recommendations.recomendaciones_cluster ?? []} />
                ) : (
                  <ProductList title="Más vendidos" articles={recommendations.recomendaciones_populares ?? []} />
                )}
                <button className="refresh-button home-search-button" onClick={closeCart}>
                  Seguir comprando
                </button>
              </div>
            ) : (
              <>
                {cart.length === 0 ? (
                  <p className="muted">Tu carrito está vacío.</p>
                ) : (
                  <ul className="cart-lines">
                    {cart.map((line) => (
                      <li key={line.article.article_id} className="cart-line">
                        <div className="cart-line-info">
                          <span className="product-name">{line.article.prod_name}</span>
                          <span className="muted">{line.article.colour}</span>
                          <span className="tabular">{line.article.avg_price.toFixed(4)}</span>
                        </div>
                        <div className="cart-line-actions">
                          <button className="qty-button" onClick={() => changeQty(line.article.article_id, -1)}>−</button>
                          <span className="tabular">{line.qty}</span>
                          <button className="qty-button" onClick={() => changeQty(line.article.article_id, 1)}>+</button>
                          <button className="refresh-button" onClick={() => removeLine(line.article.article_id)}>
                            Quitar
                          </button>
                        </div>
                      </li>
                    ))}
                  </ul>
                )}

                {cart.length > 0 && (
                  <p className="cart-total tabular">Total: {cartTotal.toFixed(4)} (precio norm.)</p>
                )}

                {!profile && (
                  <form className="profile-form" onSubmit={(e) => e.preventDefault()}>
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
                  </form>
                )}

                {checkoutError && <p className="error">{checkoutError}</p>}

                <div className="panel-toolbar">
                  <button
                    className="refresh-button home-search-button"
                    onClick={handleCheckout}
                    disabled={cart.length === 0 || checkoutLoading}
                  >
                    {checkoutLoading ? "Procesando…" : "Finalizar compra"}
                  </button>
                  <button className="refresh-button" onClick={handleSkipToRecommendations} disabled={checkoutLoading}>
                    Ver recomendaciones sin comprar
                  </button>
                </div>
              </>
            )}
          </aside>
        </div>
      )}
    </div>
  );
}

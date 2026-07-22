export function StoreNav({
  categories,
  activeCategory,
  onCategoryChange,
  query,
  onQueryChange,
  profileName,
  cartCount,
  onCartClick,
  onSwitchToPanel,
}: {
  categories: string[];
  activeCategory: string | null;
  onCategoryChange: (category: string | null) => void;
  query: string;
  onQueryChange: (query: string) => void;
  profileName: string | null;
  cartCount: number;
  onCartClick: () => void;
  onSwitchToPanel: () => void;
}) {
  return (
    <header className="store-shell">
      <div className="store-topbar">
        Proyecto académico · Recomendaciones con Machine Learning en tiempo real
      </div>

      <div className="store-nav">
        <span className="brand">
          <span className="brand-mark" aria-hidden="true">S</span>
          <span className="brand-word">
            STUDIO<span className="brand-dot">.</span>
          </span>
        </span>

        <nav className="store-nav-links">
          <button
            className={activeCategory === null ? "store-nav-link active" : "store-nav-link"}
            onClick={() => onCategoryChange(null)}
          >
            Todos
          </button>
          {categories.map((cat) => (
            <button
              key={cat}
              className={activeCategory === cat ? "store-nav-link active" : "store-nav-link"}
              onClick={() => onCategoryChange(cat)}
            >
              {cat}
            </button>
          ))}
        </nav>

        <div className="store-nav-actions">
          <input
            className="store-search-input"
            type="text"
            placeholder="Buscar…"
            value={query}
            onChange={(e) => onQueryChange(e.target.value)}
            aria-label="Buscar en el catálogo"
          />
          <span className="store-account">{profileName ? `Hola, ${profileName}` : "Iniciar sesión"}</span>
          <button type="button" className="cart-button" onClick={onCartClick} aria-label="Ver carrito">
            🛒
            {cartCount > 0 && <span className="cart-badge">{cartCount}</span>}
          </button>
          <button type="button" className="store-switch-link" onClick={onSwitchToPanel}>
            Panel de análisis →
          </button>
        </div>
      </div>
    </header>
  );
}

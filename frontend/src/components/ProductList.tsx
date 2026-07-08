import type { ArticleCard } from "../api";

export function ProductCard({ article }: { article: ArticleCard }) {
  return (
    <li className="product-card">
      <span className="product-name">{article.prod_name}</span>
      <span className="muted">{article.product_type_name} · {article.section_name}</span>
      <span className="tabular">{article.avg_price.toFixed(4)} (precio norm.)</span>
    </li>
  );
}

export function ProductList({ title, articles }: { title: string; articles: ArticleCard[] }) {
  return (
    <div className="product-column">
      <h3>{title}</h3>
      {articles.length === 0 ? (
        <p className="muted">Sin recomendaciones disponibles.</p>
      ) : (
        <ul className="product-list">
          {articles.map((a) => (
            <ProductCard key={a.article_id} article={a} />
          ))}
        </ul>
      )}
    </div>
  );
}

import type { ArticleCard } from "../api";

interface ProductActionProps {
  actionLabel?: string;
  onAction?: (article: ArticleCard) => void;
}

export function ProductCard({ article, actionLabel, onAction }: { article: ArticleCard } & ProductActionProps) {
  return (
    <li className="product-card">
      <div className="product-thumb" aria-hidden="true">
        {article.product_type_name}
      </div>
      <span className="product-name">{article.prod_name}</span>
      <span className="muted">{article.section_name}</span>
      <span className="tabular">{article.avg_price.toFixed(4)} (precio norm.)</span>
      {onAction && (
        <button type="button" className="product-action" onClick={() => onAction(article)}>
          {actionLabel ?? "Añadir"}
        </button>
      )}
    </li>
  );
}

export function ProductList({
  title,
  articles,
  actionLabel,
  onAction,
}: { title: string; articles: ArticleCard[] } & ProductActionProps) {
  return (
    <div className="product-column">
      <h3>{title}</h3>
      {articles.length === 0 ? (
        <p className="muted">Sin recomendaciones disponibles.</p>
      ) : (
        <ul className="product-list">
          {articles.map((a) => (
            <ProductCard key={a.article_id} article={a} actionLabel={actionLabel} onAction={onAction} />
          ))}
        </ul>
      )}
    </div>
  );
}

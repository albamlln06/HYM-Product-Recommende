import { useState } from "react";
import type { ArticleCard } from "../api";

export interface ProductGroup {
  product_code: number;
  prod_name: string;
  product_type_name: string;
  variants: ArticleCard[];
}

export function groupArticlesByProduct(articles: ArticleCard[]): ProductGroup[] {
  const groups = new Map<number, ProductGroup>();

  for (const article of articles) {
    let group = groups.get(article.product_code);
    if (!group) {
      group = {
        product_code: article.product_code,
        prod_name: article.prod_name,
        product_type_name: article.product_type_name,
        variants: [],
      };
      groups.set(article.product_code, group);
    }
    if (!group.variants.some((v) => v.colour === article.colour)) {
      group.variants.push(article);
    }
  }

  return [...groups.values()];
}

export function StoreProductCard({
  group,
  onAddToCart,
}: {
  group: ProductGroup;
  onAddToCart: (article: ArticleCard) => void;
}) {
  const [selectedIndex, setSelectedIndex] = useState(0);
  const selected = group.variants[selectedIndex];

  return (
    <li className="store-product-card">
      <div className="product-thumb" aria-hidden="true">
        {group.product_type_name}
      </div>
      <span className="product-name">{group.prod_name}</span>
      <select
        className="variant-select"
        value={selectedIndex}
        onChange={(e) => setSelectedIndex(Number(e.target.value))}
      >
        {group.variants.map((variant, i) => (
          <option key={variant.article_id} value={i}>
            {variant.colour}
          </option>
        ))}
      </select>
      <span className="tabular store-product-price">{selected.avg_price.toFixed(4)} (precio norm.)</span>
      <button type="button" className="add-to-cart-button" onClick={() => onAddToCart(selected)}>
        Añadir al carrito
      </button>
    </li>
  );
}

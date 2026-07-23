export const FAMILY_COLORS: Record<string, string> = {
  XGBoost: "var(--series-1)",
  Cluster: "var(--series-2)",
  Popular: "var(--series-3)",
  Random: "var(--series-4)",
  Optuna: "var(--series-6)",
  Otro: "var(--series-5)",
};

export const FAMILY_ORDER = ["XGBoost", "Cluster", "Popular", "Random", "Optuna", "Otro"];

// Rampa de tono único (claro -> oscuro) para distinguir, DENTRO de los runs
// de Optuna, con qué n_customers/candidate_pool_size se corrió cada trial:
// una búsqueda con una muestra distinta no es comparable a otra, así que
// conviene verlo de un vistazo en el color en vez de solo en el tooltip.
const OPTUNA_SHADES = ["var(--palette-amber-1)", "var(--palette-amber-2)", "var(--palette-amber-3)", "var(--palette-amber-4)"];

// Hash estable (no depende de qué otras combinaciones haya en pantalla, así
// una combinación siempre sale con el mismo color aunque cambie el resto de
// runs cargados) para elegir el tono dentro de OPTUNA_SHADES.
function hashOptunaConfig(nCustomers: string, poolSize: string): number {
  const s = `${nCustomers}:${poolSize}`;
  let h = 0;
  for (let i = 0; i < s.length; i++) {
    h = (h * 31 + s.charCodeAt(i)) | 0;
  }
  return Math.abs(h);
}

export interface RunGroup {
  key: string;
  label: string;
  color: string;
}

/**
 * Agrupa un run para pintarlo/etiquetarlo: los runs normales usan su family
 * tal cual (color fijo de FAMILY_COLORS); los runs de Optuna se subdividen
 * por la combinación de n_customers/candidate_pool_size con la que se
 * corrió la búsqueda (params logueados por optuna_search.py), cada una con
 * su propio tono de la rampa ámbar y su propia etiqueta de leyenda.
 */
export function runGroup(run: { family: string; params: Record<string, string> }): RunGroup {
  if (run.family !== "Optuna") {
    return { key: run.family, label: run.family, color: FAMILY_COLORS[run.family] ?? FAMILY_COLORS.Otro };
  }

  const nCustomers = run.params.n_customers;
  const poolSize = run.params.candidate_pool_size;
  if (!nCustomers || !poolSize) {
    return { key: "Optuna", label: "Optuna", color: FAMILY_COLORS.Optuna };
  }

  const shade = OPTUNA_SHADES[hashOptunaConfig(nCustomers, poolSize) % OPTUNA_SHADES.length];
  return {
    key: `Optuna:${nCustomers}:${poolSize}`,
    label: `Optuna · ${nCustomers} clientes / ${poolSize} candidatos`,
    color: shade,
  };
}

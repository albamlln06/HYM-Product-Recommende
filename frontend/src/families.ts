export const FAMILY_COLORS: Record<string, string> = {
  XGBoost: "var(--series-1)",
  Cluster: "var(--series-2)",
  Popular: "var(--series-3)",
  Random: "var(--series-4)",
  Optuna: "var(--series-6)",
  Otro: "var(--series-5)",
};

// Rampa de tono único (claro -> oscuro) para distinguir, DENTRO de los runs
// de Optuna, qué estudio corrió cada trial: dos estudios pueden usar
// candidate pools de evaluación distintos (use_hybrid_candidates) y por
// tanto no son comparables entre sí, así que conviene verlo de un vistazo en
// el color en vez de solo en el tooltip.
const OPTUNA_SHADES = ["var(--palette-amber-1)", "var(--palette-amber-2)", "var(--palette-amber-3)", "var(--palette-amber-4)"];

// Hash estable (no depende de qué otros estudios haya en pantalla, así un
// estudio siempre sale con el mismo color aunque cambie el resto de runs
// cargados) para elegir el tono dentro de OPTUNA_SHADES.
function hashOptunaConfig(key: string): number {
  let h = 0;
  for (let i = 0; i < key.length; i++) {
    h = (h * 31 + key.charCodeAt(i)) | 0;
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
 * por optuna_study (params logueados por optuna_search.py), cada uno con su
 * propio tono de la rampa ámbar y su propia etiqueta de leyenda. Se agrupa
 * por estudio (no por n_customers/candidate_pool_size) porque lo que de
 * verdad hace a dos estudios NO comparables es si tunearon con el mismo
 * candidate pool de evaluación (use_hybrid_candidates) que train.py — la
 * etiqueta lo deja explícito.
 */
export function runGroup(run: { family: string; params: Record<string, string> }): RunGroup {
  if (run.family !== "Optuna") {
    return { key: run.family, label: run.family, color: FAMILY_COLORS[run.family] ?? FAMILY_COLORS.Otro };
  }

  const study = run.params.optuna_study;
  if (!study) {
    return { key: "Optuna", label: "Optuna", color: FAMILY_COLORS.Optuna };
  }

  const nCustomers = run.params.n_customers ?? "?";
  const hybridCandidates = run.params.use_hybrid_candidates === "True";
  const candidatesTag = hybridCandidates ? "candidatos híbridos" : "candidatos por popularidad";
  const shade = OPTUNA_SHADES[hashOptunaConfig(study) % OPTUNA_SHADES.length];
  return {
    key: `Optuna:${study}`,
    label: `Optuna · ${study} (${nCustomers} clientes, ${candidatesTag})`,
    color: shade,
  };
}

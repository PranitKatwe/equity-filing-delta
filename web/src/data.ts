// Study results embedded from the honest full-S&P-500 run (see README).

export const QUINT = {
  labels: ["Q1 (fewest)", "Q2", "Q3", "Q4", "Q5 (most)"],
  insample: [-0.484, 0.08, 0.081, 0.375, 0.197],
  holdout: [0.253, 0.704, 0.193, 0.538, -0.017],
};

export type Coef = { name: string; coef: number; p: number };

export const COEF: Coef[] = [
  { name: "net_added", coef: -0.154, p: 0.147 },
  { name: "Δ tone negative", coef: 0.025, p: 0.821 },
  { name: "Δ tone uncertainty", coef: -0.004, p: 0.972 },
  { name: "doc_similarity", coef: 0.117, p: 0.34 },
];

export const TILES = [
  { n: "4,705", l: "filing events (480 names)" },
  { n: "+0.07%", l: "net-of-cost spread (p = 0.51)" },
  { n: "0.03", l: "holdout p-value, best window" },
  { n: "clean", l: "placebo / no lookahead" },
];

// Shared types, formatters, and typed API helpers for the two grounded LLM features.

export type CompanyEvent = {
  key: string;
  filing_date: string;
  net_added: number;
  car_0_5: number | null;
};
export type Company = { sector: string; events: CompanyEvent[] };
export type PanelIndex = Record<string, Company>;

export type NarrateResponse = {
  event: string;
  memo: string;
  facts: {
    ticker: string;
    sector: string;
    accession: string;
    filing_date: string;
    net_added: number;
    car_0_5: number | null;
    car_0_21: number | null;
  };
  cached?: boolean;
};

export type AskResponse = {
  company: string;
  question: string;
  answer: string;
  accession: string;
  filing_date: string;
  counts: { added: number; removed: number; reworded: number };
};

/** Fraction -> signed percent, e.g. 0.0051 -> "+0.51%". */
export const pct = (v: number | null | undefined): string =>
  v == null ? "n/a" : `${v >= 0 ? "+" : ""}${(v * 100).toFixed(2)}%`;

/** GET JSON; on a non-OK response, throw an Error carrying the API's message. */
export async function getJSON<T>(url: string): Promise<T> {
  let r: Response;
  try {
    r = await fetch(url);
  } catch {
    throw new Error(
      "Couldn't reach the API. Confirm you're on the Vercel deployment, then retry.",
    );
  }
  let body: unknown = null;
  try {
    body = await r.json();
  } catch {
    /* non-JSON (gateway/timeout page) */
  }
  if (!r.ok) {
    const msg = (body as { error?: string } | null)?.error;
    if (msg) throw new Error(msg);
    if (r.status === 504)
      throw new Error(
        "The model took too long this time (timeout). Try again. Successful results are cached, so a retry is usually fast.",
      );
    throw new Error(`Request failed (HTTP ${r.status}). Give it a few seconds and try again.`);
  }
  return body as T;
}

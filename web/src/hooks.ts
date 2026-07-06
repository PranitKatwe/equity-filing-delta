import { useEffect, useState } from "react";
import type { PanelIndex } from "./lib";

type State = { index: PanelIndex | null; error: string | null };

/** Load the lightweight company/event search index once (served statically). */
export function useCompanyIndex(): State {
  const [state, setState] = useState<State>({ index: null, error: null });
  useEffect(() => {
    let alive = true;
    fetch("/panel_index.json")
      .then((r) => r.json())
      .then((idx: PanelIndex) => alive && setState({ index: idx, error: null }))
      .catch(() => alive && setState({ index: null, error: "Couldn't load the company index." }));
    return () => {
      alive = false;
    };
  }, []);
  return state;
}

export const INPUT =
  "rounded-lg border border-line bg-surface px-3 py-2.5 text-ink outline-none transition focus:border-holdout focus:ring-2 focus:ring-holdout/30 disabled:opacity-50";

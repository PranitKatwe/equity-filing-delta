import { useMemo, useState } from "react";
import { getJSON, pct, type NarrateResponse, type PanelIndex } from "../lib";
import { INPUT, useCompanyIndex } from "../hooks";
import { Button, Card, Facts, Note, Output } from "./ui";

export function Narrator() {
  const { index, error: indexError } = useCompanyIndex();
  const [company, setCompany] = useState("");
  const [eventKey, setEventKey] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<NarrateResponse | null>(null);
  const [note, setNote] = useState("");

  const tickers = useMemo(() => (index ? Object.keys(index).sort() : []), [index]);
  const chosen = index?.[company.trim().toUpperCase()];

  // Default the year selection to the company's most recent filing.
  const events = chosen?.events ?? [];
  const key = eventKey && events.some((e) => e.key === eventKey) ? eventKey : events[0]?.key ?? "";

  async function run() {
    if (!key) return;
    setLoading(true);
    setNote("");
    setResult(null);
    try {
      setResult(await getJSON<NarrateResponse>(`/api/narrate?event=${encodeURIComponent(key)}`));
    } catch (e) {
      setNote((e as Error).message);
    } finally {
      setLoading(false);
    }
  }

  const status = indexError
    ? indexError
    : !index
      ? "Loading companies…"
      : chosen
        ? `${company.trim().toUpperCase()} · ${chosen.sector} · ${events.length} filing${events.length > 1 ? "s" : ""}`
        : `${tickers.length} companies loaded — type a ticker to begin.`;

  return (
    <Card className="mt-4 p-5">
      <CompanyList id="colist" tickers={tickers} index={index} />
      <div className="flex flex-wrap items-center gap-2.5">
        <input
          list="colist"
          value={company}
          onChange={(e) => {
            setCompany(e.target.value);
            setEventKey("");
          }}
          placeholder="Company ticker — e.g. AAPL"
          autoComplete="off"
          aria-label="Search a company by ticker"
          className={`${INPUT} min-w-[200px]`}
        />
        <select
          value={key}
          onChange={(e) => setEventKey(e.target.value)}
          disabled={!chosen}
          aria-label="Choose a filing year"
          className={`${INPUT} max-w-full`}
        >
          {events.map((ev) => (
            <option key={ev.key} value={ev.key}>
              {`10-K filed ${ev.filing_date} · net-added ${ev.net_added} · CAR[0,+5] ${pct(ev.car_0_5)}`}
            </option>
          ))}
        </select>
        <Button onClick={run} disabled={!chosen} loading={loading}>
          {loading ? "Generating…" : "Generate memo"}
        </Button>
      </div>

      {result && (
        <Output>
          <p>{result.memo}</p>
          <Facts>
            Source: accession {result.facts.accession} · net-added {result.facts.net_added} ·
            CAR[0,+5] {pct(result.facts.car_0_5)} · CAR[0,+21] {pct(result.facts.car_0_21)}
            {result.cached ? " · cached" : ""}
          </Facts>
        </Output>
      )}
      <Note>{note || status}</Note>
    </Card>
  );
}

function CompanyList({
  id,
  tickers,
  index,
}: {
  id: string;
  tickers: string[];
  index: PanelIndex | null;
}) {
  return (
    <datalist id={id}>
      {tickers.map((t) => (
        <option key={t} value={t}>
          {index?.[t]?.sector}
        </option>
      ))}
    </datalist>
  );
}

import { useMemo, useState } from "react";
import { getJSON, type AskResponse } from "../lib";
import { INPUT, useCompanyIndex } from "../hooks";
import { Button, Card, Facts, Note, Output } from "./ui";

export function FilingQA() {
  const { index } = useCompanyIndex();
  const [company, setCompany] = useState("");
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AskResponse | null>(null);
  const [note, setNote] = useState("");

  const tickers = useMemo(() => (index ? Object.keys(index).sort() : []), [index]);

  async function ask() {
    const c = company.trim().toUpperCase();
    const q = question.trim();
    if (!c || !q) {
      setNote("Enter a company ticker and a question.");
      return;
    }
    setLoading(true);
    setNote("");
    setResult(null);
    try {
      setResult(
        await getJSON<AskResponse>(
          `/api/ask?company=${encodeURIComponent(c)}&q=${encodeURIComponent(q)}`,
        ),
      );
    } catch (e) {
      setNote((e as Error).message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <Card className="mt-4 p-5">
      <datalist id="qacolist">
        {tickers.map((t) => (
          <option key={t} value={t}>
            {index?.[t]?.sector}
          </option>
        ))}
      </datalist>
      <div className="flex flex-wrap items-center gap-2.5">
        <input
          list="qacolist"
          value={company}
          onChange={(e) => setCompany(e.target.value)}
          placeholder="Company — e.g. BA"
          autoComplete="off"
          aria-label="Company for the question"
          className={`${INPUT} min-w-[160px]`}
        />
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && ask()}
          placeholder="e.g. What new risks were added this year?"
          maxLength={300}
          aria-label="Your question"
          className={`${INPUT} min-w-[220px] flex-1`}
        />
        <Button onClick={ask} loading={loading}>
          {loading ? "Asking…" : "Ask"}
        </Button>
      </div>

      {result && (
        <Output>
          <p className="whitespace-pre-wrap">{result.answer}</p>
          <Facts>
            Grounded in {result.counts.added} added / {result.counts.removed} removed /{" "}
            {result.counts.reworded} reworded sentences · accession {result.accession} (filed{" "}
            {result.filing_date})
          </Facts>
        </Output>
      )}
      <Note>{note}</Note>
    </Card>
  );
}

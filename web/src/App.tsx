import { Section } from "./components/ui";
import { Hero, Verdict, Tiles, WhyCredible, Footer } from "./components/Static";
import { QuintileChart, CoefChart } from "./components/Charts";
import { Narrator } from "./components/Narrator";
import { FilingQA } from "./components/FilingQA";

export default function App() {
  return (
    <div className="mx-auto max-w-content px-6 pb-20 pt-12 font-sans">
      <Hero />
      <Verdict />
      <Tiles />

      <Section title="Return by amount of change — noisy, and it flips">
        <p className="leading-relaxed text-ink-mute">
          Mean 5-day abnormal return (CAR[0,+5]) by net-added-risk quintile, Q1 (fewest changes) → Q5
          (most). If “Lazy Prices” held cleanly, Q5 would sit well below Q1. It doesn't — the pattern
          is non-monotonic and the in-sample and holdout periods disagree.
        </p>
        <QuintileChart />
      </Section>

      <Section title="Every signal, on the holdout — none significant">
        <p className="leading-relaxed text-ink-mute">
          Cross-sectional coefficient on CAR[0,+5] per 1 SD of each signal, controlling for momentum
          and beta, with filing-date-clustered standard errors. Blue = the Lazy-Prices direction; red
          = against it. All p-values are above 0.05 — including the transparent Loughran-McDonald tone
          deltas, a cross-check that the null isn't an artifact of one signal.
        </p>
        <CoefChart />
      </Section>

      <Section title="Why this is the credible outcome">
        <WhyCredible />
      </Section>

      <Section title="Try it: the grounded narrator">
        <p className="leading-relaxed text-ink-mute">
          Search <strong>any of the 480 S&amp;P 500 companies</strong> in the study, pick one of its
          10-K filings, and a server-side model (GLM&nbsp;5.2, free via NVIDIA) writes a short memo
          from the harness's <em>pre-computed</em> numbers only — it never computes a figure and never
          gives advice. It can narrate <em>only</em> real, measured filing events (the client sends
          just an event ID; the numbers are looked up server-side), so a public endpoint can't be
          coaxed into predicting, advising, or inventing anything.
        </p>
        <Narrator />
      </Section>

      <Section title="Ask about the filing changes">
        <p className="leading-relaxed text-ink-mute">
          Ask a plain-English question about a company's <strong>most recent</strong> 10-K risk-factor
          changes. The model answers <em>only</em> from the actual added / removed / reworded sentences
          the harness diffs out of that filing (retrieved server-side, cited) — if the answer isn't in
          those changes, it says so. Descriptive only: it won't predict, value, or advise.
        </p>
        <FilingQA />
      </Section>

      <Footer />
    </div>
  );
}

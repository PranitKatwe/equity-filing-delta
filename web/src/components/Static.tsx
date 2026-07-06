import { TILES } from "../data";
import { Card } from "./ui";

export function Hero() {
  return (
    <header className="pt-4">
      <h1 className="text-3xl font-semibold leading-tight tracking-tight sm:text-4xl">
        Do year-over-year SEC filing changes predict returns?
      </h1>
      <p className="mt-3 text-lg text-ink-soft">
        A point-in-time, no-lookahead event study across the S&amp;P 500, with costs included,
        out-of-sample, and against an honest null.
      </p>
    </header>
  );
}

export function Verdict() {
  return (
    <Card className="mt-7 border-l-4 border-l-warn p-5">
      <p className="leading-relaxed">
        <strong>The calibrated answer: no, the effect is weak and does not survive as a tradeable
        signal.</strong>{" "}
        On the full universe, point-in-time and net of costs, more added risk-factor language shows
        the documented “Lazy Prices” direction only <em>marginally</em> on the holdout
        (CAR[0,+21]&nbsp;p&nbsp;=&nbsp;0.03), flips to the <em>wrong</em> sign in-sample, and produces
        a net-of-cost long-short spread of just <strong>+0.07%</strong> that is statistically
        insignificant (p&nbsp;=&nbsp;0.51). This is a legitimate, credible result.{" "}
        <strong>The value is the harness that says so honestly, not a signal that doesn't hold up.</strong>
      </p>
    </Card>
  );
}

export function Tiles() {
  return (
    <div className="mt-5 grid grid-cols-2 gap-3 lg:grid-cols-4">
      {TILES.map((t) => (
        <Card key={t.l} className="p-4">
          <div className="text-2xl font-semibold tracking-tight">{t.n}</div>
          <div className="mt-0.5 text-[13px] text-ink-mute">{t.l}</div>
        </Card>
      ))}
    </div>
  );
}

export function WhyCredible() {
  const items: [string, string][] = [
    [
      "No lookahead.",
      "Each filing is aligned to the first session tradeable after its EDGAR acceptance instant; every feature uses only data dated ≤ that instant, enforced in code. A pre-event placebo window confirms the signal has no leakage.",
    ],
    [
      "Two or more expected-return models.",
      "Market-adjusted, sector-adjusted, and a market-model β estimated on a pre-event window.",
    ],
    [
      "Costs included.",
      "Every tradeable claim is reported net of a round-trip cost. A gross-only result is fiction.",
    ],
    [
      "Run the holdout once.",
      "Signals and specs were frozen in-sample; the holdout is a single look.",
    ],
  ];
  return (
    <>
      <p className="leading-relaxed">
        A 30-name subsample showed a <em>strong, significant</em> holdout effect. The full 500 washes
        it out, a textbook demonstration of why out-of-sample, full-universe, net-of-cost discipline
        exists. The methodology is the point:
      </p>
      <ul className="mt-4 space-y-2.5">
        {items.map(([h, body]) => (
          <li key={h} className="leading-relaxed">
            <strong>{h}</strong> {body}
          </li>
        ))}
      </ul>
      <p className="mt-4 text-ink-mute">
        Not claimed: <code className="rounded bg-surface-card px-1.5 py-0.5 text-[13px]">net_added</code>{" "}
        is a blunt mechanical count. A diff-grounded LLM classifier sharpens it into “substantive new
        risks vs boilerplate.” Whether that survives is an open, documented next step.
      </p>
    </>
  );
}

export function Footer() {
  return (
    <footer className="mt-16 border-t border-line pt-5 text-[13px] text-ink-mute">
      Reproduce end-to-end from free data:{" "}
      <code className="rounded bg-surface-card px-1.5 py-0.5">build_panel.py → run_study.py</code>.
      Source &amp; methodology:{" "}
      <a
        className="text-holdout hover:underline"
        href="https://github.com/PranitKatwe/equity-filing-delta"
      >
        github.com/PranitKatwe/equity-filing-delta
      </a>
      . Research artifact, descriptive only, not investment advice.
    </footer>
  );
}

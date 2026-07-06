import { COEF, QUINT } from "../data";

const AXIS = "var(--text-secondary)";

function Legend() {
  const items = [
    ["In-sample (pre-2022)", "var(--insample)"],
    ["Holdout (2022+)", "var(--holdout)"],
  ] as const;
  return (
    <div className="mb-2 flex gap-5 text-[13px] text-ink-soft">
      {items.map(([label, color]) => (
        <span key={label} className="inline-flex items-center gap-1.5">
          <span className="inline-block h-3 w-3 rounded-[3px]" style={{ background: color }} />
          {label}
        </span>
      ))}
    </div>
  );
}

function DataTable({ head, rows }: { head: string[]; rows: (string | number)[][] }) {
  return (
    <details className="mt-2">
      <summary className="cursor-pointer text-[13px] text-ink-mute">Data table</summary>
      <div className="overflow-x-auto">
        <table className="mt-2 w-full border-collapse text-sm">
          <thead>
            <tr>
              {head.map((h, i) => (
                <th
                  key={h}
                  className={`border-b border-line px-2.5 py-1.5 ${i === 0 ? "text-left" : "text-right"}`}
                >
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r, ri) => (
              <tr key={ri}>
                {r.map((c, ci) => (
                  <td
                    key={ci}
                    className={`border-b border-line px-2.5 py-1.5 ${ci === 0 ? "text-left" : "text-right"}`}
                  >
                    {c}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </details>
  );
}

export function QuintileChart() {
  const W = 720, H = 300, L = 48, R = 16, T = 16, B = 44;
  const iw = W - L - R, ih = H - T - B;
  const vals = [...QUINT.insample, ...QUINT.holdout];
  const max = Math.max(...vals, 0.1), min = Math.min(...vals, -0.1);
  const y = (v: number) => T + (ih * (max - v)) / (max - min);
  const y0 = y(0);
  const grid = [-0.4, -0.2, 0, 0.2, 0.4, 0.6];
  const groupW = iw / QUINT.labels.length, bw = 26, gap = 8;
  const series = [
    { key: "insample" as const, color: "var(--insample)" },
    { key: "holdout" as const, color: "var(--holdout)" },
  ];

  return (
    <figure className="my-4">
      <Legend />
      <svg
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label="Grouped bar chart of mean abnormal return by net-added quintile, in-sample versus holdout"
        className="block h-auto w-full overflow-visible"
      >
        {grid.map((g) => (
          <g key={g}>
            <line x1={L} x2={W - R} y1={y(g)} y2={y(g)} stroke="var(--border)" strokeWidth={1} />
            <text x={L - 8} y={y(g) + 4} textAnchor="end" fontSize={12} fill={AXIS}>
              {g.toFixed(1)}
            </text>
          </g>
        ))}
        <line x1={L} x2={W - R} y1={y0} y2={y0} stroke="var(--text-muted)" strokeWidth={1} />
        {QUINT.labels.map((lab, i) => {
          const cx = L + groupW * i + groupW / 2;
          return (
            <g key={lab}>
              {series.map((s, j) => {
                const v = QUINT[s.key][i];
                const yv = y(v);
                const x = cx - bw - gap / 2 + j * (bw + gap);
                const top = Math.min(yv, y0);
                const h = Math.max(Math.abs(yv - y0), 1);
                return (
                  <rect
                    key={s.key}
                    x={x}
                    y={top}
                    width={bw}
                    height={h}
                    rx={3}
                    fill={s.color}
                    className="transition-opacity hover:opacity-80"
                  >
                    <title>{`${lab} · ${s.key}: ${v.toFixed(2)}%`}</title>
                  </rect>
                );
              })}
              <text x={cx} y={H - 22} textAnchor="middle" fontSize={12} fill={AXIS}>
                {lab}
              </text>
            </g>
          );
        })}
      </svg>
      <figcaption className="mt-1.5 text-[13px] text-ink-mute">
        Values in %. Bars grow from the zero line; below-zero bars point down.
      </figcaption>
      <DataTable
        head={["Quintile", "In-sample %", "Holdout %"]}
        rows={QUINT.labels.map((l, i) => [
          l,
          QUINT.insample[i].toFixed(3),
          QUINT.holdout[i].toFixed(3),
        ])}
      />
    </figure>
  );
}

export function CoefChart() {
  const W = 720, H = 220, L = 150, R = 60, T = 12, B = 24;
  const iw = W - L - R, ih = H - T - B;
  const m = Math.max(...COEF.map((d) => Math.abs(d.coef)), 0.2);
  const x = (v: number) => L + (iw * (v + m)) / (2 * m);
  const x0 = x(0), rh = ih / COEF.length, bh = 22;
  const grid = [-0.2, -0.1, 0, 0.1, 0.2];

  return (
    <figure className="my-4">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label="Bar chart of holdout signal coefficients, none statistically significant"
        className="block h-auto w-full overflow-visible"
      >
        {grid.map((g) => (
          <g key={g}>
            <line
              x1={x(g)}
              x2={x(g)}
              y1={T}
              y2={T + ih}
              stroke={g === 0 ? "var(--text-muted)" : "var(--border)"}
              strokeWidth={1}
            />
            <text x={x(g)} y={T + ih + 16} textAnchor="middle" fontSize={12} fill={AXIS}>
              {g.toFixed(1)}
            </text>
          </g>
        ))}
        {COEF.map((d, i) => {
          const cy = T + rh * i + rh / 2;
          const xv = x(d.coef);
          const left = Math.min(xv, x0);
          const w = Math.max(Math.abs(xv - x0), 1);
          const color = d.coef < 0 ? "var(--neg)" : "var(--pos)";
          return (
            <g key={d.name}>
              <rect
                x={left}
                y={cy - bh / 2}
                width={w}
                height={bh}
                rx={3}
                fill={color}
                className="transition-opacity hover:opacity-80"
              >
                <title>{`${d.name}: ${d.coef.toFixed(3)}% / SD, p=${d.p}`}</title>
              </rect>
              <text x={L - 12} y={cy + 4} textAnchor="end" fontSize={12} fill="var(--text-primary)">
                {d.name}
              </text>
              <text
                x={xv + (d.coef < 0 ? -6 : 6)}
                y={cy + 4}
                textAnchor={d.coef < 0 ? "end" : "start"}
                fontSize={12}
                fill={AXIS}
              >
                {`p=${d.p.toFixed(2)}`}
              </text>
            </g>
          );
        })}
      </svg>
      <figcaption className="mt-1.5 text-[13px] text-ink-mute">
        Coefficient in % per 1 SD. Labels show p-values.
      </figcaption>
      <DataTable
        head={["Signal", "Coef (% / SD)", "p"]}
        rows={COEF.map((d) => [d.name, d.coef.toFixed(3), d.p.toFixed(3)])}
      />
    </figure>
  );
}

import { FIGURES } from '@/lib/figures';

const W = 360;
const H = 210;
const M = { top: 10, right: 34, bottom: 26, left: 38 };
const PLOT = { w: W - M.left - M.right, h: H - M.top - M.bottom };

/// A linear scale over the plot area, in the axis's own units.
function scale(lo: number, hi: number, size: number, flip = false) {
  return (v: number) => {
    const t = (v - lo) / (hi - lo);
    return flip ? size - t * size : t * size;
  };
}

/// Round tick values over `[lo, hi]`, at most `count` of them, chosen so the
/// labels read as numbers a person would write.
function ticks(lo: number, hi: number, count = 4): number[] {
  const raw = (hi - lo) / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? mag * 10;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) {
    out.push(Number(v.toFixed(6)));
  }
  return out;
}

type Axis = { lo: number; hi: number; label: string; format?: (v: number) => string };

function Chart({
  title,
  params,
  x,
  y,
  children,
}: {
  title: string;
  params: string;
  x: Axis;
  y: Axis;
  children: (sx: (v: number) => number, sy: (v: number) => number) => React.ReactNode;
}) {
  const sx = scale(x.lo, x.hi, PLOT.w);
  const sy = scale(y.lo, y.hi, PLOT.h, true);
  const fx = x.format ?? String;
  const fy = y.format ?? String;

  return (
    <figure className="flex flex-col rounded-xl border border-fd-border bg-fd-card p-5">
      <figcaption className="flex items-baseline justify-between gap-3">
        <span className="text-sm font-semibold">{title}</span>
        <span className="font-mono text-[11px] text-fd-muted-foreground">{params}</span>
      </figcaption>

      <svg viewBox={`0 0 ${W} ${H}`} className="mt-3 w-full" role="img" aria-label={title}>
        <g transform={`translate(${M.left} ${M.top})`}>
          {ticks(y.lo, y.hi).map((v) => (
            <g key={v}>
              <line
                x1={0}
                x2={PLOT.w}
                y1={sy(v)}
                y2={sy(v)}
                className="stroke-current opacity-10"
                strokeWidth={0.6}
              />
              <text
                x={-6}
                y={sy(v) + 2.6}
                textAnchor="end"
                className="fill-current font-mono text-[8px] opacity-45"
              >
                {fy(v)}
              </text>
            </g>
          ))}

          {ticks(x.lo, x.hi).map((v) => (
            <text
              key={v}
              x={sx(v)}
              y={PLOT.h + 11}
              textAnchor="middle"
              className="fill-current font-mono text-[8px] opacity-45"
            >
              {fx(v)}
            </text>
          ))}

          <line
            x1={0}
            x2={PLOT.w}
            y1={PLOT.h}
            y2={PLOT.h}
            className="stroke-current opacity-25"
            strokeWidth={0.8}
          />

          {children(sx, sy)}

          <text
            x={PLOT.w}
            y={PLOT.h + 22}
            textAnchor="end"
            className="fill-current font-mono text-[8px] opacity-45"
          >
            {x.label}
          </text>
          <text
            x={-M.left + 2}
            y={-1}
            className="fill-current font-mono text-[8px] opacity-45"
          >
            {y.label}
          </text>
        </g>
      </svg>
    </figure>
  );
}

/// `ys` sampled over `[x.lo, x.hi]` as an SVG point list.
function line(ys: readonly number[], sx: (v: number) => number, sy: (v: number) => number, xlo: number, xhi: number) {
  const n = ys.length;
  return ys.map((v, i) => `${sx(xlo + (i * (xhi - xlo)) / (n - 1)).toFixed(1)},${sy(v).toFixed(1)}`).join(' ');
}

/// The area between two series, as a closed polygon.
function band(
  upper: readonly number[],
  lower: readonly number[],
  sx: (v: number) => number,
  sy: (v: number) => number,
  xlo: number,
  xhi: number,
) {
  return `${line(upper, sx, sy, xlo, xhi)} ${line(lower, sx, sy, xlo, xhi).split(' ').reverse().join(' ')}`;
}

const { heston, fbm, svi } = FIGURES;

export function Figures() {
  const hAll = [...heston.q05, ...heston.q95];
  const hLo = Math.floor(Math.min(...hAll) / 10) * 10;
  const hHi = Math.ceil(Math.max(...hAll) / 10) * 10;

  const fAll = fbm.flatMap((r) => r.y as readonly number[]);
  const fLo = Math.floor(Math.min(...fAll));
  const fHi = Math.ceil(Math.max(...fAll));

  const sAll = svi.flatMap((r) => r.y as readonly number[]);
  const sLo = Math.floor(Math.min(...sAll) * 20) / 20;
  const sHi = Math.ceil(Math.max(...sAll) * 20) / 20;

  return (
    <section className="mt-24 w-full max-w-5xl">
      <h2 className="text-2xl font-semibold tracking-tight">What comes out</h2>

      <div className="mt-8 grid gap-6 lg:grid-cols-3">
        <Chart
          title="Heston"
          params="κ 2 · ξ 0.3 · ρ −0.7"
          x={{ lo: 0, hi: 1, label: 't', format: (v) => v.toFixed(2) }}
          y={{ lo: hLo, hi: hHi, label: 'S' }}
        >
          {(sx, sy) => (
            <>
              <polygon
                points={band(heston.q95, heston.q05, sx, sy, 0, 1)}
                className="fill-current opacity-[0.08]"
              />
              <polygon
                points={band(heston.q75, heston.q25, sx, sy, 0, 1)}
                className="fill-current opacity-[0.14]"
              />
              {heston.paths.map((p, i) => (
                <polyline
                  key={i}
                  points={line(p, sx, sy, 0, 1)}
                  fill="none"
                  className="stroke-current opacity-55"
                  strokeWidth={0.8}
                />
              ))}
              <text
                x={PLOT.w - 2}
                y={sy(heston.q95[heston.q95.length - 1]) - 3}
                textAnchor="end"
                className="fill-current font-mono text-[8px] opacity-45"
              >
                95%
              </text>
              <text
                x={PLOT.w - 2}
                y={sy(heston.q05[heston.q05.length - 1]) + 8}
                textAnchor="end"
                className="fill-current font-mono text-[8px] opacity-45"
              >
                5%
              </text>
            </>
          )}
        </Chart>

        <Chart
          title="Fractional Brownian motion"
          params="one seed, three H"
          x={{ lo: 0, hi: 1, label: 't', format: (v) => v.toFixed(2) }}
          y={{ lo: fLo, hi: fHi, label: 'B' }}
        >
          {(sx, sy) => (
            <>
              {fbm.map(({ h, y }, i) => (
                <g key={h}>
                  <polyline
                    points={line(y, sx, sy, 0, 1)}
                    fill="none"
                    className="stroke-current"
                    strokeWidth={0.9}
                    strokeOpacity={0.45 + i * 0.28}
                  />
                  <text
                    x={PLOT.w + 3}
                    y={sy(y[y.length - 1]) + 2.6}
                    className="fill-current font-mono text-[8px] opacity-60"
                  >
                    H {h}
                  </text>
                </g>
              ))}
            </>
          )}
        </Chart>

        <Chart
          title="SVI surface"
          params="four maturities"
          x={{ lo: -0.5, hi: 0.5, label: 'log-moneyness', format: (v) => v.toFixed(2) }}
          y={{ lo: sLo, hi: sHi, label: 'σ', format: (v) => v.toFixed(2) }}
        >
          {(sx, sy) => (
            <>
              <line
                x1={sx(0)}
                x2={sx(0)}
                y1={0}
                y2={PLOT.h}
                className="stroke-current opacity-20"
                strokeDasharray="3 3"
                strokeWidth={0.7}
              />
              {svi.map(({ tau, y }, i) => (
                <g key={tau}>
                  <polyline
                    points={line(y, sx, sy, -0.5, 0.5)}
                    fill="none"
                    className="stroke-current"
                    strokeWidth={1.1}
                    strokeOpacity={0.4 + i * 0.2}
                  />
                  <text
                    x={PLOT.w + 3}
                    y={sy(y[y.length - 1]) + 2.6}
                    className="fill-current font-mono text-[8px] opacity-60"
                  >
                    {tau < 1 ? `${tau * 12}M` : `${tau}Y`}
                  </text>
                </g>
              ))}
            </>
          )}
        </Chart>
      </div>
    </section>
  );
}

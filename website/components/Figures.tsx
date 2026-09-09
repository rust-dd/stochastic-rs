import { FBM_PATHS, HESTON_PATHS, SVI_RANGE, SVI_SMILES } from '@/lib/figures';

const W = 320;
const H = 140;

/// A framed plot: a caption, the drawing, and a line saying where the numbers
/// came from. Every figure on this page is the crate's own output rather than
/// an illustration, and the footnote is what makes that checkable.
function Panel({
  title,
  note,
  children,
}: {
  title: string;
  note: string;
  children: React.ReactNode;
}) {
  return (
    <figure className="flex flex-col rounded-xl border border-fd-border bg-fd-card p-5">
      <figcaption className="text-sm font-semibold">{title}</figcaption>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="mt-4 w-full"
        role="img"
        aria-label={title}
        preserveAspectRatio="none"
        height={H}
      >
        {children}
      </svg>
      <p className="mt-4 text-xs leading-relaxed text-fd-muted-foreground">{note}</p>
    </figure>
  );
}

export function Figures() {
  return (
    <section className="mt-24 w-full max-w-5xl">
      <h2 className="text-2xl font-semibold tracking-tight">Real output</h2>
      <p className="mt-2 max-w-2xl text-fd-muted-foreground">
        Three figures, each drawn from numbers the crate produced — generated
        by an example in the repository, thinned and rounded to keep the page
        small, and otherwise untouched.
      </p>

      <div className="mt-8 grid gap-6 lg:grid-cols-3">
        <Panel
          title="Heston, fourteen paths"
          note="κ = 2, θ = 0.04, ξ = 0.3, ρ = −0.7 over one year on a 512-point grid. The price component of a two-component system; the variance rides along beside it."
        >
          {HESTON_PATHS.map((points, i) => (
            <polyline
              key={i}
              points={points}
              fill="none"
              stroke="currentColor"
              strokeWidth={i === 0 ? 1.4 : 0.7}
              strokeOpacity={i === 0 ? 0.95 : 0.32}
              vectorEffect="non-scaling-stroke"
            />
          ))}
        </Panel>

        <Panel
          title="Fractional Brownian motion"
          note="H = 0.3, 0.5 and 0.8 from the top down, same seed, so only the Hurst exponent differs: anti-persistent, Brownian, persistent. Each band is scaled to its own range, because what separates them is roughness rather than size. Davies–Harte embedding, exact rather than approximate."
        >
          {FBM_PATHS.map(([h, points, base], i) => (
            <g key={h}>
              <polyline
                points={points}
                fill="none"
                stroke="currentColor"
                strokeWidth={1}
                strokeOpacity={0.85}
                vectorEffect="non-scaling-stroke"
              />
              {i < FBM_PATHS.length - 1 && (
                <line
                  x1={0}
                  x2={W}
                  y1={base}
                  y2={base}
                  stroke="currentColor"
                  strokeOpacity={0.12}
                  strokeWidth={1}
                  vectorEffect="non-scaling-stroke"
                />
              )}
            </g>
          ))}
        </Panel>

        <Panel
          title="An SVI volatility surface"
          note={`Four maturities from three months to two years, implied volatility against log-moneyness from −0.5 to +0.5, ${SVI_RANGE[0].toFixed(2)} to ${SVI_RANGE[1].toFixed(2)}. Each slice arbitrage-free by construction.`}
        >
          {SVI_SMILES.map(([tau, points], i) => (
            <polyline
              key={tau}
              points={points}
              fill="none"
              stroke="currentColor"
              strokeWidth={1.4}
              strokeOpacity={0.3 + i * 0.22}
              vectorEffect="non-scaling-stroke"
            />
          ))}
        </Panel>
      </div>
    </section>
  );
}

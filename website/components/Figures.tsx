import { BOX, FBM, HESTON, SVI, SVI_AXIS } from '@/lib/figures';

const { w: W, h: H } = BOX;

function Panel({
  title,
  params,
  children,
}: {
  title: string;
  params: string;
  children: React.ReactNode;
}) {
  return (
    <figure className="flex flex-col rounded-xl border border-fd-border bg-fd-card p-5">
      <figcaption className="flex items-baseline justify-between gap-3">
        <span className="text-sm font-semibold">{title}</span>
        <span className="font-mono text-[11px] text-fd-muted-foreground">{params}</span>
      </figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} className="mt-4 w-full" role="img" aria-label={title}>
        {children}
      </svg>
    </figure>
  );
}

/// A tick label inside the plot, in the figure's own units.
function Tick({ x, y, children, anchor = 'start' }: {
  x: number;
  y: number;
  children: React.ReactNode;
  anchor?: 'start' | 'end';
}) {
  return (
    <text
      x={x}
      y={y}
      textAnchor={anchor}
      className="fill-current font-mono text-[8px] opacity-50"
    >
      {children}
    </text>
  );
}

export function Figures() {
  return (
    <section className="mt-24 w-full max-w-5xl">
      <h2 className="text-2xl font-semibold tracking-tight">What comes out</h2>
      <div className="mt-8 grid gap-6 lg:grid-cols-3">
        <Panel title="Heston" params="κ 2 · ξ 0.3 · ρ −0.7">
          <polygon points={HESTON.band90} className="fill-current opacity-[0.07]" />
          <polygon points={HESTON.band50} className="fill-current opacity-[0.12]" />
          <line
            x1={0}
            x2={W}
            y1={HESTON.start}
            y2={HESTON.start}
            className="stroke-current opacity-25"
            strokeDasharray="3 3"
            strokeWidth={0.7}
          />
          {HESTON.paths.map((points, i) => (
            <polyline
              key={i}
              points={points}
              fill="none"
              className="stroke-current opacity-30"
              strokeWidth={0.7}
            />
          ))}
          <polyline points={HESTON.median} fill="none" className="stroke-current" strokeWidth={1.6} />
          <Tick x={2} y={10}>{HESTON.hi}</Tick>
          <Tick x={2} y={H - 3}>{HESTON.lo}</Tick>
          <Tick x={W - 2} y={HESTON.start - 4} anchor="end">S₀ 100</Tick>
        </Panel>

        <Panel title="Fractional Brownian motion" params="one seed, three H">
          {FBM.map(({ h, points, label }) => (
            <g key={h}>
              <polyline points={points} fill="none" className="stroke-current" strokeWidth={0.9} />
              <Tick x={2} y={label}>H {h}</Tick>
            </g>
          ))}
        </Panel>

        <Panel title="SVI surface" params="implied vol · log-moneyness">
          <line
            x1={W / 2}
            x2={W / 2}
            y1={0}
            y2={H}
            className="stroke-current opacity-20"
            strokeDasharray="3 3"
            strokeWidth={0.7}
          />
          {SVI.map(({ tau, points, end }) => (
            <g key={tau}>
              <polyline points={points} fill="none" className="stroke-current" strokeWidth={1.2} />
              <Tick x={W - 2} y={end - 3} anchor="end">
                {tau < 1 ? `${tau * 12}M` : `${tau}Y`}
              </Tick>
            </g>
          ))}
          <Tick x={2} y={10}>{SVI_AXIS.hi.toFixed(2)}</Tick>
          <Tick x={2} y={H - 3}>{SVI_AXIS.lo.toFixed(2)}</Tick>
          <Tick x={W / 2 + 3} y={H - 3}>k 0</Tick>
        </Panel>
      </div>
    </section>
  );
}

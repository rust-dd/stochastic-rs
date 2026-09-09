import Link from 'next/link';

import { Figures } from '@/components/Figures';

export default function HomePage() {
  return (
    <main className="flex flex-1 flex-col items-center px-6 pb-24">
      <Hero />
      <Stats />
      <Thread />
      <Figures />
      <Tasks />
      <Why />
      <Close />
    </main>
  );
}

function Hero() {
  return (
    <section className="flex flex-col items-center pt-20 pb-16 text-center">
      <p className="mb-5 font-mono text-xs uppercase tracking-[0.2em] text-fd-muted-foreground">
        Rust · Python · CUDA · Metal
      </p>
      <h1 className="text-5xl font-bold tracking-tight sm:text-6xl">
        stochastic-rs
      </h1>
      <p className="mt-6 max-w-2xl text-balance text-lg text-fd-muted-foreground">
        Simulate 131 stochastic processes, price and calibrate against them,
        and move any of it to a GPU by naming one. The model does not change,
        the numbers keep their law, and the crate tells you when a device
        cannot take a configuration instead of quietly running it on the host.
      </p>

      <div className="mt-9 flex flex-wrap items-center justify-center gap-3">
        <Link
          href="/docs/getting-started/quickstart"
          className="rounded-lg bg-fd-primary px-5 py-3 text-sm font-medium text-fd-primary-foreground shadow-sm transition hover:opacity-90"
        >
          Start in five minutes
        </Link>
        <Link
          href="/docs"
          className="rounded-lg border border-fd-border px-5 py-3 text-sm font-medium transition hover:bg-fd-muted"
        >
          Documentation
        </Link>
        <a
          href="https://github.com/rust-dd/stochastic-rs"
          className="rounded-lg border border-fd-border px-5 py-3 text-sm font-medium transition hover:bg-fd-muted"
        >
          GitHub
        </a>
      </div>

      <p className="mt-6 font-mono text-xs text-fd-muted-foreground">
        cargo add stochastic-rs · pip install stochastic-rs
      </p>
    </section>
  );
}

const STATS: [string, string][] = [
  ['131', 'processes behind one trait'],
  ['17×', 'the CPU on a batch of 200 000 paths'],
  ['3 054', 'tests: laws, devices, reproducibility'],
  ['303', 'entries in the Python module'],
];

function Stats() {
  return (
    <section className="grid w-full max-w-5xl grid-cols-2 gap-px overflow-hidden rounded-xl border border-fd-border bg-fd-border lg:grid-cols-4">
      {STATS.map(([figure, label]) => (
        <div key={label} className="bg-fd-card px-6 py-7">
          <div className="font-mono text-3xl font-semibold tracking-tight">
            {figure}
          </div>
          <div className="mt-2 text-sm text-fd-muted-foreground">{label}</div>
        </div>
      ))}
    </section>
  );
}

const THREAD: { title: string; body: string; code: string }[] = [
  {
    title: 'Sample a batch',
    body: 'Every process is built the same way — parameters, grid, horizon, seed — and every one gives you a single path, a parallel batch, and a mapped form over it.',
    code: `let gbm = Gbm::<f32, _>::new(
  0.05, 0.2, 1_024, Some(100.0), Some(1.0), Deterministic::new(7),
);
let paths = gbm.sample_par(1_000);`,
  },
  {
    title: 'Move it to a GPU',
    body: 'Naming a backend is the only change, and the mapped form reads the batch where the kernel wrote it rather than copying it out — which at two hundred thousand paths is 17× the same call on the CPU.',
    code: `let terminal: Vec<f32> = gbm
  .on::<Metal>()
  .sample_map_view(200_000, |p| p[p.len() - 1]);`,
  },
  {
    title: 'Price the same model in closed form',
    body: 'Simulators and pricers are separate: a pricer holds the parameters and takes the query as arguments, so a whole strike-maturity grid is one vectorised sweep.',
    code: `let model = HestonPricer::new(0.04, -0.7, 2.0, 0.04, 0.3, None);
let call = model.price_call(100.0, 100.0, 0.05, 0.02, 0.75);
// 7.7311`,
  },
  {
    title: 'Read a parameter back out',
    body: 'The estimators take a path and return the number that generated it, with the diagnostics to judge how much to believe it.',
    code: `let fbm = Fbm::<f64, _>::new(0.7, 4_096, Some(1.0), Deterministic::new(11)).sample();
let h = RescaledRange::default().estimate(fbm.view())?;
// 0.61, for a true 0.7`,
  },
];

function Thread() {
  return (
    <section className="mt-24 w-full max-w-5xl">
      <h2 className="text-2xl font-semibold tracking-tight">
        One thread, end to end
      </h2>
      <p className="mt-2 max-w-2xl text-fd-muted-foreground">
        Four steps, each one real code that runs. The same four are shown in
        Python in the{' '}
        <Link href="/docs/getting-started/quickstart" className="underline underline-offset-4">
          quickstart
        </Link>
        .
      </p>

      <ol className="mt-10 flex flex-col gap-12">
        {THREAD.map((step, i) => (
          <li key={step.title} className="grid gap-6 lg:grid-cols-[1fr_1.15fr] lg:gap-10">
            <div>
              <div className="flex items-baseline gap-3">
                <span className="font-mono text-sm text-fd-muted-foreground">
                  {String(i + 1).padStart(2, '0')}
                </span>
                <h3 className="text-lg font-semibold">{step.title}</h3>
              </div>
              <p className="mt-3 text-sm leading-relaxed text-fd-muted-foreground">
                {step.body}
              </p>
            </div>
            <Code text={step.code} />
          </li>
        ))}
      </ol>
    </section>
  );
}

/// A code block with its comment lines dimmed — enough shape to read at a
/// glance without pulling a highlighter into the landing page.
function Code({ text }: { text: string }) {
  return (
    <pre className="overflow-x-auto rounded-xl border border-fd-border bg-fd-card p-5 text-[13px] leading-relaxed">
      <code className="font-mono">
        {text.split('\n').map((line, i) => (
          <span
            key={i}
            className={line.trimStart().startsWith('//') ? 'text-fd-muted-foreground' : undefined}
          >
            {line}
            {'\n'}
          </span>
        ))}
      </code>
    </pre>
  );
}

const TASKS: { title: string; body: string; links: [string, string][] }[] = [
  {
    title: 'To simulate',
    body: '131 processes behind one trait — diffusions, jumps, stochastic volatility, short rates, fractional and rough, point processes, subordinators, conditional-variance time series. Heston and its rough relatives, Bates, SABR, CGMY, Hull-White, LMM, fBm and everything built on it.',
    links: [['Processes', '/docs/processes']],
  },
  {
    title: 'To price or calibrate',
    body: 'Analytic, Fourier, ADI and Monte Carlo pricers with first- and second-order Greeks. Fourteen calibrators behind one trait, SVI and SSVI surfaces, curves, bonds, credit and risk.',
    links: [['Quant', '/docs/quant']],
  },
  {
    title: 'To estimate',
    body: 'Eight Hurst estimators, realised measures, jump tests, cointegration, changepoints, extreme value, GARCH fitting — each carrying the paper it came from.',
    links: [['Stats', '/docs/stats']],
  },
  {
    title: 'To draw',
    body: '36 SIMD distribution samplers, most with characteristic function, pdf, cdf and moments in closed form, and 23 copulas with fitting and goodness-of-fit.',
    links: [
      ['Distributions', '/docs/distributions'],
      ['Copulas', '/docs/copulas'],
    ],
  },
  {
    title: 'To do it from Python',
    body: '303 entries, numpy in and numpy out, and the same device argument. Wheels for every platform, no feature flags to choose.',
    links: [['Python', '/docs/python']],
  },
];

function Tasks() {
  return (
    <section className="mt-28 w-full max-w-5xl">
      <h2 className="text-2xl font-semibold tracking-tight">
        What are you here for?
      </h2>
      <div className="mt-8 divide-y divide-fd-border border-y border-fd-border">
        {TASKS.map((task) => (
          <div key={task.title} className="grid gap-2 py-6 lg:grid-cols-[220px_1fr] lg:gap-10">
            <h3 className="font-semibold">{task.title}</h3>
            <div>
              <p className="text-sm leading-relaxed text-fd-muted-foreground">
                {task.body}
              </p>
              <div className="mt-3 flex flex-wrap gap-4">
                {task.links.map(([label, href]) => (
                  <Link
                    key={href}
                    href={href}
                    className="text-sm font-medium underline underline-offset-4 hover:opacity-80"
                  >
                    {label} →
                  </Link>
                ))}
              </div>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

const WHY: { title: string; body: string }[] = [
  {
    title: 'A declaration, not a kernel',
    body: 'A family is written once in a small DSL — its state, its noise, its step — and the same text is rendered as CUDA C for NVRTC and as MSL for Metal. Adding a process means adding a declaration, not writing and debugging two kernels. 120 families carry 129 of the processes; the two that need pipelines of their own have them.',
  },
  {
    title: 'The speed is in the shape of the call',
    body: 'Ask for the grid and you get the grid; ask for a borrowed view and nothing is copied; ask for a fold and the kernel never writes it. No flags, no tuning — the same program, told what you actually want back.',
  },
  {
    title: 'Measured against mathematics',
    body: 'Beyond the host-versus-device suite there is one that states each model’s closed form, so a sampler wrong on both sides still fails. It earns its keep: a ziggurat tail folded inward, a circulant embedding that lost its long memory in f32, an inverse-Gaussian draw that cancelled to zero on a fine grid.',
  },
];

function Why() {
  return (
    <section className="mt-28 w-full max-w-5xl">
      <h2 className="text-2xl font-semibold tracking-tight">Why this one</h2>
      <div className="mt-8 grid gap-8 lg:grid-cols-3">
        {WHY.map((item) => (
          <div key={item.title}>
            <h3 className="font-semibold">{item.title}</h3>
            <p className="mt-3 text-sm leading-relaxed text-fd-muted-foreground">
              {item.body}
            </p>
          </div>
        ))}
      </div>
      <p className="mt-8 text-sm text-fd-muted-foreground">
        The numbers behind all three, the method that produced them, and what
        did <em>not</em> work are in{' '}
        <Link href="/docs/benchmarks" className="underline underline-offset-4">
          Benchmarks
        </Link>
        .
      </p>
    </section>
  );
}

function Close() {
  return (
    <section className="mt-28 flex w-full max-w-5xl flex-col items-center gap-5 rounded-xl border border-fd-border bg-fd-card px-6 py-12 text-center">
      <h2 className="text-2xl font-semibold tracking-tight">
        Simulate a path, price an option, estimate a Hurst exponent
      </h2>
      <p className="max-w-xl text-sm text-fd-muted-foreground">
        The quickstart does all three, in Rust and in Python side by side.
      </p>
      <div className="flex flex-wrap items-center justify-center gap-3">
        <Link
          href="/docs/getting-started/quickstart"
          className="rounded-lg bg-fd-primary px-5 py-3 text-sm font-medium text-fd-primary-foreground shadow-sm transition hover:opacity-90"
        >
          Quickstart
        </Link>
        <a
          href="https://crates.io/crates/stochastic-rs"
          className="rounded-lg border border-fd-border px-5 py-3 text-sm font-medium transition hover:bg-fd-muted"
        >
          crates.io
        </a>
        <a
          href="https://pypi.org/project/stochastic-rs/"
          className="rounded-lg border border-fd-border px-5 py-3 text-sm font-medium transition hover:bg-fd-muted"
        >
          PyPI
        </a>
      </div>
    </section>
  );
}

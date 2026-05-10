import Link from 'next/link';

export default function HomePage() {
  return (
    <main className="flex flex-1 flex-col items-center justify-center px-6 py-20 text-center">
      <p className="mb-4 font-mono text-sm uppercase tracking-widest text-fd-muted-foreground">
        Rust · Python · CUDA · Metal
      </p>
      <h1 className="text-5xl font-bold tracking-tight sm:text-6xl">
        stochastic-rs
      </h1>
      <p className="mt-6 max-w-2xl text-lg text-fd-muted-foreground">
        High-performance Rust library for stochastic process simulation,
        option pricing, calibration, statistical estimators, copulas, and
        neural-network volatility surrogates. 120+ processes, 19 SIMD
        distributions, and a typed pricing engine — with first-class Python
        bindings.
      </p>

      <div className="mt-10 flex flex-wrap items-center justify-center gap-4">
        <Link
          href="/docs"
          className="rounded-lg bg-fd-primary px-5 py-3 text-sm font-medium text-fd-primary-foreground shadow hover:opacity-90"
        >
          Read the docs
        </Link>
        <Link
          href="/docs/getting-started/quickstart"
          className="rounded-lg border border-fd-border px-5 py-3 text-sm font-medium hover:bg-fd-muted"
        >
          Quickstart
        </Link>
        <a
          href="https://crates.io/crates/stochastic-rs"
          className="rounded-lg border border-fd-border px-5 py-3 text-sm font-medium hover:bg-fd-muted"
        >
          crates.io
        </a>
      </div>

      <section className="mt-24 grid w-full max-w-5xl grid-cols-1 gap-6 text-left sm:grid-cols-2 lg:grid-cols-3">
        <Feature
          title="120+ processes"
          body="Diffusion, jump, fractional, rough, short-rate, HJM, LMM. Each with a generic-precision ProcessExt<T> impl and CUDA / SIMD acceleration where it makes a difference."
        />
        <Feature
          title="Pricing & calibration"
          body="BSM, Bachelier, Heston, SABR, Bergomi, rough Bergomi, double-Heston, CGMYSV, Hull-White swaption. Closed-form, Fourier, Monte Carlo, finite difference, lattice, Bermudan LSM."
        />
        <Feature
          title="Statistics & estimators"
          body="Hurst (Fukasawa, R/S, …), MLE for 1-D diffusions with 6 transition-density approximations, ADF / KPSS / Phillips-Perron, realised variance with BNHLS bandwidth."
        />
        <Feature
          title="Risk & credit"
          body="VaR / CVaR / expected shortfall, Sharpe / Sortino / Calmar with no hard-coded annualisation, Merton structural model, hazard bootstrap, JLT migration with Padé-13 matrix exp."
        />
        <Feature
          title="Vol surface & SVI"
          body="Implied-vol surfaces from market quotes, SVI / SSVI / SABR-smile fits with arbitrage-free interpolation. Plug a model into ImpliedVolSurface::from_flat_iv_grid for fast inversion."
        />
        <Feature
          title="Python bindings"
          body="Full coverage via PyO3 — 198 classes + 12 functions across distributions, processes, pricers, calibrators, copulas, and stats. NumPy in / NumPy out."
        />
      </section>
    </main>
  );
}

function Feature({ title, body }: { title: string; body: string }) {
  return (
    <div className="rounded-xl border border-fd-border bg-fd-card p-6 shadow-sm">
      <h2 className="font-semibold">{title}</h2>
      <p className="mt-2 text-sm text-fd-muted-foreground">{body}</p>
    </div>
  );
}

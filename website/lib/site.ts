/**
 * Single source of truth for the canonical URLs and marketing copy that feed
 * `<head>` metadata, the sitemap, robots.txt, the OG image and the JSON-LD
 * block. Anything user-visible outside the MDX content should read from here
 * so the canonical host never drifts between surfaces.
 */
export const SITE = {
  name: 'stochastic-rs',
  url: 'https://stochastic.rust-dd.com',
  repository: 'https://github.com/rust-dd/stochastic-rs',
  crates: 'https://crates.io/crates/stochastic-rs',
  docsRs: 'https://docs.rs/stochastic-rs',
  pypi: 'https://pypi.org/project/stochastic-rs/',
  author: 'Daniel Boros',
  tagline: 'Quantitative Finance in Rust',
  description:
    'Open-source quantitative finance for Rust and Python: 120+ stochastic processes, option pricing, Heston/SABR calibration, vol surfaces, fixed income and risk.',
} as const;

export const structuredData = {
  '@context': 'https://schema.org',
  '@graph': [
    {
      '@type': 'SoftwareSourceCode',
      '@id': `${SITE.url}/#software`,
      name: SITE.name,
      description: SITE.description,
      url: SITE.url,
      codeRepository: SITE.repository,
      programmingLanguage: [
        { '@type': 'ComputerLanguage', name: 'Rust' },
        { '@type': 'ComputerLanguage', name: 'Python' },
      ],
      runtimePlatform: ['Rust', 'CPython'],
      license: 'https://opensource.org/licenses/MIT',
      author: { '@type': 'Person', name: SITE.author },
      applicationCategory: 'DeveloperApplication',
      keywords: [
        'quantitative finance',
        'option pricing',
        'stochastic processes',
        'Monte Carlo',
        'model calibration',
        'rough volatility',
        'fixed income',
        'copulas',
      ].join(', '),
    },
    {
      '@type': 'WebSite',
      '@id': `${SITE.url}/#website`,
      name: `${SITE.name} — ${SITE.tagline}`,
      description: SITE.description,
      url: SITE.url,
      inLanguage: 'en',
      publisher: { '@type': 'Person', name: SITE.author },
    },
  ],
};

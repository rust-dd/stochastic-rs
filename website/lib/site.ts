import { FACTS } from '@/lib/facts';

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
  /** Zenodo concept DOI — always resolves to the latest release. */
  doi: '10.5281/zenodo.21553307',
  author: 'Daniel Boros',
  authorSameAs: ['https://orcid.org/0009-0008-1207-2251', 'https://github.com/dancixx'],
  tagline: 'Quantitative Finance in Rust',
  /** The sentence an answer to "what is stochastic-rs?" should be able to quote. */
  definition:
    'stochastic-rs is an open-source quantitative-finance library for Rust and Python: it simulates stochastic processes on the CPU and the GPU, prices and calibrates option models, and estimates their parameters from data.',
  description: `Open-source quantitative finance for Rust and Python: ${FACTS.processes} stochastic processes, option pricing, Heston/SABR calibration, vol surfaces, fixed income and risk.`,
} as const;

const AUTHOR_ID = `${SITE.url}/#author`;
const SOFTWARE_ID = `${SITE.url}/#software`;
const WEBSITE_ID = `${SITE.url}/#website`;

export const JSON_LD_IDS = { author: AUTHOR_ID, software: SOFTWARE_ID, website: WEBSITE_ID };

export const structuredData = {
  '@context': 'https://schema.org',
  '@graph': [
    {
      '@type': 'SoftwareSourceCode',
      '@id': SOFTWARE_ID,
      name: SITE.name,
      description: SITE.description,
      abstract: SITE.definition,
      url: SITE.url,
      codeRepository: SITE.repository,
      programmingLanguage: [
        { '@type': 'ComputerLanguage', name: 'Rust' },
        { '@type': 'ComputerLanguage', name: 'Python' },
      ],
      runtimePlatform: ['Rust', 'CPython'],
      license: 'https://opensource.org/licenses/MIT',
      version: FACTS.version,
      identifier: {
        '@type': 'PropertyValue',
        propertyID: 'DOI',
        value: SITE.doi,
        url: `https://doi.org/${SITE.doi}`,
      },
      sameAs: [SITE.repository, SITE.crates, SITE.pypi, SITE.docsRs, `https://doi.org/${SITE.doi}`],
      author: { '@id': AUTHOR_ID },
      applicationCategory: 'DeveloperApplication',
      keywords: [
        'quantitative finance',
        'option pricing',
        'stochastic processes',
        'Monte Carlo',
        'model calibration',
        'rough volatility',
        'Heston model',
        'fractional Brownian motion',
        'fixed income',
        'copulas',
      ].join(', '),
    },
    {
      '@type': 'Person',
      '@id': AUTHOR_ID,
      name: SITE.author,
      alternateName: 'Dániel Boros',
      url: 'https://rust-dd.com',
      sameAs: SITE.authorSameAs,
    },
    {
      '@type': 'WebSite',
      '@id': WEBSITE_ID,
      name: `${SITE.name} — ${SITE.tagline}`,
      description: SITE.description,
      url: SITE.url,
      inLanguage: 'en',
      publisher: { '@id': AUTHOR_ID },
    },
  ],
};

import './global.css';
import { RootProvider } from 'fumadocs-ui/provider/next';
import type { ReactNode } from 'react';
import type { Metadata } from 'next';
import { SITE, structuredData } from '@/lib/site';

export const metadata: Metadata = {
  metadataBase: new URL(SITE.url),
  title: {
    default: `${SITE.name} — ${SITE.tagline}`,
    template: `%s | ${SITE.name}`,
  },
  description: SITE.description,
  applicationName: SITE.name,
  category: 'technology',
  keywords: [
    'quantitative finance',
    'rust',
    'python',
    'option pricing',
    'stochastic processes',
    'monte carlo',
    'heston model',
    'SABR',
    'rough volatility',
    'model calibration',
    'volatility surface',
    'SVI',
    'fixed income',
    'yield curve',
    'credit risk',
    'copula',
    'fractional brownian motion',
    'hurst exponent',
    'quantlib alternative',
  ],
  authors: [{ name: SITE.author, url: SITE.repository }],
  creator: SITE.author,
  publisher: SITE.author,
  alternates: { canonical: '/' },
  openGraph: {
    type: 'website',
    url: SITE.url,
    siteName: SITE.name,
    title: `${SITE.name} — ${SITE.tagline}`,
    description: SITE.description,
    locale: 'en_US',
  },
  twitter: {
    card: 'summary_large_image',
    title: `${SITE.name} — ${SITE.tagline}`,
    description: SITE.description,
  },
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      'max-image-preview': 'large',
      'max-snippet': -1,
      'max-video-preview': -1,
    },
  },
};

export default function Layout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{ __html: JSON.stringify(structuredData) }}
        />
      </head>
      <body className="flex flex-col min-h-screen">
        <RootProvider>{children}</RootProvider>
      </body>
    </html>
  );
}

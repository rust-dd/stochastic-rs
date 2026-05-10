import './global.css';
import { RootProvider } from 'fumadocs-ui/provider/next';
import type { ReactNode } from 'react';
import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: {
    default: 'stochastic-rs',
    template: '%s | stochastic-rs',
  },
  description:
    'High-performance Rust library for stochastic process simulation, quantitative finance, and statistical modelling.',
  metadataBase: new URL('https://stochastic-rs.vercel.app'),
};

export default function Layout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="flex flex-col min-h-screen">
        <RootProvider>{children}</RootProvider>
      </body>
    </html>
  );
}

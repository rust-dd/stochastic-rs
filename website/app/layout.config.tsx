import type { BaseLayoutProps } from 'fumadocs-ui/layouts/shared';

export const baseOptions: BaseLayoutProps = {
  nav: {
    title: (
      <>
        <span className="font-mono font-semibold">stochastic-rs</span>
      </>
    ),
  },
  links: [
    {
      text: 'Documentation',
      url: '/docs',
      active: 'nested-url',
    },
    {
      text: 'crates.io',
      url: 'https://crates.io/crates/stochastic-rs',
      external: true,
    },
    {
      text: 'docs.rs',
      url: 'https://docs.rs/stochastic-rs',
      external: true,
    },
    {
      text: 'GitHub',
      url: 'https://github.com/dancixx/stochastic-rs',
      external: true,
    },
  ],
  githubUrl: 'https://github.com/dancixx/stochastic-rs',
};

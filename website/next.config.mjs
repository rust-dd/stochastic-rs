import { createMDX } from 'fumadocs-mdx/next';

const withMDX = createMDX();

/** A request whose `Accept` header asks for Markdown gets the page's Markdown twin. */
const wantsMarkdown = [
  { type: 'header', key: 'accept', value: '(.*)text/(markdown|x-markdown)(.*)' },
];

/** @type {import('next').NextConfig} */
const config = {
  reactStrictMode: true,
  experimental: {
    optimizePackageImports: ['fumadocs-ui', 'fumadocs-core'],
  },
  async rewrites() {
    return {
      beforeFiles: [
        { source: '/docs.md', destination: '/llms.mdx/docs' },
        { source: '/docs.mdx', destination: '/llms.mdx/docs' },
        { source: '/docs/:path*.md', destination: '/llms.mdx/docs/:path*' },
        { source: '/docs/:path*.mdx', destination: '/llms.mdx/docs/:path*' },
        { source: '/docs', has: wantsMarkdown, destination: '/llms.mdx/docs' },
        { source: '/docs/:path*', has: wantsMarkdown, destination: '/llms.mdx/docs/:path*' },
      ],
    };
  },
};

export default withMDX(config);

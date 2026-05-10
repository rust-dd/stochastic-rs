import type { ReactNode } from 'react';

export interface PaperRefProps {
  author: string;
  year: number;
  title: string;
  doi?: string;
  arxiv?: string;
  url?: string;
  children?: ReactNode;
}

export function PaperRef(props: PaperRefProps) {
  const link = props.doi
    ? { href: `https://doi.org/${props.doi}`, label: `doi:${props.doi}` }
    : props.arxiv
      ? { href: `https://arxiv.org/abs/${props.arxiv}`, label: `arXiv:${props.arxiv}` }
      : props.url
        ? { href: props.url, label: props.url }
        : null;

  return (
    <div className="my-3 rounded-md border border-fd-border bg-fd-card px-4 py-3 text-sm">
      <span className="font-medium">{props.author}</span>{' '}
      <span className="text-fd-muted-foreground">({props.year}).</span>{' '}
      <span className="italic">{props.title}.</span>{' '}
      {link ? (
        <a
          href={link.href}
          className="font-mono text-fd-primary hover:underline"
          target="_blank"
          rel="noreferrer noopener"
        >
          {link.label}
        </a>
      ) : null}
      {props.children}
    </div>
  );
}

import type { Reference } from '@/lib/frontmatter';
import { referenceLink } from '@/lib/source';

/** The page's frontmatter `references`, rendered as its closing section. */
export function References({ references }: { references: Reference[] }) {
  if (references.length === 0) return null;

  return (
    <>
      <h2 id="references">References</h2>
      <ul>
        {references.map((ref) => {
          const link = referenceLink(ref);
          return (
            <li key={`${ref.author}-${ref.year}-${ref.title}`}>
              {ref.author} ({ref.year}). <em>{ref.title}</em>.{' '}
              {link ? (
                <a href={link.href} className="font-mono text-sm">
                  {link.label}
                </a>
              ) : null}
            </li>
          );
        })}
      </ul>
    </>
  );
}

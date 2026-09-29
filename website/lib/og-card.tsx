import { ImageResponse } from 'next/og';
import { SITE } from '@/lib/site';

export const OG_SIZE = { width: 1200, height: 630 };

/** Band reserved at the very bottom edge, clear of the footer baseline. */
const PATH_HEIGHT = 74;

/**
 * Deterministic fractional-ish walk used as the card's background motif. A
 * fixed LCG keeps the rendered PNG byte-identical across builds, which stops
 * social platforms from re-fetching a "changed" image on every deploy.
 */
function samplePath(): string {
  const points = 120;
  const width = 1200;
  const height = PATH_HEIGHT;
  let seed = 0x2545f491;
  let value = 0;

  const coords: string[] = [];
  for (let i = 0; i < points; i += 1) {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    value += (seed / 0x7fffffff - 0.5) * 13;
    value *= 0.97;
    const x = (i / (points - 1)) * width;
    const y = height / 2 - value;
    coords.push(`${x.toFixed(1)},${y.toFixed(1)}`);
  }

  return `M ${coords.join(' L ')}`;
}

export interface OgCardProps {
  /** Small caps line above the title. */
  eyebrow: string;
  title: string;
  subtitle?: string;
  /** Kept to five so the row never wraps into the sample-path motif below it. */
  chips?: string[];
}

/** Titles longer than a line at 92px drop to a size that still fits two lines. */
function titleSize(title: string): number {
  if (title.length <= 20) return 92;
  if (title.length <= 34) return 72;
  return 60;
}

export function ogCard({ eyebrow, title, subtitle, chips = [] }: OgCardProps): ImageResponse {
  return new ImageResponse(
    (
      <div
        style={{
          width: '100%',
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'space-between',
          background: '#0b0b0e',
          backgroundImage:
            'radial-gradient(circle at 78% 12%, rgba(120,130,255,0.16), transparent 55%)',
          padding: '72px 76px',
        }}
      >
        <div style={{ display: 'flex', flexDirection: 'column' }}>
          <div
            style={{
              display: 'flex',
              fontSize: 22,
              letterSpacing: 6,
              textTransform: 'uppercase',
              color: '#8b8b98',
            }}
          >
            {eyebrow}
          </div>
          <div
            style={{
              display: 'flex',
              marginTop: 26,
              fontSize: titleSize(title),
              fontWeight: 700,
              letterSpacing: -3,
              lineHeight: 1.05,
              color: '#fafafa',
            }}
          >
            {title}
          </div>
          {subtitle ? (
            <div
              style={{
                display: 'flex',
                marginTop: 18,
                maxWidth: 1000,
                fontSize: 34,
                fontWeight: 500,
                lineHeight: 1.3,
                color: '#c9c9d4',
              }}
            >
              {subtitle}
            </div>
          ) : null}
        </div>

        {chips.length > 0 ? (
          <div style={{ display: 'flex', gap: 12 }}>
            {chips.map((chip) => (
              <div
                key={chip}
                style={{
                  display: 'flex',
                  padding: '10px 20px',
                  borderRadius: 999,
                  border: '1px solid #2a2a33',
                  background: '#141419',
                  color: '#d4d4de',
                  fontSize: 24,
                }}
              >
                {chip}
              </div>
            ))}
          </div>
        ) : null}

        <div
          style={{
            display: 'flex',
            alignItems: 'flex-end',
            justifyContent: 'space-between',
          }}
        >
          <div style={{ display: 'flex', fontSize: 26, color: '#8b8b98' }}>
            {new URL(SITE.url).host}
          </div>
          <div style={{ display: 'flex', fontSize: 26, color: '#8b8b98' }}>
            MIT · crates.io · PyPI
          </div>
        </div>

        <svg
          width="1200"
          height={PATH_HEIGHT}
          viewBox={`0 0 1200 ${PATH_HEIGHT}`}
          style={{ position: 'absolute', left: 0, bottom: 0, opacity: 0.34 }}
        >
          <path
            d={samplePath()}
            fill="none"
            stroke="#7b83ff"
            strokeWidth="2.5"
            strokeLinejoin="round"
          />
        </svg>
      </div>
    ),
    OG_SIZE,
  );
}

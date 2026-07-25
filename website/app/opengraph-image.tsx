import { ImageResponse } from 'next/og';
import { SITE } from '@/lib/site';

export const alt = `${SITE.name} — ${SITE.tagline}`;
export const size = { width: 1200, height: 630 };
export const contentType = 'image/png';

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

/** Kept to five so the row never wraps into the sample-path motif below it. */
const CHIPS = [
  '120+ processes',
  'Option pricing',
  'Calibration',
  'SIMD / GPU',
  'Python',
];

export default function Image() {
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
            Rust · Python · CUDA · Metal
          </div>
          <div
            style={{
              display: 'flex',
              marginTop: 26,
              fontSize: 92,
              fontWeight: 700,
              letterSpacing: -3,
              color: '#fafafa',
            }}
          >
            {SITE.name}
          </div>
          <div
            style={{
              display: 'flex',
              marginTop: 14,
              fontSize: 40,
              fontWeight: 500,
              color: '#c9c9d4',
            }}
          >
            {SITE.tagline}
          </div>
        </div>

        <div style={{ display: 'flex', gap: 12 }}>
          {CHIPS.map((chip) => (
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

        <div
          style={{
            display: 'flex',
            alignItems: 'flex-end',
            justifyContent: 'space-between',
          }}
        >
          <div style={{ display: 'flex', fontSize: 26, color: '#8b8b98' }}>
            stochastic.rust-dd.com
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
    size,
  );
}

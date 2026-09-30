import { PROCESSES_ROUNDED } from '@/lib/facts';
import { OG_SIZE, ogCard } from '@/lib/og-card';
import { SITE } from '@/lib/site';

export const alt = `${SITE.name} — ${SITE.tagline}`;
export const size = OG_SIZE;
export const contentType = 'image/png';

export default function Image() {
  return ogCard({
    eyebrow: 'Rust · Python · CUDA · Metal',
    title: SITE.name,
    subtitle: SITE.tagline,
    chips: [`${PROCESSES_ROUNDED} processes`, 'Option pricing', 'Calibration', 'SIMD / GPU', 'Python'],
  });
}

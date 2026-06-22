import Reveal from './Reveal'

export default function SectionHeader({ eyebrow, title, subtitle, accentDot = '#22d3ee', align = 'left' }) {
  return (
    <Reveal>
      <div className={align === 'center' ? 'text-center max-w-3xl mx-auto' : 'max-w-3xl'}>
        <span className="eyebrow">
          <span className="h-1.5 w-1.5 rounded-full" style={{ background: accentDot, boxShadow: `0 0 12px ${accentDot}` }} />
          {eyebrow}
        </span>
        <h2 className="mt-4 text-3xl sm:text-4xl md:text-5xl font-bold tracking-tight text-white">
          {title}
        </h2>
        {subtitle && (
          <p className="mt-4 text-base sm:text-lg text-slate-400 leading-relaxed">{subtitle}</p>
        )}
      </div>
    </Reveal>
  )
}

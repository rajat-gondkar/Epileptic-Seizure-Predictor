import { useEffect, useState } from 'react'
import { motion, useScroll, useSpring } from 'framer-motion'

const LINKS = [
  { id: 'datasets', label: 'Datasets' },
  { id: 'preprocessing', label: 'Preprocessing' },
  { id: 'genetics', label: 'Genetics' },
  { id: 'ctgan', label: 'CTGAN' },
  { id: 'training', label: 'Training' },
  { id: 'fusion', label: 'Fusion' },
  { id: 'demo', label: 'Live Demo' },
  { id: 'results', label: 'Results' },
]

export default function Nav() {
  const [active, setActive] = useState('hero')
  const [scrolled, setScrolled] = useState(false)
  const { scrollYProgress } = useScroll()
  const scaleX = useSpring(scrollYProgress, { stiffness: 120, damping: 30, restDelta: 0.001 })

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 40)
    onScroll()
    window.addEventListener('scroll', onScroll)
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  useEffect(() => {
    const ids = ['hero', ...LINKS.map((l) => l.id)]
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          if (e.isIntersecting) setActive(e.target.id)
        })
      },
      { rootMargin: '-45% 0px -50% 0px' }
    )
    ids.forEach((id) => {
      const el = document.getElementById(id)
      if (el) observer.observe(el)
    })
    return () => observer.disconnect()
  }, [])

  return (
    <>
      <motion.div
        className="fixed top-0 left-0 right-0 h-[2px] origin-left z-[60] bg-gradient-to-r from-eeg via-gene to-fuse"
        style={{ scaleX }}
      />
      <header
        className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
          scrolled ? 'bg-ink-950/80 backdrop-blur-xl border-b border-white/[0.06]' : ''
        }`}
      >
        <nav className="max-w-7xl mx-auto px-5 sm:px-8 h-16 flex items-center justify-between">
          <a href="#hero" className="flex items-center gap-2.5 group">
            <span className="relative grid place-items-center h-8 w-8 rounded-lg bg-gradient-to-br from-eeg/20 to-gene/20 border border-white/10">
              <span className="h-2 w-2 rounded-full bg-gradient-to-r from-eeg to-gene animate-pulse-line" />
            </span>
            <span className="font-semibold tracking-tight text-white">
              Neuro<span className="grad-text">Genix</span>
            </span>
          </a>
          <div className="hidden lg:flex items-center gap-1">
            {LINKS.map((l) => (
              <a
                key={l.id}
                href={`#${l.id}`}
                className={`px-3 py-1.5 rounded-lg text-sm transition-colors ${
                  active === l.id ? 'text-white bg-white/[0.06]' : 'text-slate-400 hover:text-white'
                }`}
              >
                {l.label}
              </a>
            ))}
          </div>
          <a
            href="#fusion"
            className="text-sm font-medium px-4 py-2 rounded-lg bg-fuse/15 text-fuse border border-fuse/30 hover:bg-fuse/25 transition-colors"
          >
            Try the Simulator
          </a>
        </nav>
      </header>
    </>
  )
}

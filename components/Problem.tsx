import { motion } from 'framer-motion'
import { useInView } from 'framer-motion'
import { useRef } from 'react'
import { Globe, EyeOff, TrendingDown, Clock, Users2, AlertTriangle } from 'lucide-react'

const problems = [
  {
    icon:  Globe,
    title: 'Outdated or No Website',
    desc:  'Your competitors are showing up on Google. You\'re invisible. Customers Google you and find nothing — or worse, an old, broken site.',
    color: 'rgba(239,68,68,0.15)',
    border: 'rgba(239,68,68,0.2)',
    iconColor: '#ef4444',
  },
  {
    icon:  EyeOff,
    title: 'Zero Online Presence',
    desc:  'No reviews, no social media, no ads. In 2026, if you\'re not online, you don\'t exist to 90% of potential customers.',
    color: 'rgba(245,158,11,0.15)',
    border: 'rgba(245,158,11,0.2)',
    iconColor: '#f59e0b',
  },
  {
    icon:  TrendingDown,
    title: 'Losing Customers Daily',
    desc:  'Every day without a proper system, leads slip through the cracks. No follow-up, no automation, no pipeline — just missed revenue.',
    color: 'rgba(239,68,68,0.15)',
    border: 'rgba(239,68,68,0.2)',
    iconColor: '#ef4444',
  },
  {
    icon:  Clock,
    title: 'Wasting Time on Manual Tasks',
    desc:  'Responding to every inquiry manually, chasing leads, posting on social media one by one. This kills productivity and growth.',
    color: 'rgba(245,158,11,0.15)',
    border: 'rgba(245,158,11,0.2)',
    iconColor: '#f59e0b',
  },
  {
    icon:  Users2,
    title: 'No Referral or Lead System',
    desc:  'Relying only on word-of-mouth is not a strategy. Without a predictable lead system, your revenue is unpredictable.',
    color: 'rgba(239,68,68,0.15)',
    border: 'rgba(239,68,68,0.2)',
    iconColor: '#ef4444',
  },
  {
    icon:  AlertTriangle,
    title: 'Competitors Are Moving Fast',
    desc:  'Right now, your competitors are running ads, ranking on Google, and automating follow-ups. The gap is growing every week.',
    color: 'rgba(168,85,247,0.15)',
    border: 'rgba(168,85,247,0.2)',
    iconColor: '#a855f7',
  },
]

function ProblemCard({ problem, index }: { problem: typeof problems[0]; index: number }) {
  const ref = useRef(null)
  const inView = useInView(ref, { once: true, margin: '-80px' })
  const Icon = problem.icon

  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 40 }}
      animate={inView ? { opacity: 1, y: 0 } : {}}
      transition={{ duration: 0.6, delay: index * 0.08, ease: [0.16, 1, 0.3, 1] }}
      className="group relative rounded-2xl p-6 cursor-default transition-all duration-300"
      style={{
        background: problem.color,
        border: `1px solid ${problem.border}`,
        boxShadow: '0 4px 24px rgba(0,0,0,0.4)',
      }}
      whileHover={{ y: -4, transition: { duration: 0.2 } }}
    >
      <div className="w-11 h-11 rounded-xl flex items-center justify-center mb-4"
        style={{ background: 'rgba(0,0,0,0.3)', border: `1px solid ${problem.border}` }}>
        <Icon size={20} style={{ color: problem.iconColor }} />
      </div>
      <h3 className="text-white font-semibold text-base mb-2">{problem.title}</h3>
      <p className="text-white/50 text-sm leading-relaxed">{problem.desc}</p>

      {/* Hover glow */}
      <div className="absolute inset-0 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none"
        style={{ boxShadow: `0 0 30px ${problem.iconColor}20` }} />
    </motion.div>
  )
}

export default function Problem() {
  const ref = useRef(null)
  const inView = useInView(ref, { once: true, margin: '-100px' })

  return (
    <section className="relative py-32 px-4">
      <div className="max-w-7xl mx-auto">

        {/* Header */}
        <div ref={ref} className="text-center mb-16">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={inView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.5 }}
            className="section-label mb-6 mx-auto w-fit"
          >
            <span className="glow-dot bg-red-500" style={{ boxShadow: '0 0 6px rgba(239,68,68,0.8)' }} />
            The Problem
          </motion.div>
          <motion.h2
            initial={{ opacity: 0, y: 20 }}
            animate={inView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.6, delay: 0.1 }}
            className="font-black tracking-tight text-white max-w-3xl mx-auto"
            style={{ fontSize: 'clamp(2rem, 4vw, 3.5rem)' }}
          >
            Most local businesses are
            <br />
            <span style={{
              background: 'linear-gradient(135deg, #ef4444, #f59e0b)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              backgroundClip: 'text',
            }}>
              invisible online.
            </span>
          </motion.h2>
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={inView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="mt-5 text-white/40 text-lg max-w-xl mx-auto"
          >
            And losing thousands in revenue to competitors who figured this out.
            The gap widens every single day.
          </motion.p>
        </div>

        {/* Cards grid */}
        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {problems.map((p, i) => (
            <ProblemCard key={p.title} problem={p} index={i} />
          ))}
        </div>

        {/* Bottom callout */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: '-80px' }}
          transition={{ duration: 0.6 }}
          className="mt-12 text-center"
        >
          <p className="text-white/30 text-sm">
            Sound familiar?{' '}
            <a href="#solution" className="text-brand-400 hover:text-brand-300 underline underline-offset-4 transition-colors">
              Here's the solution →
            </a>
          </p>
        </motion.div>
      </div>
    </section>
  )
}

import { motion } from 'framer-motion'
import PageLayout from '@/components/PageLayout'
import { Zap, Target, TrendingUp, Heart } from 'lucide-react'

const values = [
  { icon: Target,     title: 'Results Over Promises', desc: 'We don\'t sell dreams. We build systems that deliver measurable, trackable growth — and we show you the numbers.' },
  { icon: Zap,        title: 'Speed Without Sacrifice', desc: 'Your website launches in 7 days. Your first leads arrive in 14. No months-long projects, no slow agencies.' },
  { icon: TrendingUp, title: 'Long-Term Partnership',   desc: 'We don\'t disappear after launch. Your success is how we grow — so we\'re invested in yours for the long run.' },
  { icon: Heart,      title: 'Local Business First',    desc: 'We only work with local businesses. Your needs, budget, and market are what we specialise in — nothing else.' },
]

export default function About() {
  return (
    <PageLayout
      title="About Us"
      description="PantaSocial LLC — built by Andreja Pantic to help local businesses compete online with automated growth systems."
    >
      {/* Header */}
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6 }} className="mb-16">
        <div className="section-label mb-6 w-fit"><span className="glow-dot" />Our Story</div>
        <h1 className="text-5xl font-black text-white mb-6 leading-tight">
          Built for the businesses
          <br />
          <span className="gradient-text-brand">that built your community.</span>
        </h1>
        <p className="text-white/50 text-lg leading-relaxed max-w-2xl">
          PantaSocial LLC was founded by Andreja Pantic with one mission: give local businesses
          the same digital firepower that big corporations have — without the enterprise price tag.
        </p>
      </motion.div>

      {/* Founder */}
      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.1 }}
        className="glass-card rounded-2xl p-8 mb-16 flex flex-col sm:flex-row gap-8 items-start"
      >
        <div className="w-20 h-20 rounded-2xl flex items-center justify-center text-2xl font-black text-white flex-shrink-0"
          style={{ background: 'linear-gradient(135deg, #6366f1, #a855f7)' }}>
          AP
        </div>
        <div>
          <div className="text-white font-bold text-xl mb-1">Andreja Pantic</div>
          <div className="text-brand-400 text-sm font-medium mb-4">Founder & CEO, PantaSocial LLC</div>
          <p className="text-white/50 leading-relaxed text-sm">
            "I started PantaSocial because I watched great local businesses lose customers to
            inferior competitors with better websites. That's not a talent problem — it's a
            visibility problem. We fix that. Every system we build is designed to make the
            best business win, not just the one with the biggest marketing budget."
          </p>
        </div>
      </motion.div>

      {/* Values */}
      <div className="mb-16">
        <h2 className="text-2xl font-bold text-white mb-8">What we stand for</h2>
        <div className="grid sm:grid-cols-2 gap-5">
          {values.map(({ icon: Icon, title, desc }, i) => (
            <motion.div
              key={title}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: i * 0.07 }}
              className="glass-card rounded-2xl p-6 group"
              whileHover={{ y: -3, transition: { duration: 0.2 } }}
            >
              <div className="w-10 h-10 rounded-xl flex items-center justify-center mb-4"
                style={{ background: 'rgba(99,102,241,0.1)', border: '1px solid rgba(99,102,241,0.2)' }}>
                <Icon size={18} className="text-brand-400" />
              </div>
              <h3 className="text-white font-semibold text-base mb-2">{title}</h3>
              <p className="text-white/50 text-sm leading-relaxed">{desc}</p>
            </motion.div>
          ))}
        </div>
      </div>

      {/* CTA strip */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        className="rounded-2xl p-8 text-center"
        style={{ background: 'linear-gradient(135deg, rgba(99,102,241,0.1), rgba(168,85,247,0.08))', border: '1px solid rgba(99,102,241,0.2)' }}
      >
        <h3 className="text-white font-bold text-2xl mb-3">Ready to grow?</h3>
        <p className="text-white/50 text-sm mb-6">Get your free website preview and market audit today.</p>
        <a href="/contact" className="btn-primary inline-flex">Get Started Free</a>
      </motion.div>
    </PageLayout>
  )
}

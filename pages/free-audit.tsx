import { motion } from 'framer-motion'
import PageLayout from '@/components/PageLayout'
import ContactForm from '@/components/ContactForm'
import { Search, BarChart2, Globe, CheckCircle2 } from 'lucide-react'

const included = [
  { icon: Globe,     title: 'Website Analysis',      desc: 'We review your current site (or note you have none) — speed, SEO, mobile, trust signals.' },
  { icon: Search,    title: 'Local Search Audit',     desc: 'We check where you rank for your top keywords and what your competitors are doing differently.' },
  { icon: BarChart2, title: 'Growth Opportunity Map', desc: 'A clear breakdown of the fastest wins available for your business right now.' },
]

export default function FreeAudit() {
  return (
    <PageLayout
      title="Free Growth Audit"
      description="Get a free, no-obligation growth audit for your local business. We'll analyse your website, local search presence, and growth opportunities."
    >
      <div className="grid lg:grid-cols-2 gap-12 items-start">
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6 }}>
          <div className="section-label mb-6 w-fit"><span className="glow-dot" />Free — No Strings</div>
          <h1 className="text-4xl font-black text-white mb-5 leading-tight">
            Your free
            <br />
            <span className="gradient-text-brand">growth audit.</span>
          </h1>
          <p className="text-white/50 text-base leading-relaxed mb-8">
            Before spending a penny, see exactly where your business stands online and what the fastest
            path to more customers looks like. No jargon, no sales pressure.
          </p>

          <div className="space-y-4">
            {included.map(({ icon: Icon, title, desc }) => (
              <div key={title} className="flex items-start gap-4 glass-card rounded-xl p-5">
                <div className="w-9 h-9 rounded-xl flex items-center justify-center flex-shrink-0"
                  style={{ background: 'rgba(99,102,241,0.1)', border: '1px solid rgba(99,102,241,0.2)' }}>
                  <Icon size={15} className="text-brand-400" />
                </div>
                <div>
                  <div className="text-white font-semibold text-sm mb-1">{title}</div>
                  <div className="text-white/45 text-sm">{desc}</div>
                </div>
              </div>
            ))}
          </div>

          <div className="mt-8 flex items-center gap-2 text-xs text-white/30">
            <CheckCircle2 size={13} className="text-brand-400" />
            Delivered within 48 hours · 100% free · No credit card
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, x: 30 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.7, delay: 0.1 }}
          className="relative rounded-2xl p-8"
          style={{
            background: 'rgba(13,13,26,0.9)',
            border: '1px solid rgba(255,255,255,0.07)',
            boxShadow: '0 8px 60px rgba(0,0,0,0.5), 0 0 0 1px rgba(99,102,241,0.07)',
          }}
        >
          <div className="absolute -inset-px rounded-2xl pointer-events-none"
            style={{ background: 'linear-gradient(135deg, rgba(99,102,241,0.07), transparent 60%)' }} />
          <div className="relative">
            <h2 className="text-white font-bold text-xl mb-1">Request Your Free Audit</h2>
            <p className="text-white/40 text-sm mb-6">We'll get back to you within 48 hours.</p>
            <ContactForm compact />
          </div>
        </motion.div>
      </div>
    </PageLayout>
  )
}

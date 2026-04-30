import { useRef } from 'react'
import { motion, useInView } from 'framer-motion'
import { CheckCircle2 } from 'lucide-react'
import ContactForm from '@/components/ContactForm'

const perks = [
  'Free website preview within 48 hours',
  'No credit card required',
  'Full market analysis included',
  'Cancel anytime — no contracts',
]

export default function CTA() {
  const ref = useRef(null)
  const inView = useInView(ref, { once: true, margin: '-80px' })

  return (
    <section id="contact" className="relative py-32 px-4 overflow-hidden">

      {/* Background */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute inset-0"
          style={{ background: 'radial-gradient(ellipse at 50% 0%, rgba(99,102,241,0.1) 0%, transparent 65%)' }} />
        <div className="absolute top-0 left-0 right-0 h-px"
          style={{ background: 'linear-gradient(90deg, transparent, rgba(99,102,241,0.4) 30%, rgba(168,85,247,0.4) 70%, transparent)' }} />
      </div>

      <div className="max-w-6xl mx-auto" ref={ref}>
        <div className="grid lg:grid-cols-2 gap-16 items-center">

          {/* Left — copy */}
          <div>
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={inView ? { opacity: 1, scale: 1 } : {}}
              transition={{ duration: 0.5 }}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-full mb-8"
              style={{ background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.25)' }}
            >
              <span className="w-1.5 h-1.5 rounded-full bg-red-400 animate-pulse"
                style={{ boxShadow: '0 0 6px rgba(239,68,68,0.8)' }} />
              <span className="text-xs font-semibold text-red-400 tracking-wide">
                Your competitors are upgrading right now
              </span>
            </motion.div>

            <motion.h2
              initial={{ opacity: 0, y: 30 }}
              animate={inView ? { opacity: 1, y: 0 } : {}}
              transition={{ duration: 0.7, delay: 0.1, ease: [0.16, 1, 0.3, 1] }}
              className="font-black tracking-tight text-white mb-6"
              style={{ fontSize: 'clamp(2rem, 4vw, 3.5rem)', lineHeight: 1.1 }}
            >
              Stop losing customers
              <br />
              <span className="gradient-text-brand">to businesses half your quality.</span>
            </motion.h2>

            <motion.p
              initial={{ opacity: 0, y: 20 }}
              animate={inView ? { opacity: 1, y: 0 } : {}}
              transition={{ duration: 0.6, delay: 0.2 }}
              className="text-white/45 text-lg mb-10 leading-relaxed"
            >
              We'll build a custom preview of your new website — free, no strings attached.
              See exactly what your business looks like when it's finally taken seriously online.
            </motion.p>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={inView ? { opacity: 1, y: 0 } : {}}
              transition={{ duration: 0.6, delay: 0.3 }}
              className="space-y-3"
            >
              {perks.map((perk) => (
                <div key={perk} className="flex items-center gap-3 text-sm text-white/50">
                  <CheckCircle2 size={15} className="text-brand-400 flex-shrink-0" />
                  {perk}
                </div>
              ))}
            </motion.div>
          </div>

          {/* Right — form */}
          <motion.div
            initial={{ opacity: 0, x: 40 }}
            animate={inView ? { opacity: 1, x: 0 } : {}}
            transition={{ duration: 0.7, delay: 0.15, ease: [0.16, 1, 0.3, 1] }}
            className="relative rounded-2xl p-8"
            style={{
              background: 'rgba(13,13,26,0.9)',
              border: '1px solid rgba(255,255,255,0.07)',
              boxShadow: '0 8px 60px rgba(0,0,0,0.6), 0 0 0 1px rgba(99,102,241,0.08)',
            }}
          >
            {/* Card glow */}
            <div className="absolute -inset-px rounded-2xl pointer-events-none"
              style={{
                background: 'linear-gradient(135deg, rgba(99,102,241,0.08), transparent 60%)',
              }} />

            <div className="relative">
              <h3 className="text-white font-bold text-xl mb-1">Get Your Free Preview</h3>
              <p className="text-white/40 text-sm mb-6">Fill in the form — we'll be in touch within 24h.</p>
              <ContactForm />
            </div>
          </motion.div>
        </div>
      </div>
    </section>
  )
}

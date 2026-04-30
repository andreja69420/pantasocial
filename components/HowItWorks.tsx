import { useRef } from 'react'
import { motion, useInView } from 'framer-motion'
import { Search, Layers, Rocket } from 'lucide-react'

const steps = [
  {
    step:  '01',
    icon:  Search,
    title: 'We Audit Your Online Presence',
    desc:  'We look at your current website, Google listing, social media, and local competitors. You get a clear picture of where you stand and exactly what needs to be fixed.',
    bullets: [
      'Website & SEO review',
      'Google Business Profile audit',
      'Local competitor analysis',
    ],
    accent: '#6366f1',
  },
  {
    step:  '02',
    icon:  Layers,
    title: 'We Build & Launch Everything',
    desc:  'We design your website, set up your Google Business Profile, launch your ads, and get your Instagram and Facebook looking sharp — all within the first two weeks.',
    bullets: [
      'Custom website live in 7 days',
      'Google & Facebook Ads setup',
      'Instagram & Facebook profiles',
    ],
    accent: '#a855f7',
  },
  {
    step:  '03',
    icon:  Rocket,
    title: 'Customers Start Finding You',
    desc:  'Once everything is live, people searching on Google and scrolling Instagram start seeing your business. More calls, more bookings, more revenue — every month.',
    bullets: [
      'Ranking higher on Google',
      'Consistent social media presence',
      'Monthly reporting & improvements',
    ],
    accent: '#06b6d4',
  },
]

export default function HowItWorks() {
  const ref = useRef(null)
  const inView = useInView(ref, { once: true })

  return (
    <section id="how-it-works" className="relative py-32 px-4 overflow-hidden">

      {/* Background accent */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] rounded-full opacity-5"
          style={{ background: 'radial-gradient(circle, #6366f1 0%, transparent 70%)' }} />
      </div>

      <div className="max-w-7xl mx-auto">

        {/* Header */}
        <div ref={ref} className="text-center mb-20">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={inView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.5 }}
            className="section-label mb-6 mx-auto w-fit"
          >
            <span className="glow-dot" />
            Process
          </motion.div>
          <motion.h2
            initial={{ opacity: 0, y: 20 }}
            animate={inView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.6, delay: 0.1 }}
            className="font-black tracking-tight text-white"
            style={{ fontSize: 'clamp(2rem, 4vw, 3.5rem)' }}
          >
            From zero to leads in{' '}
            <span className="gradient-text-brand">3 steps.</span>
          </motion.h2>
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={inView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="mt-4 text-white/40 text-lg"
          >
            No complexity. No long onboarding. Just results.
          </motion.p>
        </div>

        {/* Steps */}
        <div className="relative">

          {/* Connector line (desktop) */}
          <div className="hidden lg:block absolute top-12 left-0 right-0 h-px"
            style={{ background: 'linear-gradient(90deg, transparent, rgba(99,102,241,0.3) 20%, rgba(168,85,247,0.3) 50%, rgba(6,182,212,0.3) 80%, transparent)' }} />

          <div className="grid lg:grid-cols-3 gap-8">
            {steps.map((step, i) => {
              const Icon = step.icon
              return (
                <motion.div
                  key={step.step}
                  initial={{ opacity: 0, y: 50 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true, margin: '-80px' }}
                  transition={{ duration: 0.7, delay: i * 0.15, ease: [0.16, 1, 0.3, 1] }}
                  className="relative group"
                >
                  {/* Step indicator */}
                  <div className="flex items-center gap-4 mb-6">
                    <div className="relative w-12 h-12 rounded-2xl flex items-center justify-center flex-shrink-0 z-10"
                      style={{
                        background: `linear-gradient(135deg, ${step.accent}30, ${step.accent}10)`,
                        border: `1px solid ${step.accent}40`,
                      }}>
                      <Icon size={22} style={{ color: step.accent }} />
                      <div className="absolute -top-1 -right-1 w-5 h-5 rounded-full flex items-center justify-center text-[9px] font-bold text-white"
                        style={{ background: step.accent }}>
                        {i + 1}
                      </div>
                    </div>
                    <div className="text-5xl font-black opacity-[0.06] text-white select-none">{step.step}</div>
                  </div>

                  {/* Content card */}
                  <div className="glass-card rounded-2xl p-7 group-hover:border-white/10 transition-all duration-300"
                    style={{
                      boxShadow: '0 4px 30px rgba(0,0,0,0.4)',
                    }}>
                    <h3 className="text-white font-bold text-xl mb-3">{step.title}</h3>
                    <p className="text-white/50 text-sm leading-relaxed mb-5">{step.desc}</p>

                    <ul className="space-y-2.5">
                      {step.bullets.map((b) => (
                        <li key={b} className="flex items-center gap-3 text-sm text-white/60">
                          <div className="w-1.5 h-1.5 rounded-full flex-shrink-0"
                            style={{ background: step.accent, boxShadow: `0 0 6px ${step.accent}` }} />
                          {b}
                        </li>
                      ))}
                    </ul>
                  </div>

                  {/* Hover glow */}
                  <div className="absolute inset-0 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none"
                    style={{ boxShadow: `0 8px 50px ${step.accent}15` }} />
                </motion.div>
              )
            })}
          </div>
        </div>

        {/* Timeline note */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: '-60px' }}
          transition={{ duration: 0.6 }}
          className="mt-16 flex flex-col sm:flex-row items-center justify-center gap-8 text-center"
        >
          {[
            { label: 'Day 1–2',  text: 'Audit & strategy' },
            { label: 'Day 3–7',  text: 'Website design & build' },
            { label: 'Day 8–14', text: 'Google, ads & social launch' },
            { label: 'Day 15+',  text: 'Customers finding you' },
          ].map((item, i) => (
            <div key={item.label} className="flex items-center gap-3">
              {i > 0 && <div className="hidden sm:block w-8 h-px bg-white/10" />}
              <div>
                <div className="text-xs font-semibold text-brand-400 mb-1">{item.label}</div>
                <div className="text-xs text-white/40">{item.text}</div>
              </div>
            </div>
          ))}
        </motion.div>
      </div>
    </section>
  )
}

import { useRef } from 'react'
import { motion, useInView } from 'framer-motion'
import { Bot, Globe, Megaphone, BarChart3, ArrowRight } from 'lucide-react'

const solutions = [
  {
    icon:  Bot,
    label: 'Lead Generation',
    title: 'Get More Calls & Bookings',
    desc:  'We run Google Ads, Facebook Ads, and local outreach campaigns that put your business in front of people actively searching for what you offer — in your city, right now.',
    features: ['Google Ads management', 'Facebook & Instagram Ads', 'Local outreach campaigns', 'Lead follow-up & tracking'],
    accent: '#6366f1',
    grad:   'from-[#6366f1] to-[#8b5cf6]',
  },
  {
    icon:  Globe,
    label: 'Website Design',
    title: 'A Website That Actually Wins Customers',
    desc:  'We design and build fast, mobile-first websites that look professional, rank on Google, and turn visitors into paying customers. Live in 7 days.',
    features: ['Custom design, built for your brand', 'Mobile-first & fast-loading', 'SEO-ready from day one', 'Contact forms & click-to-call'],
    accent: '#a855f7',
    grad:   'from-[#a855f7] to-[#ec4899]',
  },
  {
    icon:  Megaphone,
    label: 'Social Media',
    title: 'Instagram & Facebook, Handled For You',
    desc:  'We create content, post consistently, manage your profiles, and grow your following on Instagram and Facebook — so your business stays visible and looks credible.',
    features: ['Instagram & Facebook management', 'Content creation & posting', 'Review & reputation management', 'Monthly performance recap'],
    accent: '#06b6d4',
    grad:   'from-[#06b6d4] to-[#6366f1]',
  },
  {
    icon:  BarChart3,
    label: 'SEO & Reporting',
    title: 'Rank Higher. See What\'s Working.',
    desc:  'We optimize your Google Business Profile, build local citations, and send you a plain-English monthly report showing your rankings, traffic, and where every lead came from.',
    features: ['Google Business Profile setup', 'Local SEO & citations', 'Keyword rank tracking', 'Monthly results reports'],
    accent: '#10b981',
    grad:   'from-[#10b981] to-[#06b6d4]',
  },
]

function SolutionCard({ s, index }: { s: typeof solutions[0]; index: number }) {
  const ref = useRef(null)
  const inView = useInView(ref, { once: true, margin: '-80px' })
  const Icon = s.icon

  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 50 }}
      animate={inView ? { opacity: 1, y: 0 } : {}}
      transition={{ duration: 0.7, delay: index * 0.1, ease: [0.16, 1, 0.3, 1] }}
      className="group relative glass-card rounded-2xl p-8 hover:border-white/10 transition-all duration-500"
      style={{ boxShadow: '0 4px 40px rgba(0,0,0,0.5)' }}
      whileHover={{ y: -6, transition: { duration: 0.25 } }}
    >
      {/* Icon */}
      <div className={`w-12 h-12 rounded-xl bg-gradient-to-br ${s.grad} flex items-center justify-center mb-6 shadow-lg`}>
        <Icon size={22} className="text-white" />
      </div>

      {/* Label */}
      <div className="text-xs font-semibold tracking-widest uppercase mb-3"
        style={{ color: s.accent }}>
        {s.label}
      </div>

      {/* Title */}
      <h3 className="text-white font-bold text-xl mb-3 leading-snug">{s.title}</h3>

      {/* Description */}
      <p className="text-white/50 text-sm leading-relaxed mb-6">{s.desc}</p>

      {/* Features */}
      <ul className="space-y-2.5">
        {s.features.map((f) => (
          <li key={f} className="flex items-center gap-3 text-sm text-white/60">
            <div className="w-1.5 h-1.5 rounded-full flex-shrink-0"
              style={{ background: s.accent, boxShadow: `0 0 6px ${s.accent}` }} />
            {f}
          </li>
        ))}
      </ul>

      {/* Hover glow */}
      <div className="absolute inset-0 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none"
        style={{ boxShadow: `0 0 40px ${s.accent}18, inset 0 1px 0 ${s.accent}15` }} />
    </motion.div>
  )
}

export default function Solution() {
  const ref = useRef(null)
  const inView = useInView(ref, { once: true })

  return (
    <section id="solution" className="relative py-32 px-4">
      {/* Divider glow */}
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-px h-32 bg-gradient-to-b from-transparent via-brand-500/40 to-transparent" />

      <div className="max-w-7xl mx-auto">

        {/* Header */}
        <div ref={ref} className="text-center mb-16">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={inView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.5 }}
            className="section-label mb-6 mx-auto w-fit"
          >
            <span className="glow-dot" />
            What We Do
          </motion.div>
          <motion.h2
            initial={{ opacity: 0, y: 20 }}
            animate={inView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.6, delay: 0.1 }}
            className="font-black tracking-tight text-white max-w-3xl mx-auto"
            style={{ fontSize: 'clamp(2rem, 4vw, 3.5rem)' }}
          >
            One system.
            <br />
            <span className="gradient-text-brand">Every growth lever covered.</span>
          </motion.h2>
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={inView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="mt-5 text-white/40 text-lg max-w-xl mx-auto"
          >
            We handle your website, Google, Instagram, Facebook, and ads —
            so you can focus on running your business.
          </motion.p>
        </div>

        {/* Cards */}
        <div className="grid sm:grid-cols-2 gap-5">
          {solutions.map((s, i) => (
            <SolutionCard key={s.label} s={s} index={i} />
          ))}
        </div>

        {/* CTA row */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: '-60px' }}
          transition={{ duration: 0.6 }}
          className="mt-12 flex justify-center"
        >
          <a href="#how-it-works" className="btn-secondary group">
            See How It Works
            <ArrowRight size={16} className="transition-transform group-hover:translate-x-1" />
          </a>
        </motion.div>
      </div>
    </section>
  )
}

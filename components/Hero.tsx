import { motion } from 'framer-motion'
import { ArrowRight, TrendingUp, Users, Zap, ChevronRight } from 'lucide-react'

const STATS = [
  { value: '3.2x', label: 'Avg. Lead Increase', icon: TrendingUp },
  { value: '400+', label: 'Businesses Powered',  icon: Users },
  { value: '98%',  label: 'Client Retention',    icon: Zap },
]

const INDUSTRIES = [
  'Plumbers', 'HVAC', 'Dentists', 'Lawyers', 'Contractors',
  'Restaurants', 'Auto Shops', 'Landscapers', 'Roofers', 'Salons',
  'Electricians', 'Chiropractors', 'Real Estate', 'Gyms', 'Cleaners',
]

export default function Hero() {
  return (
    <section className="relative min-h-screen flex flex-col items-center justify-center px-4 pt-24 pb-0 overflow-hidden">

      {/* Orbs */}
      <div className="absolute top-1/4 left-1/4 w-[600px] h-[600px] rounded-full pointer-events-none"
        style={{ background: 'radial-gradient(circle, rgba(99,102,241,0.12) 0%, transparent 70%)', filter: 'blur(60px)', animation: 'float 8s ease-in-out infinite' }} />
      <div className="absolute bottom-1/3 right-1/5 w-96 h-96 rounded-full pointer-events-none"
        style={{ background: 'radial-gradient(circle, rgba(168,85,247,0.1) 0%, transparent 70%)', filter: 'blur(50px)', animation: 'float 11s ease-in-out infinite reverse' }} />
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-64 h-64 rounded-full pointer-events-none"
        style={{ background: 'radial-gradient(circle, rgba(6,182,212,0.06) 0%, transparent 70%)', filter: 'blur(40px)', animation: 'float 14s ease-in-out infinite 3s' }} />

      {/* Badge */}
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6 }}
        className="section-label mb-8">
        <span className="glow-dot" />
        Website · SEO · Social Media · Google Ads
      </motion.div>

      {/* Headline */}
      <motion.h1
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, delay: 0.1, ease: [0.16, 1, 0.3, 1] }}
        className="text-center font-black leading-[1.05] tracking-tight max-w-5xl"
        style={{ fontSize: 'clamp(2.5rem, 7vw, 6rem)' }}
      >
        <span className="gradient-text-hero">Your website. Your Google.</span>
        <br />
        <span className="text-white">Your Instagram & Facebook.</span>
        <br />
        <span className="gradient-text-brand">All done for you.</span>
      </motion.h1>

      {/* Sub */}
      <motion.p
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.25 }}
        className="mt-6 text-center text-white/45 max-w-xl leading-relaxed"
        style={{ fontSize: 'clamp(1rem, 2vw, 1.2rem)' }}
      >
        We design your website, run your Google and Facebook ads, manage your Instagram,
        and get you ranking higher — so customers find you, not your competition.
      </motion.p>

      {/* CTAs */}
      <motion.div
        id="hero-cta"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.35 }}
        className="mt-10 flex flex-col sm:flex-row items-center gap-4"
      >
        <a href="#contact" className="btn-primary text-base px-8 py-4 group">
          Get My Free Website Preview
          <ArrowRight size={18} className="transition-transform group-hover:translate-x-1" />
        </a>
        <a href="#results" className="group flex items-center gap-2 text-base font-medium text-white/50 hover:text-white transition-colors duration-200 py-4 px-2">
          See Real Results
          <ChevronRight size={16} className="transition-transform group-hover:translate-x-1 opacity-50 group-hover:opacity-100" />
        </a>
      </motion.div>

      {/* Trust */}
      <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.5 }}
        className="mt-5 text-xs text-white/25 tracking-wide">
        Free website preview · No contracts · First results within 30 days
      </motion.p>

      {/* Stats bar */}
      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.7, delay: 0.55 }}
        className="mt-16 grid grid-cols-3 gap-px glass rounded-2xl overflow-hidden max-w-lg w-full cyber-corner"
        style={{ border: '1px solid rgba(255,255,255,0.06)' }}
      >
        {STATS.map(({ value, label, icon: Icon }) => (
          <div key={label} className="flex flex-col items-center py-5 px-4 hover:bg-white/[0.03] transition-colors group">
            <Icon size={16} className="text-brand-400 mb-2 opacity-60 group-hover:opacity-100 transition-opacity" />
            <span className="text-xl font-bold gradient-text-brand">{value}</span>
            <span className="text-[10px] text-white/35 mt-0.5 text-center leading-tight">{label}</span>
          </div>
        ))}
      </motion.div>

      {/* Industry ticker */}
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.9, duration: 0.8 }}
        className="mt-14 w-full max-w-4xl overflow-hidden">
        <p className="text-center text-[10px] text-white/15 tracking-widest uppercase mb-3">
          Serving local businesses across every industry
        </p>
        <div className="relative overflow-hidden"
          style={{ maskImage: 'linear-gradient(90deg, transparent, black 10%, black 90%, transparent)' }}>
          <div className="flex animate-marquee whitespace-nowrap">
            {[...INDUSTRIES, ...INDUSTRIES].map((name, i) => (
              <span key={i} className="inline-flex items-center gap-3 px-5 py-2 text-xs font-medium text-white/25">
                <span className="w-1 h-1 rounded-full bg-brand-500/40 flex-shrink-0" />
                {name}
              </span>
            ))}
          </div>
        </div>
      </motion.div>

      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 1.4 }}
        className="mt-10 mb-8 flex flex-col items-center">
        <div className="w-px h-10 bg-gradient-to-b from-white/10 to-transparent" />
      </motion.div>
    </section>
  )
}

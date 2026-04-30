import { useRef } from 'react'
import { motion, useInView } from 'framer-motion'
import { Globe, Bot, BarChart2, Cog, ChevronRight } from 'lucide-react'

const services = [
  {
    icon:   Globe,
    name:   'Web Design & Development',
    desc:   'High-converting, mobile-first websites built for speed, SEO, and trust. Designed to turn visitors into paying customers.',
    tags:   ['Next.js', 'SEO-Ready', '7-Day Delivery', 'Custom Design'],
    accent: '#6366f1',
    grad:   'from-[#6366f1]/20 to-transparent',
    border: 'rgba(99,102,241,0.2)',
    popular: false,
  },
  {
    icon:   Bot,
    name:   'Lead Generation',
    desc:   'We run Google Ads, Facebook Ads, and local outreach to bring qualified leads to your door. You take the calls — we keep them coming.',
    tags:   ['Google Ads', 'Facebook Ads', 'Local Outreach', 'Qualified Leads'],
    accent: '#a855f7',
    grad:   'from-[#a855f7]/20 to-transparent',
    border: 'rgba(168,85,247,0.3)',
    popular: true,
  },
  {
    icon:   BarChart2,
    name:   'SEO & Google Ads',
    desc:   'Dominate local search results. We optimize your Google Business Profile, build citations, and run high-ROI ad campaigns.',
    tags:   ['Google Business', 'Local SEO', 'PPC Campaigns', 'Rank Tracking'],
    accent: '#06b6d4',
    grad:   'from-[#06b6d4]/20 to-transparent',
    border: 'rgba(6,182,212,0.2)',
    popular: false,
  },
  {
    icon:   Cog,
    name:   'Full Online Presence',
    desc:   'Everything in one package — website, Google, Instagram, Facebook, ads, SEO, and monthly reporting. We manage it all so you never have to think about marketing again.',
    tags:   ['Done-For-You', 'All Channels', 'Monthly Reports', 'Priority Support'],
    accent: '#10b981',
    grad:   'from-[#10b981]/20 to-transparent',
    border: 'rgba(16,185,129,0.2)',
    popular: false,
  },
]

function ServiceCard({ service, index }: { service: typeof services[0]; index: number }) {
  const Icon = service.icon
  return (
    <motion.div
      initial={{ opacity: 0, y: 40 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-80px' }}
      transition={{ duration: 0.6, delay: index * 0.1, ease: [0.16, 1, 0.3, 1] }}
      className="relative group rounded-2xl overflow-hidden transition-all duration-300"
      style={{
        background: 'rgba(13,13,26,0.8)',
        border: `1px solid ${service.border}`,
        boxShadow: '0 4px 30px rgba(0,0,0,0.5)',
      }}
      whileHover={{ y: -6, transition: { duration: 0.25 } }}
    >
      {/* Popular badge */}
      {service.popular && (
        <div className="absolute top-0 left-0 right-0 h-0.5"
          style={{ background: `linear-gradient(90deg, transparent, ${service.accent}, transparent)` }} />
      )}
      {service.popular && (
        <div className="absolute top-4 right-4">
          <span className="text-[10px] font-bold px-2.5 py-1 rounded-full text-white"
            style={{ background: `${service.accent}30`, border: `1px solid ${service.accent}40` }}>
            MOST POPULAR
          </span>
        </div>
      )}

      {/* Gradient top */}
      <div className={`absolute top-0 left-0 right-0 h-40 bg-gradient-to-b ${service.grad} pointer-events-none`} />

      {/* flex-col so CTA always pins to bottom regardless of content height */}
      <div className="relative p-8 flex flex-col h-full">
        {/* Icon */}
        <div className="w-12 h-12 rounded-xl flex items-center justify-center mb-6 flex-shrink-0"
          style={{ background: `${service.accent}20`, border: `1px solid ${service.accent}30` }}>
          <Icon size={22} style={{ color: service.accent }} />
        </div>

        {/* Name */}
        <h3 className="text-white font-bold text-xl mb-4">{service.name}</h3>

        {/* Description — flex-1 pushes everything below it to the bottom */}
        <p className="text-white/50 text-sm leading-relaxed mb-6 flex-1">{service.desc}</p>

        {/* Tags */}
        <div className="flex flex-wrap gap-2 mb-6">
          {service.tags.map((tag) => (
            <span key={tag} className="text-[10px] font-medium px-2.5 py-1 rounded-lg text-white/50"
              style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)' }}>
              {tag}
            </span>
          ))}
        </div>

        {/* CTA — always at the same vertical position across all cards */}
        <a href="#contact"
          className="group/btn flex items-center justify-between w-full px-5 py-3 rounded-xl text-sm font-semibold text-white transition-all duration-300 mt-auto"
          style={{ background: `${service.accent}15`, border: `1px solid ${service.accent}30` }}
          onMouseEnter={(e) => {
            e.currentTarget.style.background = `${service.accent}25`
            e.currentTarget.style.borderColor = `${service.accent}50`
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.background = `${service.accent}15`
            e.currentTarget.style.borderColor = `${service.accent}30`
          }}
        >
          Get Free Audit
          <ChevronRight size={16} className="transition-transform group-hover/btn:translate-x-1" />
        </a>
      </div>

      {/* Hover glow */}
      <div className="absolute inset-0 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none"
        style={{ boxShadow: `0 0 50px ${service.accent}12` }} />
    </motion.div>
  )
}

export default function Services() {
  const ref = useRef(null)
  const inView = useInView(ref, { once: true })

  return (
    <section id="services" className="relative py-32 px-4">
      {/* Top divider */}
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-px h-24 bg-gradient-to-b from-transparent via-white/10 to-transparent" />

      <div className="max-w-7xl mx-auto">

        <div ref={ref} className="text-center mb-16">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={inView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.5 }}
            className="section-label mb-6 mx-auto w-fit"
          >
            <span className="glow-dot" />
            Services & Pricing
          </motion.div>
          <motion.h2
            initial={{ opacity: 0, y: 20 }}
            animate={inView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.6, delay: 0.1 }}
            className="font-black tracking-tight text-white"
            style={{ fontSize: 'clamp(2rem, 4vw, 3.5rem)' }}
          >
            Everything local businesses need{' '}
            <span className="gradient-text-brand">to get found online.</span>
          </motion.h2>
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={inView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="mt-4 text-white/40 text-lg max-w-xl mx-auto"
          >
            Pick one service or let us handle your entire online presence.
          </motion.p>
        </div>

        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {services.map((s, i) => (
            <ServiceCard key={s.name} service={s} index={i} />
          ))}
        </div>

        {/* Bottom note */}
        <motion.p
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ delay: 0.4 }}
          className="text-center text-white/25 text-sm mt-10"
        >
          All plans include a free strategy call · No long-term contracts · Cancel anytime
        </motion.p>
      </div>
    </section>
  )
}

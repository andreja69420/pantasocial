import { useRef } from 'react'
import { motion, useInView } from 'framer-motion'
import { Star, Quote } from 'lucide-react'

const testimonials = [
  {
    name:    'Marcus Webb',
    role:    'Owner, Webb HVAC Services',
    city:    'Chicago, IL',
    avatar:  'MW',
    rating:  5,
    text:    'PantaSocial completely transformed our business. Within 30 days of launching our new site and lead system, we were getting 3–4 qualified calls a day. Went from 0 online presence to #2 on Google local.',
    result:  '+340% more leads in 60 days',
    color:   '#6366f1',
  },
  {
    name:    'Priya Nair',
    role:    'Owner, Nair Dental Studio',
    city:    'Austin, TX',
    avatar:  'PN',
    rating:  5,
    text:    'I was skeptical at first, but the results speak for themselves. Our website looks like a Fortune 500 company\'s now, and the automated follow-up system books consultations while I sleep.',
    result:  '68 new patients in first 90 days',
    color:   '#a855f7',
  },
  {
    name:    'Carlos Mendez',
    role:    'Owner, Mendez Landscaping',
    city:    'Miami, FL',
    avatar:  'CM',
    rating:  5,
    text:    'Best investment I\'ve made in 10 years of business. The team at PantaSocial built a beautiful website, ran our ads, and set up email automation. Revenue doubled in 4 months.',
    result:  '2x revenue in 4 months',
    color:   '#06b6d4',
  },
  {
    name:    'Jennifer Kowalski',
    role:    'Owner, Kowalski Law Group',
    city:    'New York, NY',
    avatar:  'JK',
    rating:  5,
    text:    'As a law firm, we were hesitant to rely on digital marketing. PantaSocial changed our perspective completely. Professional, results-driven, and they actually understand local business needs.',
    result:  '22 new clients in 45 days',
    color:   '#10b981',
  },
  {
    name:    'Tony Batista',
    role:    'Owner, Batista Auto Repair',
    city:    'Phoenix, AZ',
    avatar:  'TB',
    rating:  5,
    text:    'We had a terrible website and no social media. Now we rank on the first page, have 200+ Google reviews, and our appointment calendar is always full. The team is incredibly responsive.',
    result:  '200+ new Google reviews',
    color:   '#f59e0b',
  },
  {
    name:    'Linda Forsythe',
    role:    'Owner, Forsythe Yoga Studio',
    city:    'Denver, CO',
    avatar:  'LF',
    rating:  5,
    text:    'PantaSocial helped me compete against the big gym chains. My studio now shows up before them on Google, and the lead nurturing emails have tripled my membership sign-ups.',
    result:  '3x membership sign-ups',
    color:   '#ec4899',
  },
]

function TestimonialCard({ t, index }: { t: typeof testimonials[0]; index: number }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 40 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-60px' }}
      transition={{ duration: 0.6, delay: index * 0.07, ease: [0.16, 1, 0.3, 1] }}
      className="group relative glass-card rounded-2xl p-7 flex flex-col gap-5 hover:border-white/10 transition-all duration-300"
      whileHover={{ y: -4, transition: { duration: 0.2 } }}
    >
      {/* Quote icon */}
      <Quote size={20} className="text-white/10 absolute top-5 right-5" />

      {/* Stars */}
      <div className="flex gap-1">
        {Array.from({ length: t.rating }).map((_, i) => (
          <Star key={i} size={13} className="fill-yellow-400 text-yellow-400" />
        ))}
      </div>

      {/* Text */}
      <p className="text-white/60 text-sm leading-relaxed flex-1">"{t.text}"</p>

      {/* Result pill */}
      <div className="flex items-center gap-2 px-3 py-2 rounded-xl self-start"
        style={{ background: `${t.color}15`, border: `1px solid ${t.color}30` }}>
        <div className="w-1.5 h-1.5 rounded-full" style={{ background: t.color, boxShadow: `0 0 5px ${t.color}` }} />
        <span className="text-xs font-semibold" style={{ color: t.color }}>{t.result}</span>
      </div>

      {/* Author */}
      <div className="flex items-center gap-3 pt-2 border-t border-white/[0.05]">
        <div className="w-9 h-9 rounded-full flex items-center justify-center text-xs font-bold text-white flex-shrink-0"
          style={{ background: `linear-gradient(135deg, ${t.color}60, ${t.color}30)`, border: `1px solid ${t.color}40` }}>
          {t.avatar}
        </div>
        <div>
          <div className="text-white font-semibold text-sm">{t.name}</div>
          <div className="text-white/40 text-xs">{t.role} · {t.city}</div>
        </div>
      </div>

      {/* Hover glow */}
      <div className="absolute inset-0 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-400 pointer-events-none"
        style={{ boxShadow: `0 0 35px ${t.color}10` }} />
    </motion.div>
  )
}

export default function Testimonials() {
  const ref = useRef(null)
  const inView = useInView(ref, { once: true })

  return (
    <section id="testimonials" className="relative py-32 px-4">
      <div className="max-w-7xl mx-auto">

        <div ref={ref} className="text-center mb-16">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={inView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.5 }}
            className="section-label mb-6 mx-auto w-fit"
          >
            <span className="glow-dot bg-yellow-400" style={{ boxShadow: '0 0 6px rgba(250,204,21,0.8)' }} />
            Client Results
          </motion.div>
          <motion.h2
            initial={{ opacity: 0, y: 20 }}
            animate={inView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.6, delay: 0.1 }}
            className="font-black tracking-tight text-white"
            style={{ fontSize: 'clamp(2rem, 4vw, 3.5rem)' }}
          >
            Trusted by local businesses
            <br />
            <span className="gradient-text-brand">across the country.</span>
          </motion.h2>
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={inView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="mt-4 text-white/40 text-lg"
          >
            Real owners. Real results. No hype.
          </motion.p>
        </div>

        {/* Aggregate score */}
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          whileInView={{ opacity: 1, scale: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
          className="flex flex-col sm:flex-row items-center justify-center gap-8 mb-14 glass rounded-2xl py-6 px-8 max-w-xl mx-auto"
        >
          <div className="text-center">
            <div className="text-4xl font-black text-white">4.9</div>
            <div className="flex gap-0.5 justify-center mt-1">
              {[...Array(5)].map((_, i) => <Star key={i} size={12} className="fill-yellow-400 text-yellow-400" />)}
            </div>
            <div className="text-xs text-white/30 mt-1">Average rating</div>
          </div>
          <div className="w-px h-12 bg-white/10 hidden sm:block" />
          <div className="text-center">
            <div className="text-4xl font-black text-white">400+</div>
            <div className="text-xs text-white/30 mt-2">Local businesses served</div>
          </div>
          <div className="w-px h-12 bg-white/10 hidden sm:block" />
          <div className="text-center">
            <div className="text-4xl font-black text-white">98%</div>
            <div className="text-xs text-white/30 mt-2">Client retention rate</div>
          </div>
        </motion.div>

        {/* Grid */}
        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {testimonials.map((t, i) => (
            <TestimonialCard key={t.name} t={t} index={i} />
          ))}
        </div>
      </div>
    </section>
  )
}

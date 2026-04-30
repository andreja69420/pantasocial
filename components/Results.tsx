import { useRef, useState, useEffect } from 'react'
import { motion, useInView, animate } from 'framer-motion'
import { TrendingUp, Users, Star, DollarSign, ArrowUpRight, Activity } from 'lucide-react'

function CountUp({ to, suffix = '', prefix = '' }: { to: number; suffix?: string; prefix?: string }) {
  const [val, setVal] = useState(0)
  const ref = useRef(null)
  const inView = useInView(ref, { once: true })

  useEffect(() => {
    if (!inView) return
    const controls = animate(0, to, {
      duration: 1.8,
      ease: 'easeOut',
      onUpdate: (v) => setVal(Math.round(v)),
    })
    return controls.stop
  }, [inView, to])

  return (
    <span ref={ref}>
      {prefix}{val.toLocaleString()}{suffix}
    </span>
  )
}

const metrics = [
  { icon: Users,       label: 'Leads Generated',    value: 2847,  suffix: '',   prefix: '',  accent: '#6366f1' },
  { icon: TrendingUp,  label: 'Avg. Conversion Lift', value: 312, suffix: '%',  prefix: '+', accent: '#10b981' },
  { icon: DollarSign,  label: 'Revenue Unlocked',   value: 1200,  suffix: 'K+', prefix: '$', accent: '#a855f7' },
  { icon: Star,        label: 'Google Reviews Won',  value: 5890,  suffix: '',   prefix: '',  accent: '#f59e0b' },
]

const weeklyData = [40, 65, 55, 80, 75, 95, 110, 130, 115, 145, 140, 165]

function MiniChart({ data, color }: { data: number[]; color: string }) {
  const max = Math.max(...data)
  const width = 220
  const height = 60

  const points = data
    .map((v, i) => `${(i / (data.length - 1)) * width},${height - (v / max) * height}`)
    .join(' ')

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-12" preserveAspectRatio="none">
      <defs>
        <linearGradient id={`grad-${color.replace('#', '')}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.3" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      <polyline
        points={points}
        fill="none"
        stroke={color}
        strokeWidth="2"
        strokeLinejoin="round"
        strokeLinecap="round"
      />
    </svg>
  )
}

const leadsData = [
  { month: 'Jan', leads: 18, close: 12 },
  { month: 'Feb', leads: 24, close: 16 },
  { month: 'Mar', leads: 31, close: 22 },
  { month: 'Apr', leads: 28, close: 20 },
  { month: 'May', leads: 42, close: 31 },
  { month: 'Jun', leads: 57, close: 43 },
]

export default function Results() {
  const ref = useRef(null)
  const inView = useInView(ref, { once: true })

  return (
    <section id="results" className="relative py-32 px-4">
      <div className="max-w-7xl mx-auto">

        {/* Header */}
        <div ref={ref} className="text-center mb-16">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={inView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.5 }}
            className="section-label mb-6 mx-auto w-fit"
          >
            <span className="glow-dot bg-green-500" style={{ boxShadow: '0 0 6px rgba(16,185,129,0.8)' }} />
            Real Results
          </motion.div>
          <motion.h2
            initial={{ opacity: 0, y: 20 }}
            animate={inView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.6, delay: 0.1 }}
            className="font-black tracking-tight text-white"
            style={{ fontSize: 'clamp(2rem, 4vw, 3.5rem)' }}
          >
            Numbers that{' '}
            <span className="gradient-text-brand">speak for themselves.</span>
          </motion.h2>
        </div>

        {/* Metric cards */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
          {metrics.map(({ icon: Icon, label, value, suffix, prefix, accent }, i) => (
            <motion.div
              key={label}
              initial={{ opacity: 0, scale: 0.95 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true, margin: '-60px' }}
              transition={{ duration: 0.5, delay: i * 0.08 }}
              className="glass-card rounded-2xl p-6 group hover:border-white/10 transition-all duration-300"
              whileHover={{ y: -4, transition: { duration: 0.2 } }}
            >
              <div className="flex items-center justify-between mb-4">
                <div className="w-9 h-9 rounded-xl flex items-center justify-center"
                  style={{ background: `${accent}20`, border: `1px solid ${accent}30` }}>
                  <Icon size={17} style={{ color: accent }} />
                </div>
                <ArrowUpRight size={14} className="text-white/20 group-hover:text-white/50 transition-colors" />
              </div>
              <div className="text-3xl font-black text-white mb-1">
                <CountUp to={value} suffix={suffix} prefix={prefix} />
              </div>
              <div className="text-xs text-white/40 leading-tight">{label}</div>
              <div className="mt-3">
                <MiniChart data={weeklyData} color={accent} />
              </div>
            </motion.div>
          ))}
        </div>

        {/* Dashboard mock */}
        <div className="grid lg:grid-cols-5 gap-4">

          {/* Main chart */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: '-60px' }}
            transition={{ duration: 0.7 }}
            className="lg:col-span-3 glass-card rounded-2xl p-6"
          >
            <div className="flex items-center justify-between mb-6">
              <div>
                <div className="text-white font-semibold mb-0.5">Lead Volume</div>
                <div className="text-xs text-white/30">Last 6 months · Plumber Pro LLC</div>
              </div>
              <div className="flex items-center gap-2">
                <div className="flex items-center gap-1.5 text-xs text-white/40">
                  <div className="w-2 h-2 rounded-full bg-brand-500" />
                  Leads
                </div>
                <div className="flex items-center gap-1.5 text-xs text-white/40">
                  <div className="w-2 h-2 rounded-full bg-green-500" />
                  Closed
                </div>
              </div>
            </div>

            {/* Bar chart */}
            <div className="flex items-end gap-3 h-32">
              {leadsData.map(({ month, leads, close }) => (
                <div key={month} className="flex-1 flex flex-col items-center gap-1">
                  <div className="w-full flex items-end gap-0.5 h-24">
                    <motion.div
                      className="flex-1 rounded-t-lg"
                      style={{ background: 'linear-gradient(to top, #6366f1, #8b5cf6)' }}
                      initial={{ height: 0 }}
                      whileInView={{ height: `${(leads / 57) * 100}%` }}
                      viewport={{ once: true }}
                      transition={{ duration: 0.8, delay: 0.1, ease: [0.16, 1, 0.3, 1] }}
                    />
                    <motion.div
                      className="flex-1 rounded-t-lg"
                      style={{ background: 'linear-gradient(to top, #10b981, #34d399)' }}
                      initial={{ height: 0 }}
                      whileInView={{ height: `${(close / 57) * 100}%` }}
                      viewport={{ once: true }}
                      transition={{ duration: 0.8, delay: 0.2, ease: [0.16, 1, 0.3, 1] }}
                    />
                  </div>
                  <span className="text-[10px] text-white/30">{month}</span>
                </div>
              ))}
            </div>

            <div className="mt-4 pt-4 border-t border-white/[0.06] flex items-center gap-2">
              <Activity size={13} className="text-green-400" />
              <span className="text-xs text-green-400 font-medium">+217% increase over 6 months</span>
            </div>
          </motion.div>

          {/* Side panels */}
          <div className="lg:col-span-2 flex flex-col gap-4">

            {/* Live feed */}
            <motion.div
              initial={{ opacity: 0, x: 30 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true, margin: '-60px' }}
              transition={{ duration: 0.6, delay: 0.15 }}
              className="glass-card rounded-2xl p-5 flex-1"
            >
              <div className="flex items-center justify-between mb-4">
                <span className="text-white font-semibold text-sm">Live Activity</span>
                <span className="flex items-center gap-1.5 text-[10px] text-green-400">
                  <span className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />
                  LIVE
                </span>
              </div>

              <div className="space-y-3">
                {[
                  { action: 'New lead captured',      name: 'Mike T.',    time: '2m ago',  color: '#6366f1' },
                  { action: 'Meeting booked',          name: 'Sarah K.',   time: '11m ago', color: '#10b981' },
                  { action: 'Email sequence started',  name: 'James R.',   time: '34m ago', color: '#a855f7' },
                  { action: 'Review request sent',     name: 'Linda M.',   time: '1h ago',  color: '#f59e0b' },
                ].map((item) => (
                  <div key={item.name} className="flex items-center gap-3">
                    <div className="w-2 h-2 rounded-full flex-shrink-0"
                      style={{ background: item.color, boxShadow: `0 0 5px ${item.color}` }} />
                    <div className="flex-1 min-w-0">
                      <div className="text-xs text-white/70 truncate">{item.action}</div>
                      <div className="text-[10px] text-white/30">{item.name}</div>
                    </div>
                    <div className="text-[10px] text-white/25 flex-shrink-0">{item.time}</div>
                  </div>
                ))}
              </div>
            </motion.div>

            {/* Score card */}
            <motion.div
              initial={{ opacity: 0, x: 30 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true, margin: '-60px' }}
              transition={{ duration: 0.6, delay: 0.25 }}
              className="glass-card rounded-2xl p-5"
            >
              <div className="text-xs text-white/40 mb-3">Website Performance</div>
              <div className="flex items-center gap-3 mb-2">
                <div className="text-4xl font-black text-green-400">97</div>
                <div>
                  <div className="text-white text-sm font-semibold">PageSpeed Score</div>
                  <div className="text-xs text-white/30">Avg. client website</div>
                </div>
              </div>
              <div className="space-y-2 mt-3">
                {[
                  { label: 'Performance', val: 97, color: '#10b981' },
                  { label: 'SEO',          val: 100, color: '#6366f1' },
                  { label: 'Accessibility', val: 95, color: '#a855f7' },
                ].map((row) => (
                  <div key={row.label} className="flex items-center gap-3">
                    <span className="text-[10px] text-white/40 w-20">{row.label}</span>
                    <div className="flex-1 h-1 rounded-full bg-white/[0.06] overflow-hidden">
                      <motion.div
                        className="h-full rounded-full"
                        style={{ background: row.color }}
                        initial={{ width: 0 }}
                        whileInView={{ width: `${row.val}%` }}
                        viewport={{ once: true }}
                        transition={{ duration: 1, ease: [0.16, 1, 0.3, 1] }}
                      />
                    </div>
                    <span className="text-[10px] text-white/50 w-6 text-right">{row.val}</span>
                  </div>
                ))}
              </div>
            </motion.div>
          </div>
        </div>
      </div>
    </section>
  )
}

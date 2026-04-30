import { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { ArrowUpRight, TrendingUp } from 'lucide-react'
import { useRealtimeMetrics } from '@/hooks/useRealtimeMetrics'

// ── Animated number counter ────────────────────────────────────────────────
function AnimatedNumber({ value }: { value: number }) {
  const [displayed, setDisplayed] = useState(value)
  const prevRef = useRef(value)
  useEffect(() => {
    if (value === prevRef.current) return
    const start = prevRef.current, end = value, frames = 24
    let frame = 0
    const id = setInterval(() => {
      frame++
      const t = 1 - Math.pow(1 - frame / frames, 3)
      setDisplayed(Math.round(start + (end - start) * t))
      if (frame >= frames) { clearInterval(id); prevRef.current = end }
    }, 16)
    return () => clearInterval(id)
  }, [value])
  return <>{displayed.toLocaleString()}</>
}

// ── 7-bar weekly sparkline ─────────────────────────────────────────────────
function MiniChart({ seed }: { seed: number }) {
  const bars = Array.from({ length: 7 }, (_, i) => {
    const s = ((seed + i * 37) * 1234567) & 0xfffff
    return 20 + (s % 70)
  })
  const todayIdx = (new Date().getDay() + 6) % 7
  return (
    <div className="flex items-end gap-[3px] h-5 mt-1">
      {bars.map((h, i) => (
        <div key={i} className="flex-1 rounded-sm"
          style={{
            height: `${h}%`,
            background: i === todayIdx
              ? 'linear-gradient(180deg, #a855f7 0%, #6366f1 100%)'
              : i < todayIdx
              ? 'rgba(99,102,241,0.32)'
              : 'rgba(255,255,255,0.06)',
          }} />
      ))}
    </div>
  )
}

// ── Growth Dashboard widget ────────────────────────────────────────────────
function DashboardWidget() {
  const m     = useRealtimeMetrics()
  const [pulse, setPulse] = useState(false)
  const prev  = useRef(m.leads)
  const pct   = Math.min(Math.round((m.leads / m.leadsGoal) * 100), 100)
  const month = new Date().toLocaleString('en', { month: 'long' })
  const seed  = new Date().getFullYear() * 10000 + (new Date().getMonth() + 1) * 100 + new Date().getDate()

  useEffect(() => {
    if (m.leads === prev.current) return
    setPulse(true)
    prev.current = m.leads
    const t = setTimeout(() => setPulse(false), 900)
    return () => clearTimeout(t)
  }, [m.leads])

  return (
    <div style={{
      background: 'rgba(5,5,12,0.97)',
      border: '1px solid rgba(99,102,241,0.22)',
      borderRadius: 18,
      boxShadow: '0 16px 56px rgba(0,0,0,0.9), 0 0 0 1px rgba(99,102,241,0.07), 0 0 40px rgba(99,102,241,0.07)',
      width: 242,
      backdropFilter: 'blur(24px)',
      WebkitBackdropFilter: 'blur(24px)',
    }}>

      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-white/[0.05]">
        <div className="flex items-center gap-2">
          <div className="w-5 h-5 rounded-md flex items-center justify-center flex-shrink-0"
            style={{ background: 'rgba(99,102,241,0.15)', border: '1px solid rgba(99,102,241,0.28)' }}>
            <TrendingUp size={10} className="text-indigo-400" />
          </div>
          <span className="text-[10px] font-semibold text-white/45 tracking-widest uppercase">Growth Dashboard</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="relative flex h-1.5 w-1.5">
            <span className="animate-ping absolute inset-0 rounded-full bg-emerald-400 opacity-70" />
            <span className="relative rounded-full h-1.5 w-1.5 bg-emerald-400 block" />
          </span>
          <span className="text-[9px] font-bold text-emerald-400 tracking-widest">LIVE</span>
        </div>
      </div>

      <div className="px-4 pt-3.5 pb-3 space-y-3">

        {/* Online now */}
        <div className="flex items-center">
          <span className="flex-1 text-[10px] text-white/35 truncate">Browsing right now</span>
          <div className="flex items-center gap-1.5 flex-shrink-0">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 flex-shrink-0"
              style={{ boxShadow: '0 0 6px rgba(52,211,153,0.9)' }} />
            <span className="text-[11px] font-bold text-white tabular-nums">{m.onlineNow} visitors</span>
          </div>
        </div>

        <div className="h-px bg-white/[0.05]" />

        {/* Leads this month */}
        <div>
          <div className="flex items-center mb-2">
            <span className="flex-1 text-[10px] text-white/35 truncate">New leads · {month}</span>
            <div className="flex items-baseline gap-1 flex-shrink-0">
              {m.leadsToday > 0 && (
                <span className="text-[9px] font-semibold text-emerald-400 flex items-center gap-0.5">
                  <ArrowUpRight size={8} />+{m.leadsToday}
                </span>
              )}
              <motion.span
                animate={pulse ? { scale: [1, 1.14, 1] } : {}}
                transition={{ duration: 0.35 }}
                className="text-[14px] font-black text-white tabular-nums ml-1"
              >
                <AnimatedNumber value={m.leads} />
              </motion.span>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <div className="flex-1 h-[3px] rounded-full bg-white/[0.06] overflow-hidden">
              <motion.div className="h-full rounded-full"
                style={{ background: 'linear-gradient(90deg, #6366f1, #a855f7)' }}
                initial={{ width: 0 }}
                animate={{ width: `${pct}%` }}
                transition={{ duration: 1.4, ease: [0.16, 1, 0.3, 1] }} />
            </div>
            <span className="text-[9px] text-white/22 tabular-nums flex-shrink-0">{pct}% of goal</span>
          </div>
        </div>

        <div className="h-px bg-white/[0.05]" />

        {/* Site visits + sparkline */}
        <div>
          <div className="flex items-center">
            <span className="flex-1 text-[10px] text-white/35 truncate">Site visits · this week</span>
            <div className="flex items-baseline gap-1 flex-shrink-0">
              <span className="text-[9px] font-semibold text-emerald-400 flex items-center">
                <ArrowUpRight size={8} />+{m.visitsTrend}%
              </span>
              <span className="text-[13px] font-bold text-white tabular-nums ml-0.5">
                <AnimatedNumber value={m.visits} />
              </span>
            </div>
          </div>
          <MiniChart seed={seed} />
        </div>

        <div className="h-px bg-white/[0.05]" />

        {/* Conversion rate */}
        <div className="flex items-center">
          <span className="flex-1 text-[10px] text-white/35 truncate">Conversion rate</span>
          <span className="text-[12px] font-bold flex-shrink-0 tabular-nums"
            style={{ color: '#a78bfa' }}>{m.conversion}%</span>
        </div>

      </div>

      {/* Last lead footer */}
      <div className="px-4 py-3 border-t border-white/[0.05] flex items-center gap-2.5">
        <div className="w-6 h-6 rounded-full flex-shrink-0 flex items-center justify-center text-[9px] font-black text-white select-none"
          style={{ background: 'linear-gradient(135deg, #6366f1, #a855f7)' }}>
          {m.lastLead.name[0]}
        </div>
        <div className="min-w-0">
          <p className="text-[10px] text-white/60 font-semibold leading-tight truncate">
            {m.lastLead.name}
            <span className="text-white/30 font-normal"> · {m.lastLead.biz}</span>
          </p>
          <p className="text-[9px] text-white/25 leading-tight mt-0.5">
            {m.lastLead.action} · {m.minutesAgo}m ago
          </p>
        </div>
      </div>

    </div>
  )
}

// ── Container: fixed bottom-left from first load ───────────────────────────
export default function FixedWidgets() {
  const [footerVisible, setFooterVisible] = useState(false)
  const [mounted,       setMounted]       = useState(false)

  useEffect(() => { setMounted(true) }, [])

  useEffect(() => {
    const footer = document.querySelector('footer')
    if (!footer) return
    const obs = new IntersectionObserver(
      ([e]) => setFooterVisible(e.isIntersecting),
      { threshold: 0.05 }
    )
    obs.observe(footer)
    return () => obs.disconnect()
  }, [])

  if (!mounted) return null

  return (
    <AnimatePresence>
      {!footerVisible && (
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: 16 }}
          transition={{ duration: 0.5, delay: 0.3, ease: [0.16, 1, 0.3, 1] }}
          className="fixed bottom-6 left-5 z-40"
        >
          <DashboardWidget />
        </motion.div>
      )}
    </AnimatePresence>
  )
}

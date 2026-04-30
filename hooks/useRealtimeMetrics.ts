import { useState, useEffect } from 'react'

// ── Deterministic pseudo-random from integer seed (mulberry32) ──────────────
function rand(seed: number) {
  let s = seed | 0
  s = (s + 0x6d2b79f5) | 0
  let t = Math.imul(s ^ (s >>> 15), 1 | s)
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296
}

// ── Build a stable day-seed so every visitor on the same day gets the same
//    baseline numbers.  Changes once per calendar day. ──────────────────────
function dayInt(d: Date) {
  return d.getFullYear() * 10000 + (d.getMonth() + 1) * 100 + d.getDate()
}
function monthInt(d: Date) {
  return d.getFullYear() * 100 + d.getMonth() + 1
}

// ── Simulate "business-hour weight" (peak 9-18, quiet at night) ─────────────
function businessProgress(d: Date): number {
  const h = d.getHours() + d.getMinutes() / 60
  // Piecewise: ramp up 6-10, plateau 10-18, ramp down 18-22, flat night
  if (h < 6)  return 0.04
  if (h < 10) return 0.04 + ((h - 6) / 4) * 0.55
  if (h < 18) return 0.59 + ((h - 10) / 8) * 0.35
  if (h < 22) return 0.94 + ((h - 18) / 4) * 0.06
  return 1.0
}

// ── Pool of realistic names / businesses ────────────────────────────────────
export const LEAD_POOL = [
  { name: 'Marcus W.',   biz: 'Webb HVAC Services',    action: 'requested a free audit' },
  { name: 'Sarah K.',    biz: 'Nair Dental Studio',    action: 'submitted a contact form' },
  { name: 'Carlos M.',   biz: 'Mendez Landscaping',    action: 'requested website preview' },
  { name: 'Jennifer K.', biz: 'Kowalski Law Group',    action: 'booked a free audit' },
  { name: 'Tony B.',     biz: 'Batista Auto Repair',   action: 'requested a free audit' },
  { name: 'Linda F.',    biz: 'Forsythe Yoga Studio',  action: 'submitted a contact form' },
  { name: 'David W.',    biz: 'Wade Plumbing',         action: 'requested website preview' },
  { name: 'Emma C.',     biz: 'Chiro & Wellness Co.',  action: 'requested a free audit' },
  { name: 'Mike T.',     biz: 'Taylor Roofing LLC',    action: 'submitted a contact form' },
  { name: 'Ana R.',      biz: 'Bright Clean Services', action: 'booked a free audit' },
  { name: 'James P.',    biz: 'Parker Electric',       action: 'requested website preview' },
  { name: 'Lisa M.',     biz: 'Main Street Salon',     action: 'requested a free audit' },
]

export interface Metrics {
  leads: number            // leads captured this month so far
  leadsGoal: number        // monthly goal
  leadsToday: number       // leads captured today
  visits: number           // site visits this week
  visitsTrend: number      // % change vs last week (always positive, realistic)
  conversion: string       // e.g. "12.4"
  onlineNow: number        // 1–6 visitors right now
  lastLead: typeof LEAD_POOL[0]
  minutesAgo: number       // how long ago the last lead came in
}

function compute(now: Date): Metrics {
  const ds  = dayInt(now)
  const ms  = monthInt(now)

  // ── Monthly lead goal (55–79, stable per month) ─────────────────────────
  const leadsGoal = 55 + Math.floor(rand(ms * 137) * 25)

  // ── Leads so far this month ──────────────────────────────────────────────
  // Each calendar day contributes a deterministic slice based on day-seed.
  // We accumulate up to yesterday, then add today's in-progress amount.
  const daysInMonth = new Date(now.getFullYear(), now.getMonth() + 1, 0).getDate()
  const leadsPerDay = leadsGoal / daysInMonth          // float, avg per day
  let monthTotal = 0
  for (let d = 1; d < now.getDate(); d++) {
    // Each past day gets slightly varied count (0.6×–1.4× of average)
    const variation = 0.6 + rand((ms * 31 + d) * 97) * 0.8
    monthTotal += Math.round(leadsPerDay * variation)
  }
  // Today's partial: business-hour progress × daily average × variation
  const todayVariation = 0.7 + rand(ds * 53) * 0.6
  const leadsToday = Math.floor(businessProgress(now) * leadsPerDay * todayVariation)
  const leads = monthTotal + leadsToday

  // ── Site visits this week ────────────────────────────────────────────────
  // Weekly target: 900–1600 visits
  const weeklyTarget = 900 + Math.floor(rand(ms * 61) * 700)
  const monday = new Date(now)
  monday.setDate(now.getDate() - ((now.getDay() + 6) % 7))
  monday.setHours(0, 0, 0, 0)
  const weekElapsed = Math.max(0, (now.getTime() - monday.getTime()) / (7 * 24 * 3600 * 1000))
  const visits = Math.floor(weeklyTarget * weekElapsed * (0.8 + rand(ds * 11) * 0.4))

  // ── Visits trend vs last week (always 3–27% improvement) ─────────────────
  const visitsTrend = 3 + Math.floor(rand(ms * 73) * 24)

  // ── Conversion rate (10.8–14.6 %, stable per month) ─────────────────────
  const conversion = (10.8 + rand(ms * 29) * 3.8).toFixed(1)

  // ── Online now: 1–6, changes every ~90 s, deterministic ─────────────────
  const bucket90s = Math.floor(now.getTime() / 90000)
  const onlineNow = 1 + Math.floor(rand((bucket90s * 19 + ds) * 41) * 6)

  // ── Last lead: cycles every 6 minutes, deterministic ────────────────────
  const bucket6m   = Math.floor(now.getTime() / (6 * 60 * 1000))
  const lastLead   = LEAD_POOL[bucket6m % LEAD_POOL.length]
  const minutesAgo = Math.floor((now.getTime() % (6 * 60 * 1000)) / 60000) + 1

  return { leads, leadsGoal, leadsToday, visits, visitsTrend, conversion, onlineNow, lastLead, minutesAgo }
}

// ── Hook: recalculates every 60 s, always increasing ────────────────────────
export function useRealtimeMetrics(): Metrics {
  const [metrics, setMetrics] = useState<Metrics>(() => compute(new Date()))

  useEffect(() => {
    const id = setInterval(() => setMetrics(compute(new Date())), 60_000)
    return () => clearInterval(id)
  }, [])

  return metrics
}

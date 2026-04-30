import { useState, FormEvent } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Send, CheckCircle2, AlertCircle, Loader2, ChevronDown } from 'lucide-react'

const SERVICES = [
  'Web Design & Development',
  'Lead Generation',
  'SEO & Google Ads',
  'Full Automation System',
  'Not sure — need advice',
]

type Status = 'idle' | 'loading' | 'success' | 'error'

interface Props {
  compact?: boolean
}

export default function ContactForm({ compact = false }: Props) {
  const [status, setStatus]   = useState<Status>('idle')
  const [errorMsg, setErrorMsg] = useState('')
  const [form, setForm] = useState({
    name: '', business: '', email: '', phone: '', service: '', message: '',
  })

  const set = (key: keyof typeof form) => (
    e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>
  ) => setForm(f => ({ ...f, [key]: e.target.value }))

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault()
    setStatus('loading')
    setErrorMsg('')

    try {
      const res = await fetch('/api/contact', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify(form),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.error || 'Something went wrong')
      setStatus('success')
    } catch (err: unknown) {
      setStatus('error')
      setErrorMsg(err instanceof Error ? err.message : 'Something went wrong')
    }
  }

  const inputBase =
    'w-full bg-white/[0.03] border border-white/[0.08] rounded-xl px-4 py-3 text-white text-sm placeholder-white/25 outline-none transition-all duration-200 focus:border-brand-500/50 focus:bg-white/[0.05] focus:shadow-[0_0_0_3px_rgba(99,102,241,0.08)]'

  if (status === 'success') {
    return (
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        className="flex flex-col items-center justify-center py-12 text-center gap-4"
      >
        <div className="w-16 h-16 rounded-full flex items-center justify-center"
          style={{ background: 'rgba(16,185,129,0.1)', border: '1px solid rgba(16,185,129,0.3)' }}>
          <CheckCircle2 size={28} className="text-green-400" />
        </div>
        <h3 className="text-white font-bold text-xl">Message received!</h3>
        <p className="text-white/50 text-sm max-w-xs">
          We'll review your submission and get back to you within 24 hours.
        </p>
        <button
          onClick={() => { setStatus('idle'); setForm({ name: '', business: '', email: '', phone: '', service: '', message: '' }) }}
          className="text-xs text-brand-400 hover:text-brand-300 underline underline-offset-4 transition-colors mt-2"
        >
          Send another message
        </button>
      </motion.div>
    )
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      {/* Row 1 */}
      <div className={`grid ${compact ? 'grid-cols-1' : 'sm:grid-cols-2'} gap-4`}>
        <div className="space-y-1.5">
          <label className="text-xs font-medium text-white/40 tracking-wide uppercase">Your Name *</label>
          <input
            type="text" required value={form.name} onChange={set('name')}
            placeholder="John Smith" className={inputBase}
          />
        </div>
        <div className="space-y-1.5">
          <label className="text-xs font-medium text-white/40 tracking-wide uppercase">Business Name *</label>
          <input
            type="text" required value={form.business} onChange={set('business')}
            placeholder="Smith Plumbing LLC" className={inputBase}
          />
        </div>
      </div>

      {/* Row 2 */}
      <div className={`grid ${compact ? 'grid-cols-1' : 'sm:grid-cols-2'} gap-4`}>
        <div className="space-y-1.5">
          <label className="text-xs font-medium text-white/40 tracking-wide uppercase">Email *</label>
          <input
            type="email" required value={form.email} onChange={set('email')}
            placeholder="john@smithplumbing.com" className={inputBase}
          />
        </div>
        <div className="space-y-1.5">
          <label className="text-xs font-medium text-white/40 tracking-wide uppercase">Phone</label>
          <input
            type="tel" value={form.phone} onChange={set('phone')}
            placeholder="+1 (555) 000-0000" className={inputBase}
          />
        </div>
      </div>

      {/* Service */}
      <div className="space-y-1.5">
        <label className="text-xs font-medium text-white/40 tracking-wide uppercase">I'm interested in</label>
        <div className="relative">
          <select
            value={form.service} onChange={set('service')}
            className={`${inputBase} appearance-none cursor-pointer pr-10`}
            style={{ background: 'rgba(255,255,255,0.03)' }}
          >
            <option value="" className="bg-[#0d0d1a]">Select a service...</option>
            {SERVICES.map(s => <option key={s} value={s} className="bg-[#0d0d1a]">{s}</option>)}
          </select>
          <ChevronDown size={14} className="absolute right-3.5 top-1/2 -translate-y-1/2 text-white/30 pointer-events-none" />
        </div>
      </div>

      {/* Message */}
      <div className="space-y-1.5">
        <label className="text-xs font-medium text-white/40 tracking-wide uppercase">Message *</label>
        <textarea
          required rows={compact ? 3 : 5} value={form.message} onChange={set('message')}
          placeholder="Tell us about your business and what you're looking to achieve..."
          className={`${inputBase} resize-none`}
        />
      </div>

      {/* Error */}
      <AnimatePresence>
        {status === 'error' && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="flex items-center gap-3 px-4 py-3 rounded-xl"
            style={{ background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.25)' }}
          >
            <AlertCircle size={14} className="text-red-400 flex-shrink-0" />
            <span className="text-red-400 text-sm">{errorMsg}</span>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Submit */}
      <motion.button
        type="submit"
        disabled={status === 'loading'}
        className="btn-primary w-full py-4 text-base font-semibold justify-center disabled:opacity-60 disabled:cursor-not-allowed"
        whileHover={status !== 'loading' ? { scale: 1.01 } : {}}
        whileTap={status !== 'loading' ? { scale: 0.99 } : {}}
      >
        {status === 'loading' ? (
          <>
            <Loader2 size={18} className="animate-spin" />
            Sending…
          </>
        ) : (
          <>
            Send Message
            <Send size={16} />
          </>
        )}
      </motion.button>

      <p className="text-center text-xs text-white/20">
        We respond within 24 hours · andreja@pantasocial.com
      </p>
    </form>
  )
}

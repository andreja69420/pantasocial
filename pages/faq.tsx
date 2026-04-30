import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import PageLayout from '@/components/PageLayout'
import { ChevronDown } from 'lucide-react'

const faqs = [
  {
    q: 'How quickly will I see results?',
    a: 'Most clients see their website live within 7 days. Lead generation systems typically start producing results within 14–30 days of launch. We set clear expectations upfront based on your industry and location.',
  },
  {
    q: 'Do I need to sign a long-term contract?',
    a: 'No. All of our monthly services are month-to-month. We earn your business every month based on results — not locked-in contracts. The only exception is custom website builds, which are one-time projects.',
  },
  {
    q: 'Will I own my website?',
    a: 'Yes, completely. Once your website is built and paid for, you own it entirely — the code, the domain, all of it. We have no lock-in on the website itself.',
  },
  {
    q: 'What industries do you work with?',
    a: 'We specialise in local service businesses: plumbers, HVAC, dentists, lawyers, contractors, restaurants, auto shops, landscapers, salons, gyms, electricians, chiropractors, real estate, and more. If you serve a local market, we can help.',
  },
  {
    q: 'How is PantaSocial different from other agencies?',
    a: 'We\'re a focused, hands-on operation — not a bloated agency with 50 account managers. You work directly with the team. We move faster, charge less, and stay accountable to actual numbers, not vanity metrics.',
  },
  {
    q: 'What do I need to get started?',
    a: 'Just fill out our contact form. We\'ll schedule a quick discovery call, learn about your business, and deliver a free website preview within 48 hours. No payment required to see what\'s possible.',
  },
  {
    q: 'Can you help me get more Google reviews?',
    a: 'Yes — review generation is part of our automation system. We set up automated follow-up messages that ask satisfied customers for Google reviews at exactly the right moment, dramatically increasing your rating over time.',
  },
]

function FAQItem({ q, a, index }: { q: string; a: string; index: number }) {
  const [open, setOpen] = useState(false)
  return (
    <motion.div
      initial={{ opacity: 0, y: 15 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.4, delay: index * 0.04 }}
      className="border-b border-white/[0.06] last:border-0"
    >
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between py-5 text-left gap-4 group"
      >
        <span className="text-white font-medium text-sm sm:text-base group-hover:text-brand-300 transition-colors">{q}</span>
        <motion.div
          animate={{ rotate: open ? 180 : 0 }}
          transition={{ duration: 0.25 }}
          className="flex-shrink-0"
        >
          <ChevronDown size={16} className="text-white/30" />
        </motion.div>
      </button>
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
            className="overflow-hidden"
          >
            <p className="pb-5 text-white/50 text-sm leading-relaxed">{a}</p>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  )
}

export default function FAQ() {
  return (
    <PageLayout
      title="FAQ"
      description="Frequently asked questions about PantaSocial LLC's services, pricing, timelines, and how we work."
    >
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6 }} className="mb-12">
        <div className="section-label mb-6 w-fit"><span className="glow-dot" />FAQ</div>
        <h1 className="text-5xl font-black text-white mb-4">
          Frequently asked
          <br />
          <span className="gradient-text-brand">questions.</span>
        </h1>
        <p className="text-white/50 text-lg">
          Can't find what you're looking for?{' '}
          <a href="/contact" className="text-brand-400 hover:text-brand-300 underline underline-offset-4 transition-colors">
            Send us a message.
          </a>
        </p>
      </motion.div>

      <div className="glass-card rounded-2xl px-6 sm:px-8 mb-12">
        {faqs.map((faq, i) => (
          <FAQItem key={i} q={faq.q} a={faq.a} index={i} />
        ))}
      </div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        className="rounded-2xl p-8 text-center"
        style={{ background: 'linear-gradient(135deg, rgba(99,102,241,0.08), rgba(168,85,247,0.06))', border: '1px solid rgba(99,102,241,0.15)' }}
      >
        <h3 className="text-white font-bold text-xl mb-2">Still have questions?</h3>
        <p className="text-white/40 text-sm mb-5">We're happy to answer anything — no obligation.</p>
        <a href="/contact" className="btn-primary inline-flex">Talk to Us</a>
      </motion.div>
    </PageLayout>
  )
}

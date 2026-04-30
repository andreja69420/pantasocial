import { motion } from 'framer-motion'
import PageLayout from '@/components/PageLayout'
import { Clock } from 'lucide-react'

export default function CaseStudies() {
  return (
    <PageLayout title="Case Studies" description="Real results from real local businesses that grew with PantaSocial LLC.">
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="text-center py-24">
        <div className="w-16 h-16 rounded-2xl flex items-center justify-center mx-auto mb-6"
          style={{ background: 'rgba(99,102,241,0.1)', border: '1px solid rgba(99,102,241,0.2)' }}>
          <Clock size={24} className="text-brand-400" />
        </div>
        <h1 className="text-4xl font-black text-white mb-4">Case Studies Coming Soon</h1>
        <p className="text-white/50 text-lg mb-8 max-w-md mx-auto">
          We're documenting our clients' growth stories. Check back soon — or reach out to hear results directly.
        </p>
        <a href="/contact" className="btn-primary inline-flex">Talk to Us Instead</a>
      </motion.div>
    </PageLayout>
  )
}

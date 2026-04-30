import { motion } from 'framer-motion'
import PageLayout from '@/components/PageLayout'
import { Rss } from 'lucide-react'

export default function Blog() {
  return (
    <PageLayout title="Blog" description="Growth tips, marketing insights, and local business strategies from PantaSocial LLC.">
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="text-center py-24">
        <div className="w-16 h-16 rounded-2xl flex items-center justify-center mx-auto mb-6"
          style={{ background: 'rgba(99,102,241,0.1)', border: '1px solid rgba(99,102,241,0.2)' }}>
          <Rss size={24} className="text-brand-400" />
        </div>
        <h1 className="text-4xl font-black text-white mb-4">Blog Coming Soon</h1>
        <p className="text-white/50 text-lg mb-8 max-w-md mx-auto">
          We're writing growth guides for local business owners. Subscribe to be notified when we launch.
        </p>
        <a href="/contact" className="btn-primary inline-flex">Get Notified</a>
      </motion.div>
    </PageLayout>
  )
}

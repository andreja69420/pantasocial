import { motion } from 'framer-motion'
import PageLayout from '@/components/PageLayout'
import ContactForm from '@/components/ContactForm'
import { Mail, MapPin, Clock } from 'lucide-react'

export default function Contact() {
  return (
    <PageLayout
      title="Contact Us"
      description="Get in touch with PantaSocial LLC. We'll build a free website preview for your business and show you what automated growth looks like."
    >
      <div className="grid lg:grid-cols-5 gap-12">

        {/* Left info */}
        <div className="lg:col-span-2">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <div className="section-label mb-6 w-fit">
              <span className="glow-dot" />
              Get In Touch
            </div>
            <h1 className="text-4xl font-black text-white mb-4 leading-tight">
              Let's build your
              <br />
              <span className="gradient-text-brand">growth engine.</span>
            </h1>
            <p className="text-white/50 text-base leading-relaxed mb-10">
              Tell us about your business. We'll come back with a free website preview,
              a market analysis, and a clear plan for your growth — no pitch, no pressure.
            </p>

            <div className="space-y-5">
              {[
                { icon: Mail,    label: 'Email',     value: 'andreja@pantasocial.com', href: 'mailto:andreja@pantasocial.com' },
                { icon: Clock,   label: 'Response',  value: 'Within 24 hours',         href: null },
                { icon: MapPin,  label: 'Serving',   value: 'Businesses Nationwide',   href: null },
              ].map(({ icon: Icon, label, value, href }) => (
                <div key={label} className="flex items-start gap-4">
                  <div className="w-9 h-9 rounded-xl flex items-center justify-center flex-shrink-0 mt-0.5"
                    style={{ background: 'rgba(99,102,241,0.1)', border: '1px solid rgba(99,102,241,0.2)' }}>
                    <Icon size={15} className="text-brand-400" />
                  </div>
                  <div>
                    <div className="text-xs text-white/30 uppercase tracking-wide mb-0.5">{label}</div>
                    {href
                      ? <a href={href} className="text-white text-sm hover:text-brand-400 transition-colors">{value}</a>
                      : <div className="text-white text-sm">{value}</div>
                    }
                  </div>
                </div>
              ))}
            </div>
          </motion.div>
        </div>

        {/* Right form */}
        <motion.div
          initial={{ opacity: 0, x: 30 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.7, delay: 0.1 }}
          className="lg:col-span-3 relative rounded-2xl p-8"
          style={{
            background: 'rgba(13,13,26,0.9)',
            border: '1px solid rgba(255,255,255,0.07)',
            boxShadow: '0 8px 60px rgba(0,0,0,0.5), 0 0 0 1px rgba(99,102,241,0.07)',
          }}
        >
          <div className="absolute -inset-px rounded-2xl pointer-events-none"
            style={{ background: 'linear-gradient(135deg, rgba(99,102,241,0.07), transparent 60%)' }} />
          <div className="relative">
            <h2 className="text-white font-bold text-xl mb-1">Send Us a Message</h2>
            <p className="text-white/40 text-sm mb-6">We'll get back to you within 24 hours.</p>
            <ContactForm />
          </div>
        </motion.div>
      </div>
    </PageLayout>
  )
}

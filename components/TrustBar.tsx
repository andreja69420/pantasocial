import { motion } from 'framer-motion'
import { Wrench, Wind, Smile, Scale, HardHat, UtensilsCrossed, Car, Scissors, Leaf, Dumbbell } from 'lucide-react'

const industries = [
  { icon: Wrench,          label: 'Plumbing & HVAC' },
  { icon: Smile,           label: 'Dental Clinics' },
  { icon: Scale,           label: 'Law Firms' },
  { icon: HardHat,         label: 'Contractors' },
  { icon: UtensilsCrossed, label: 'Restaurants' },
  { icon: Car,             label: 'Auto Services' },
  { icon: Scissors,        label: 'Salons & Spas' },
  { icon: Leaf,            label: 'Landscaping' },
  { icon: Dumbbell,        label: 'Gyms & Fitness' },
  { icon: Wind,            label: 'Cleaning Services' },
]

export default function TrustBar() {
  return (
    <section className="relative py-16 px-4 overflow-hidden">

      {/* Separator */}
      <div className="absolute top-0 left-0 right-0 h-px"
        style={{ background: 'linear-gradient(90deg, transparent, rgba(255,255,255,0.06) 30%, rgba(255,255,255,0.06) 70%, transparent)' }} />
      <div className="absolute bottom-0 left-0 right-0 h-px"
        style={{ background: 'linear-gradient(90deg, transparent, rgba(255,255,255,0.06) 30%, rgba(255,255,255,0.06) 70%, transparent)' }} />

      <div className="max-w-7xl mx-auto">
        <motion.p
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="text-center text-[10px] text-white/20 tracking-widest uppercase mb-8"
        >
          Trusted across every local industry
        </motion.p>

        <motion.div
          initial={{ opacity: 0, y: 10 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.7 }}
          className="flex flex-wrap items-center justify-center gap-3"
        >
          {industries.map(({ icon: Icon, label }, i) => (
            <motion.div
              key={label}
              initial={{ opacity: 0, scale: 0.9 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true }}
              transition={{ duration: 0.4, delay: i * 0.04 }}
              className="group flex items-center gap-2.5 px-4 py-2.5 rounded-xl transition-all duration-300 cursor-default"
              style={{
                background: 'rgba(255,255,255,0.02)',
                border: '1px solid rgba(255,255,255,0.05)',
              }}
              whileHover={{
                backgroundColor: 'rgba(99,102,241,0.07)',
                borderColor: 'rgba(99,102,241,0.2)',
                transition: { duration: 0.15 },
              }}
            >
              <Icon size={14} className="text-white/30 group-hover:text-brand-400 transition-colors duration-200" />
              <span className="text-xs text-white/30 group-hover:text-white/60 transition-colors duration-200 font-medium">
                {label}
              </span>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  )
}

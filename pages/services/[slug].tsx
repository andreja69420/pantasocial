import { GetStaticPaths, GetStaticProps } from 'next'
import { motion } from 'framer-motion'
import PageLayout from '@/components/PageLayout'
import ContactForm from '@/components/ContactForm'
import { Globe, Bot, BarChart2, Cog, CheckCircle2, ArrowRight } from 'lucide-react'

const SERVICES: Record<string, {
  icon: React.ElementType
  name: string
  tagline: string
  description: string
  accent: string
  features: string[]
  faqs: { q: string; a: string }[]
}> = {
  'web-design': {
    icon:    Globe,
    name:    'Web Design & Development',
    tagline: 'Websites that turn visitors into paying customers.',
    accent:  '#6366f1',
    description: 'We design and build fast, conversion-optimised websites that look premium and rank on Google. Every site is mobile-first, SEO-ready, and delivered in 7 days.',
    features: [
      'Custom design — no templates',
      'Mobile-first, fully responsive',
      'PageSpeed score 95+',
      'Local SEO architecture',
      'Contact forms and booking integrations',
      'SSL, CDN, and performance hosting',
      'Delivered in 7 business days',
      'Revisions included',
    ],
    faqs: [
      { q: 'How long does a website build take?', a: 'We deliver most websites within 7 business days of receiving your content and approval on the design direction.' },
      { q: 'Do I own the website?', a: 'Yes — fully. Once built and paid for, the website, code, and domain are entirely yours.' },
      { q: 'What if I need changes later?', a: 'Minor changes are included in the first 30 days. After that, we offer affordable maintenance plans.' },
    ],
  },
  'lead-generation': {
    icon:    Bot,
    name:    'Lead Generation',
    tagline: 'Qualified leads delivered to your inbox, automatically.',
    accent:  '#a855f7',
    description: 'Our automated system finds high-intent prospects in your local market, reaches out on your behalf, and books qualified leads directly into your calendar — 24/7.',
    features: [
      'Automated prospect discovery',
      'Smart outreach sequences',
      'Calendar booking integration',
      'Lead scoring & qualification',
      'CRM integration',
      'Daily lead reports',
      'A/B tested messaging',
      'Monthly optimisation',
    ],
    faqs: [
      { q: 'How many leads can I expect?', a: 'It depends on your market and industry, but most clients see 10–40 qualified prospects per month within the first 30 days.' },
      { q: 'What markets do you cover?', a: 'We cover the entire United States and Canada. International markets available on request.' },
      { q: 'Is this spam?', a: 'No. We target only highly relevant prospects using compliant, permission-respecting outreach methods.' },
    ],
  },
  'seo-ads': {
    icon:    BarChart2,
    name:    'SEO & Google Ads',
    tagline: 'Dominate local search. Get found before your competitors.',
    accent:  '#06b6d4',
    description: 'We optimise your Google Business Profile, build local citations, and run high-ROI Google Ads campaigns that put you in front of customers actively searching for your services.',
    features: [
      'Google Business Profile optimisation',
      'Local citation building',
      'Keyword research & strategy',
      'On-page SEO',
      'Google Ads campaign management',
      'Monthly rank tracking',
      'Competitor gap analysis',
      'Review generation campaigns',
    ],
    faqs: [
      { q: 'How long does SEO take?', a: 'Local SEO typically shows meaningful results within 60–90 days. Google Ads produce results from day one.' },
      { q: 'Do you manage the ad spend?', a: 'We manage the campaigns. Ad spend is separate and paid directly to Google — you keep full control.' },
      { q: 'What\'s included in Google Business optimisation?', a: 'Complete profile build-out, photo optimisation, category selection, Q&A setup, and ongoing post management.' },
    ],
  },
  'automation': {
    icon:    Cog,
    name:    'Full Automation System',
    tagline: 'Your entire growth engine — built and run for you.',
    accent:  '#10b981',
    description: 'The complete PantaSocial system: website, lead generation, SEO, social media, review management, and follow-up automation — all connected, all running automatically.',
    features: [
      'Everything in all other services',
      'Email & SMS automation sequences',
      'Review generation system',
      'Social media management',
      'Re-engagement campaigns',
      'Weekly performance reports',
      'Dedicated account manager',
      'Priority support',
    ],
    faqs: [
      { q: 'How is this different from individual services?', a: 'Everything is connected. Your website, lead gen, SEO, and follow-up all talk to each other — creating compounding results.' },
      { q: 'Is there a setup fee?', a: 'There is a one-time onboarding fee that covers system setup and website build. Ongoing monthly pricing is competitive with individual service bundles.' },
      { q: 'How much time will I need to spend?', a: 'Almost none. That\'s the point. You approve the strategy upfront — we handle execution and report back weekly.' },
    ],
  },
}

interface Props {
  slug: string
}

export default function ServicePage({ slug }: Props) {
  const service = SERVICES[slug]
  if (!service) return null

  const Icon = service.icon

  return (
    <PageLayout
      title={service.name}
      description={service.description}
    >
      {/* Hero */}
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6 }} className="mb-16">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-12 h-12 rounded-xl flex items-center justify-center"
            style={{ background: `${service.accent}20`, border: `1px solid ${service.accent}30` }}>
            <Icon size={22} style={{ color: service.accent }} />
          </div>
          <div className="section-label w-fit"><span className="glow-dot" />Service</div>
        </div>
        <h1 className="text-5xl font-black text-white mb-4 leading-tight">{service.name}</h1>
        <p className="text-xl font-medium mb-4" style={{ color: service.accent }}>{service.tagline}</p>
        <p className="text-white/50 text-lg leading-relaxed max-w-2xl">{service.description}</p>
      </motion.div>

      <div className="grid lg:grid-cols-2 gap-12 mb-16">
        {/* Features */}
        <motion.div initial={{ opacity: 0, x: -20 }} whileInView={{ opacity: 1, x: 0 }} viewport={{ once: true }} transition={{ duration: 0.6 }}>
          <h2 className="text-2xl font-bold text-white mb-6">What's included</h2>
          <div className="space-y-3">
            {service.features.map((f) => (
              <div key={f} className="flex items-center gap-3 py-3 border-b border-white/[0.05]">
                <CheckCircle2 size={15} style={{ color: service.accent }} className="flex-shrink-0" />
                <span className="text-white/70 text-sm">{f}</span>
              </div>
            ))}
          </div>
        </motion.div>

        {/* FAQ */}
        <motion.div initial={{ opacity: 0, x: 20 }} whileInView={{ opacity: 1, x: 0 }} viewport={{ once: true }} transition={{ duration: 0.6 }}>
          <h2 className="text-2xl font-bold text-white mb-6">Common questions</h2>
          <div className="space-y-4">
            {service.faqs.map(({ q, a }) => (
              <div key={q} className="glass-card rounded-xl p-5">
                <div className="text-white font-semibold text-sm mb-2">{q}</div>
                <div className="text-white/50 text-sm leading-relaxed">{a}</div>
              </div>
            ))}
          </div>
        </motion.div>
      </div>

      {/* CTA */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        className="rounded-2xl p-10"
        style={{
          background: `linear-gradient(135deg, ${service.accent}10, rgba(0,0,0,0))`,
          border: `1px solid ${service.accent}25`,
        }}
      >
        <div className="grid lg:grid-cols-2 gap-10 items-center">
          <div>
            <h3 className="text-white font-bold text-2xl mb-3">Ready to get started?</h3>
            <p className="text-white/50 text-sm leading-relaxed mb-4">
              Fill in the form and we'll come back within 24 hours with a free preview and a clear proposal.
            </p>
            <div className="flex items-center gap-2 text-xs text-white/30">
              <CheckCircle2 size={12} style={{ color: service.accent }} />
              Free preview · No credit card · No contracts
            </div>
          </div>
          <ContactForm compact />
        </div>
      </motion.div>
    </PageLayout>
  )
}

export const getStaticPaths: GetStaticPaths = async () => ({
  paths: Object.keys(SERVICES).map(slug => ({ params: { slug } })),
  fallback: false,
})

export const getStaticProps: GetStaticProps<Props> = async ({ params }) => ({
  props: { slug: params!.slug as string },
})

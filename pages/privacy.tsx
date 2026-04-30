import { motion } from 'framer-motion'
import PageLayout from '@/components/PageLayout'

export default function Privacy() {
  return (
    <PageLayout
      title="Privacy Policy"
      description="PantaSocial LLC privacy policy — how we collect, use, and protect your information."
    >
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6 }}>
        <div className="section-label mb-6 w-fit"><span className="glow-dot" />Legal</div>
        <h1 className="text-5xl font-black text-white mb-4">Privacy Policy</h1>
        <p className="text-white/40 text-sm mb-12">Last updated: January 2026</p>

        <div className="glass-card rounded-2xl p-8 sm:p-10 space-y-10 text-white/60 leading-relaxed text-sm">

          {[
            {
              title: '1. Information We Collect',
              body: 'We collect information you provide directly to us when you fill out our contact form, including your name, business name, email address, phone number, and message content. We do not collect any payment information on this website.',
            },
            {
              title: '2. How We Use Your Information',
              body: 'We use your information solely to respond to your enquiry, deliver the services you request, and communicate with you about your project. We do not sell, rent, or share your personal information with third parties for marketing purposes.',
            },
            {
              title: '3. Data Storage',
              body: 'Contact form submissions are transmitted directly to our business email inbox. We do not store submissions in a database. Your information is retained only as long as necessary to fulfil the purpose for which it was collected.',
            },
            {
              title: '4. Cookies',
              body: 'This website uses minimal, essential cookies only. We do not use tracking cookies, advertising cookies, or analytics that share your data with third parties.',
            },
            {
              title: '5. Third-Party Services',
              body: 'Our website may use Google Fonts for typography. This involves a request to Google\'s servers that may log your IP address. We do not use third-party analytics, advertising, or retargeting services.',
            },
            {
              title: '6. Your Rights',
              body: 'You have the right to request access to, correction of, or deletion of any personal information we hold about you. To exercise these rights, contact us at andreja@pantasocial.com.',
            },
            {
              title: '7. Changes to This Policy',
              body: 'We may update this policy occasionally. Changes will be posted on this page with an updated date. Continued use of our website after changes constitutes acceptance of the updated policy.',
            },
            {
              title: '8. Contact',
              body: 'For any privacy-related questions, contact Andreja Pantic at andreja@pantasocial.com.',
            },
          ].map(({ title, body }) => (
            <div key={title}>
              <h2 className="text-white font-bold text-lg mb-3">{title}</h2>
              <p>{body}</p>
            </div>
          ))}
        </div>
      </motion.div>
    </PageLayout>
  )
}

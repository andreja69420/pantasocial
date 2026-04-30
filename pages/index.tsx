import Head from 'next/head'
import { motion } from 'framer-motion'
import Navbar from '@/components/Navbar'
import Hero from '@/components/Hero'
import Problem from '@/components/Problem'
import Solution from '@/components/Solution'
import HowItWorks from '@/components/HowItWorks'
import Results from '@/components/Results'
import Services from '@/components/Services'
import Testimonials from '@/components/Testimonials'
import CTA from '@/components/CTA'
import Footer from '@/components/Footer'
import FloatingCTA from '@/components/FloatingCTA'
import TrustBar from '@/components/TrustBar'
import FixedWidgets from '@/components/FixedWidgets'

export default function Home() {
  return (
    <>
      <Head>
        <title>PantaSocial LLC — Automated Growth for Local Businesses</title>
        <meta name="description" content="We turn local businesses into automated client machines. Lead generation, conversion-focused websites, and marketing automation — built and managed for you." />
        <meta name="viewport" content="width=device-width, initial-scale=1" />

        {/* OG */}
        <meta property="og:title" content="PantaSocial LLC — Automated Growth for Local Businesses" />
        <meta property="og:description" content="Lead generation · Conversion websites · Marketing automation. We build your growth engine." />
        <meta property="og:type" content="website" />

        {/* Twitter */}
        <meta name="twitter:card" content="summary_large_image" />
        <meta name="twitter:title" content="PantaSocial LLC" />
        <meta name="twitter:description" content="Automated growth systems for local businesses." />

        {/* Schema */}
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{
            __html: JSON.stringify({
              '@context': 'https://schema.org',
              '@type': 'LocalBusiness',
              name: 'PantaSocial LLC',
              founder: { '@type': 'Person', name: 'Andreja Pantic' },
              description: 'Automated lead generation, web design, SEO, and marketing automation for local businesses.',
              email: 'andreja@pantasocial.com',
              priceRange: '$$',
              serviceType: ['Lead Generation', 'Web Design', 'SEO', 'Marketing Automation'],
            }),
          }}
        />
      </Head>

      {/* Fixed mesh background */}
      <div className="mesh-bg" aria-hidden="true" />

      {/* Grid overlay */}
      <div className="fixed inset-0 grid-overlay opacity-100 pointer-events-none z-0" aria-hidden="true" />

      {/* Page entry animation */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.4 }}
        className="relative z-10"
      >
        <Navbar />

        <main>
          <Hero />
          <TrustBar />
          <Problem />
          <Solution />
          <HowItWorks />
          <Results />
          <Services />
          <Testimonials />
          <CTA />
        </main>

        <Footer />
      </motion.div>

      <FloatingCTA />
      <FixedWidgets />
    </>
  )
}

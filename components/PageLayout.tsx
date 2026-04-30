import Head from 'next/head'
import { motion } from 'framer-motion'
import Navbar from '@/components/Navbar'
import Footer from '@/components/Footer'
import FloatingCTA from '@/components/FloatingCTA'

interface Props {
  title: string
  description: string
  children: React.ReactNode
}

export default function PageLayout({ title, description, children }: Props) {
  return (
    <>
      <Head>
        <title>{title} — PantaSocial LLC</title>
        <meta name="description" content={description} />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </Head>

      <div className="mesh-bg" aria-hidden="true" />
      <div className="fixed inset-0 grid-overlay pointer-events-none z-0" aria-hidden="true" />

      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.35 }}
        className="relative z-10"
      >
        <Navbar />
        <main className="pt-28 pb-16 px-4 min-h-screen">
          <div className="max-w-4xl mx-auto">
            {children}
          </div>
        </main>
        <Footer />
      </motion.div>

      <FloatingCTA />
    </>
  )
}

import { useEffect } from 'react'
import { useRouter } from 'next/router'

export default function HowItWorks() {
  const router = useRouter()
  useEffect(() => { router.replace('/#how-it-works') }, [router])
  return null
}

import { motion } from 'framer-motion'
import { Zap, Mail, Twitter, Linkedin, Instagram } from 'lucide-react'

const links = {
  Services: [
    { label: 'Web Design',        href: '/services/web-design' },
    { label: 'Lead Generation',   href: '/services/lead-generation' },
    { label: 'SEO & Google Ads',  href: '/services/seo-ads' },
    { label: 'Automation System', href: '/services/automation' },
  ],
  Company: [
    { label: 'About Us',     href: '/about' },
    { label: 'How It Works', href: '/how-it-works' },
    { label: 'Case Studies', href: '/case-studies' },
    { label: 'Blog',         href: '/blog' },
  ],
  Support: [
    { label: 'Contact',         href: '/contact' },
    { label: 'Free Audit',      href: '/free-audit' },
    { label: 'FAQ',             href: '/faq' },
    { label: 'Privacy Policy',  href: '/privacy' },
  ],
}

export default function Footer() {
  return (
    <footer className="relative border-t border-white/[0.05] pt-16 pb-8 px-4">

      {/* Top glow line */}
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-64 h-px"
        style={{ background: 'linear-gradient(90deg, transparent, rgba(99,102,241,0.5), transparent)' }} />

      <div className="max-w-7xl mx-auto">
        <div className="grid sm:grid-cols-2 lg:grid-cols-5 gap-10 mb-14">

          {/* Brand column */}
          <div className="lg:col-span-2">
            <div className="flex items-center gap-2.5 mb-5">
              <div className="w-8 h-8 rounded-xl flex items-center justify-center"
                style={{ background: 'linear-gradient(135deg, #6366f1, #a855f7)' }}>
                <Zap size={16} className="text-white" />
              </div>
              <span className="text-white font-bold text-lg">
                Panta<span className="gradient-text-brand">Social</span>
              </span>
            </div>

            <p className="text-white/40 text-sm leading-relaxed max-w-xs mb-6">
              Automated growth systems for local businesses. We build your online presence,
              generate leads, and automate your marketing — so you can focus on what you do best.
            </p>

            <div className="flex items-center gap-3 mb-6">
              <div className="w-9 h-9 rounded-full flex items-center justify-center text-xs font-bold text-white"
                style={{ background: 'linear-gradient(135deg, #6366f1, #a855f7)' }}>
                AP
              </div>
              <div>
                <div className="text-white text-sm font-semibold">Andreja Pantic</div>
                <div className="text-white/30 text-xs">Founder, PantaSocial LLC</div>
              </div>
            </div>

            <div className="flex items-center gap-3">
              {[
                { icon: Twitter,   href: '#',                              label: 'Twitter' },
                { icon: Linkedin,  href: '#',                              label: 'LinkedIn' },
                { icon: Instagram, href: '#',                              label: 'Instagram' },
                { icon: Mail,      href: 'mailto:andreja@pantasocial.com', label: 'Email' },
              ].map(({ icon: Icon, href, label }) => (
                <a
                  key={label}
                  href={href}
                  aria-label={label}
                  className="w-9 h-9 rounded-xl flex items-center justify-center text-white/40 hover:text-white transition-all duration-200"
                  style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.07)' }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.borderColor = 'rgba(99,102,241,0.4)'
                    e.currentTarget.style.boxShadow = '0 0 12px rgba(99,102,241,0.2)'
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.borderColor = 'rgba(255,255,255,0.07)'
                    e.currentTarget.style.boxShadow = 'none'
                  }}
                >
                  <Icon size={15} />
                </a>
              ))}
            </div>
          </div>

          {/* Link columns */}
          {Object.entries(links).map(([section, items]) => (
            <div key={section}>
              <div className="text-white font-semibold text-sm mb-4">{section}</div>
              <ul className="space-y-3">
                {items.map((item) => (
                  <li key={item.label}>
                    <a href={item.href}
                      className="text-white/40 hover:text-white text-sm transition-colors duration-200 hover:underline underline-offset-4 decoration-brand-500/40">
                      {item.label}
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        {/* Bottom bar */}
        <div className="flex flex-col sm:flex-row items-center justify-between gap-4 pt-8 border-t border-white/[0.05]">
          <p className="text-white/20 text-xs">
            © 2026 PantaSocial LLC. All rights reserved.
          </p>
          <p className="text-white/20 text-xs">
            Built by Andreja Pantic ·{' '}
            <a href="mailto:andreja@pantasocial.com"
              className="text-brand-500/60 hover:text-brand-400 transition-colors">
              andreja@pantasocial.com
            </a>
          </p>
        </div>
      </div>
    </footer>
  )
}

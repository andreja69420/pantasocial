# PantaSocial LLC — Project Context for Claude Code

## What this project is

PantaSocial LLC is a digital marketing agency founded by Andreja Pantic. This repository is the company's own marketing website — a Next.js landing page that sells the agency's services to local businesses (plumbers, dentists, HVAC, lawyers, contractors, restaurants, etc.). It is not a client project; it is PantaSocial's own storefront.

The site's job is to convert visitors into leads by showcasing services, building trust, and funneling people to a contact form that emails Andreja directly.

## Tech stack

- **Framework**: Next.js 14 with the Pages Router (not App Router). All pages live in `pages/`.
- **Language**: TypeScript throughout. No `.js` files in components or pages.
- **Styling**: Tailwind CSS v3 with a custom theme defined in `tailwind.config.js`. No CSS Modules. All one-off styles use inline `style={{}}` props.
- **Animations**: Framer Motion for all transitions, entrance animations, scroll effects, and motion values.
- **Icons**: Lucide React only. Do not introduce other icon libraries.
- **Email**: Nodemailer via `pages/api/contact.ts`. SMTP credentials come from environment variables.
- **Fonts**: Inter from Google Fonts, loaded via `@import` at the top of `styles/globals.css`.
- **No database**: The site is entirely static + one API route. No Prisma, no Supabase, no external data store.
- **No auth**: No login, no sessions, no protected routes.
- **Dev port**: 3000 (`npm run dev`).

## File structure

```
pages/
  index.tsx              — Main landing page, composes all sections
  _app.tsx               — Wraps app, imports globals.css
  contact.tsx            — Standalone contact page
  about.tsx              — About page
  faq.tsx                — FAQ page
  free-audit.tsx         — Free audit offer page
  blog.tsx               — Blog placeholder
  case-studies.tsx       — Case studies placeholder
  privacy.tsx            — Privacy policy
  how-it-works.tsx       — How it works standalone
  api/
    contact.ts           — POST endpoint, validates fields, sends email via nodemailer

components/
  Navbar.tsx             — Fixed top nav with scroll-aware opacity
  Hero.tsx               — Above-the-fold section: badge, headline, CTAs, stats, marquee
  TrustBar.tsx           — Industry pills (Plumbing, Dental, Law, etc.)
  Problem.tsx            — 6 problem cards explaining why local businesses struggle online
  Solution.tsx           — 4 solution cards describing PantaSocial's offerings
  HowItWorks.tsx         — Step-by-step process section
  Results.tsx            — Social proof / results section
  Services.tsx           — 4 service cards with CTAs (no prices shown)
  Testimonials.tsx       — Client testimonials
  CTA.tsx                — Final call-to-action section with embedded ContactForm
  ContactForm.tsx        — Reusable form: name, business, email, phone, service, message
  Footer.tsx             — Links, email, copyright 2026
  FloatingCTA.tsx        — Fixed bottom-center bar, appears after 700px scroll, dismissible
  FixedWidgets.tsx       — Fixed bottom-left Growth Dashboard widget (visible from load)
  PageLayout.tsx         — Shared layout wrapper for inner pages

hooks/
  useRealtimeMetrics.ts  — Deterministic pseudo-random metrics engine for the dashboard widget

styles/
  globals.css            — Tailwind directives, custom classes, animations, mesh background

pages/services/
  [slug].tsx             — Dynamic service detail pages (web-design, lead-generation, seo-ads, automation)
```

## Brand and design rules

**Color palette** (defined in `tailwind.config.js` under `theme.extend.colors`):
- `brand-400` through `brand-600`: indigo (#8193f8 → #6366f1 → #5855eb) — primary brand color
- `neon.purple`: #a855f7 — secondary accent
- `neon.cyan`: #06b6d4 — tertiary accent
- `dark-950`: #020206 — page background
- All text on dark backgrounds. Never use light backgrounds.

**Design language**: Dark mode only. Glassmorphism cards. Subtle grid overlay. Neon glows on interactive elements. Framer Motion entrance animations on scroll (`whileInView`, `once: true`). No light mode toggle exists or should be added.

**Tone**: Professional but direct. No AI buzzwords. No "automated client machines", no "smart outreach", no "on autopilot". Speak about real services: website, Instagram, Facebook, Google, SEO, ads. The client hates generic AI marketing copy.

**CSS utility classes** (defined in `globals.css` `@layer components`):
- `.glass` — frosted glass background
- `.glass-card` — darker glass card
- `.gradient-text-brand` — indigo→purple→cyan gradient text
- `.gradient-text-hero` — white→blue→purple gradient text (for hero h1)
- `.btn-primary` — gradient purple CTA button with glow
- `.btn-secondary` — subtle ghost button
- `.section-label` — small pill badge above section headings
- `.glow-dot` — animated pulsing dot used inside `.section-label`
- `.cyber-corner` — bracket decoration on corners of elements
- `.holo-card` — holographic shimmer on hover
- `.mesh-bg` — animated radial gradient background (used in index.tsx)
- `.grid-overlay` — subtle indigo grid lines (used in index.tsx)

## Component patterns

**Section heading pattern** (used in every section):
```tsx
<div className="section-label mb-6 mx-auto w-fit">
  <span className="glow-dot" />
  Section Title
</div>
<h2 className="font-black tracking-tight text-white" style={{ fontSize: 'clamp(2rem, 4vw, 3.5rem)' }}>
  Main headline with <span className="gradient-text-brand">gradient accent.</span>
</h2>
```

**Card alignment pattern** (for equal-height cards with pinned CTAs):
```tsx
<div className="relative p-8 flex flex-col h-full">
  <p className="text-white/50 text-sm leading-relaxed mb-6 flex-1">{desc}</p>
  <a href="#contact" className="mt-auto ...">CTA</a>
</div>
```

**Widget row alignment pattern** (for narrow fixed widgets):
```tsx
<div className="flex items-center">
  <span className="flex-1 text-[10px] text-white/35 truncate">Label</span>
  <div className="flex items-center gap-1.5 flex-shrink-0">
    <span className="tabular-nums font-bold text-white">{value}</span>
  </div>
</div>
```

## Services (the four things PantaSocial sells)

1. **Web Design & Development** — slug: `web-design`, accent: `#6366f1`
2. **Lead Generation** — slug: `lead-generation`, accent: `#a855f7`, marked "Most Popular"
3. **SEO & Google Ads** — slug: `seo-ads`, accent: `#06b6d4`
4. **Full Automation System** — slug: `automation`, accent: `#10b981`

No prices are displayed anywhere on the site. CTAs say "Get Free Audit" or "Get My Free Website Preview".

## The Growth Dashboard widget

`components/FixedWidgets.tsx` renders a live-looking dashboard in the bottom-left corner. It is visible from page load and disappears when the footer enters the viewport (IntersectionObserver). It uses deterministic pseudo-random numbers so all visitors see the same metrics at the same moment.

The metrics engine is in `hooks/useRealtimeMetrics.ts`:
- Uses mulberry32 PRNG seeded by the current date integer
- `businessProgress(hour)` — piecewise function that weights metrics by time of day
- Monthly leads accumulate deterministically; they only go up, never down
- Recalculates every 60 seconds

## Contact form and email

`pages/api/contact.ts` handles POST requests from `components/ContactForm.tsx`. It:
- Validates: name, business name, email, message (all required)
- Sends a styled HTML email to `andreja@pantasocial.com`
- Sets reply-to as the sender's email
- Returns 200 on success, 400 on validation error, 500 on send failure

Environment variables required (see `.env.local.example`):
- `SMTP_HOST` — e.g. smtp.gmail.com
- `SMTP_PORT` — e.g. 587
- `SMTP_USER` — sender Gmail address
- `SMTP_PASS` — Gmail app password (not account password)

## What NOT to do

- Do not mention AI, automation bots, or "autopilot" in any user-facing copy. The founder explicitly hates this language.
- Do not add prices to service cards. Pricing is handled in consultation calls.
- Do not add a light mode, theme toggle, or any light-background sections.
- Do not use `grid-cols-1` for the services section on desktop — it should be `lg:grid-cols-4`.
- Do not create a new CSS file or use CSS Modules. All styles go in `globals.css` or inline.
- Do not introduce new npm packages without a strong reason. Current bundle is intentionally lean.
- Do not use `process.env` values on the client side (only in `pages/api/`).
- Do not add comments explaining what code does — only add comments when the WHY is non-obvious.
- Do not wrap every section in a new `<motion.div>` — use `whileInView` on the elements themselves.
- Do not use `justify-between` in narrow widget rows without `flex-1 truncate` on the label and `flex-shrink-0` on the value — this causes layout jamming in 240px-wide containers.

## Founder

**Andreja Pantic** — founder of PantaSocial LLC. Contact: andreja@pantasocial.com. All enquiry emails go to this address. The site footer shows 2026 as the copyright year.

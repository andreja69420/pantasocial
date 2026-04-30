/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
      },
      colors: {
        brand: {
          50:  '#f0f4ff',
          100: '#e0e9ff',
          200: '#c7d7fe',
          300: '#a5bafc',
          400: '#8193f8',
          500: '#6366f1',
          600: '#5855eb',
          700: '#4a43d5',
          800: '#3c37ab',
          900: '#343087',
        },
        neon: {
          blue:   '#4f8eff',
          purple: '#a855f7',
          cyan:   '#06b6d4',
          green:  '#10b981',
        },
        dark: {
          950: '#020206',
          900: '#08080f',
          800: '#0d0d1a',
          700: '#111127',
          600: '#16163a',
        },
      },
      backgroundImage: {
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
        'gradient-mesh':
          'radial-gradient(at 40% 20%, hsla(228,100%,74%,0.15) 0px, transparent 50%), radial-gradient(at 80% 0%, hsla(269,100%,76%,0.1) 0px, transparent 50%), radial-gradient(at 0% 50%, hsla(355,100%,93%,0.05) 0px, transparent 50%), radial-gradient(at 80% 50%, hsla(340,100%,76%,0.08) 0px, transparent 50%), radial-gradient(at 0% 100%, hsla(225,100%,77%,0.1) 0px, transparent 50%)',
      },
      animation: {
        'pulse-slow':    'pulse 4s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'float':         'float 6s ease-in-out infinite',
        'glow':          'glow 2s ease-in-out infinite alternate',
        'gradient-x':    'gradient-x 8s ease infinite',
        'gradient-y':    'gradient-y 8s ease infinite',
        'gradient-xy':   'gradient-xy 8s ease infinite',
        'spin-slow':     'spin 8s linear infinite',
        'shimmer':       'shimmer 2s linear infinite',
        'marquee':       'marquee 28s linear infinite',
      },
      keyframes: {
        float: {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%':      { transform: 'translateY(-20px)' },
        },
        glow: {
          from: { boxShadow: '0 0 20px rgba(99,102,241,0.3)' },
          to:   { boxShadow: '0 0 40px rgba(99,102,241,0.6), 0 0 60px rgba(99,102,241,0.3)' },
        },
        'gradient-x': {
          '0%, 100%': { backgroundSize: '200% 200%', backgroundPosition: 'left center' },
          '50%':      { backgroundSize: '200% 200%', backgroundPosition: 'right center' },
        },
        shimmer: {
          '0%':   { backgroundPosition: '-200% 0' },
          '100%': { backgroundPosition: '200% 0' },
        },
        marquee: {
          '0%':   { transform: 'translateX(0%)' },
          '100%': { transform: 'translateX(-50%)' },
        },
      },
      boxShadow: {
        'neon-blue':   '0 0 20px rgba(79,142,255,0.4), 0 0 60px rgba(79,142,255,0.15)',
        'neon-purple': '0 0 20px rgba(168,85,247,0.4), 0 0 60px rgba(168,85,247,0.15)',
        'neon-cyan':   '0 0 20px rgba(6,182,212,0.4), 0 0 60px rgba(6,182,212,0.15)',
        'glass':       '0 8px 32px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.05)',
        'card':        '0 4px 24px rgba(0,0,0,0.5), 0 1px 0 rgba(255,255,255,0.03)',
        'card-hover':  '0 8px 40px rgba(0,0,0,0.6), 0 0 0 1px rgba(99,102,241,0.15), 0 1px 0 rgba(255,255,255,0.06)',
      },
      backdropBlur: {
        xs: '2px',
      },
    },
  },
  plugins: [],
}

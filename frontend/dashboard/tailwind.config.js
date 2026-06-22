/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        ink: {
          950: '#060912',
          900: '#0a0e1a',
          850: '#0f1424',
          800: '#141b2e',
          700: '#1c2438',
          600: '#283044',
        },
        eeg: {
          DEFAULT: '#22d3ee',
          dim: '#0e7490',
        },
        gene: {
          DEFAULT: '#a78bfa',
          dim: '#6d28d9',
        },
        fuse: {
          DEFAULT: '#34d399',
          dim: '#047857',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'monospace'],
      },
      boxShadow: {
        glow: '0 0 40px -10px rgba(34, 211, 238, 0.35)',
        'glow-violet': '0 0 40px -10px rgba(167, 139, 250, 0.35)',
        'glow-emerald': '0 0 40px -10px rgba(52, 211, 153, 0.35)',
      },
      keyframes: {
        'pulse-line': {
          '0%, 100%': { opacity: '0.3' },
          '50%': { opacity: '1' },
        },
        float: {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-8px)' },
        },
        shimmer: {
          '100%': { transform: 'translateX(100%)' },
        },
      },
      animation: {
        'pulse-line': 'pulse-line 2.5s ease-in-out infinite',
        float: 'float 6s ease-in-out infinite',
      },
    },
  },
  plugins: [],
}

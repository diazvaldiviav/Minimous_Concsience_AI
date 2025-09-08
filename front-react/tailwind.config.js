/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Legacy consciousness colors (keep for compatibility)
        consciousness: {
          50: '#f0f9ff',
          100: '#e0f2fe',
          200: '#bae6fd',
          300: '#7dd3fc',
          400: '#38bdf8',
          500: '#0ea5e9',
          600: '#0284c7',
          700: '#0369a1',
          800: '#075985',
          900: '#0c4a6e',
        },
        neural: {
          50: '#fdf4ff',
          100: '#fae8ff',
          200: '#f5d0fe',
          300: '#f0abfc',
          400: '#e879f9',
          500: '#d946ef',
          600: '#c026d3',
          700: '#a21caf',
          800: '#86198f',
          900: '#701a75',
        },
        // Xentauri Space Theme Colors
        'void-black': '#0A0A0F',
        'deep-space': '#1A1A2E',
        'cosmic-navy': '#16213E',
        'nebula-purple': '#2D1B69',
        'cyan-nebula': '#00D4FF',
        'magenta-pulsar': '#FF00FF',
        'quantum-green': '#00FF88',
        'yellow-star': '#FFED4E',
        'orange-nova': '#FF6B35',
        'red-shift': '#FF1744',
        'blue-hyperspace': '#0066FF',
        'purple-void': '#8A2BE2',
        'star-white': '#FFFFFF',
        'cosmic-silver': '#B0B8C4',
        'space-dust': '#8B949E',
        'cosmic-gray': '#525866',
        'asteroid-dark': '#2D3748'
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'consciousness': 'consciousness 4s ease-in-out infinite',
        'neural-activity': 'neural 2s ease-in-out infinite alternate',
      },
      keyframes: {
        consciousness: {
          '0%, 100%': { opacity: '0.5', transform: 'scale(1)' },
          '50%': { opacity: '1', transform: 'scale(1.05)' },
        },
        neural: {
          '0%': { opacity: '0.3' },
          '100%': { opacity: '0.8' },
        }
      }
    },
  },
  plugins: [
    require('@tailwindcss/forms'),
  ],
}
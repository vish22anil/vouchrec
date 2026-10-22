/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Redesigned production palette
        ivory: '#F7F5F0',
        ink: '#202722',
        slate: {
          muted: '#68716B',
        },
        stone: {
          warm: '#E5E0D8',
        },
        aubergine: {
          DEFAULT: '#60445A',
          hover: '#4E3749',
          light: '#F4EEF2',
          border: '#D9C8D4',
          dark: '#3D2A39',
        },
        copper: {
          DEFAULT: '#B87958',
          hover: '#A26748',
          light: '#F9F2ED',
          border: '#E8D4C8',
          dark: '#7D4F37',
        },
        night: {
          bg: '#171A18',
          surface: '#222724',
          card: '#29302B',
          border: '#373F39',
          text: '#F3F1EC',
          muted: '#A8B0AA',
        },
        // Legacy research tokens for Research Lab
        vouch: {
          dark: '#171A18',
          surface: '#222724',
          card: '#29302B',
          border: '#373F39',
          muted: '#68716B',
          accent: '#60445A',
          highlight: '#775770',
          green: '#52b788',
          mint: '#74c69d',
          pale: '#d8f3dc',
        }
      },
      fontFamily: {
        serif: ['Newsreader', 'Playfair Display', 'Georgia', 'serif'],
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      }
    },
  },
  plugins: [],
}

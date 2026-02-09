/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        dark: '#0f172a',
        card: '#1e293b',
        accent: '#3b82f6',
        success: '#22c55e'
      }
    },
  },
  plugins: [],
}

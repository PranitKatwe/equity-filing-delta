/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  darkMode: "media",
  theme: {
    extend: {
      colors: {
        surface: { DEFAULT: "var(--surface-0)", card: "var(--surface-1)" },
        line: "var(--border)",
        ink: {
          DEFAULT: "var(--text-primary)",
          soft: "var(--text-secondary)",
          mute: "var(--text-muted)",
        },
        insample: "var(--insample)",
        holdout: "var(--holdout)",
        pos: "var(--pos)",
        neg: "var(--neg)",
        good: "var(--good)",
        warn: "var(--warn)",
      },
      maxWidth: { content: "56rem" },
      fontFamily: {
        sans: [
          "-apple-system", "BlinkMacSystemFont", '"Segoe UI"', "Roboto",
          "Helvetica", "Arial", "sans-serif",
        ],
      },
    },
  },
  plugins: [],
};

import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: { sans: ["var(--font-inter)", "sans-serif"] },
      colors: {
        brand: {
          violet: "#7c3aed",
          purple: "#a855f7",
          cyan:   "#06b6d4",
        },
      },
      animation: {
        "fade-in":   "fadeIn 0.4s ease both",
        "slide-up":  "slideUp 0.4s ease both",
        "pulse-slow":"pulse 3s ease-in-out infinite",
        "spin-slow": "spin 8s linear infinite",
        shimmer:     "shimmer 1.8s linear infinite",
      },
      keyframes: {
        fadeIn:  { from: { opacity: "0" },                        to: { opacity: "1" } },
        slideUp: { from: { opacity: "0", transform: "translateY(12px)" }, to: { opacity: "1", transform: "translateY(0)" } },
        shimmer: { "0%": { backgroundPosition: "-200% 0" }, "100%": { backgroundPosition: "200% 0" } },
      },
      backgroundSize: { "200%": "200% 200%" },
    },
  },
  plugins: [],
};

export default config;

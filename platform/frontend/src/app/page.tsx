import Link from "next/link";
import { UploadCloud, Zap, BarChart2, GitCompare, Network, FlaskConical, Layers } from "lucide-react";

const FEATURES = [
  {
    icon: UploadCloud,
    color: "text-cyan-400",
    iconBg: "bg-cyan-900/50 border-cyan-700",
    title: "Multi-Format Upload",
    desc: "Upload CSV, XLSX, TSV, mzML, or mzXML files. Smart column detection handles diverse FT-ICR MS export formats.",
  },
  {
    icon: Zap,
    color: "text-violet-400",
    iconBg: "bg-violet-900/50 border-violet-700",
    title: "Three GNN Models",
    desc: "Choose from Model9 (baseline), Model10 (BW features), or Model11 (interaction features) for formula assignment.",
  },
  {
    icon: BarChart2,
    color: "text-purple-400",
    iconBg: "bg-purple-900/50 border-purple-700",
    title: "Rich Visualisations",
    desc: "Interactive Van Krevelen plots, element-class breakdowns, m/z histograms, and paginated results tables.",
  },
  {
    icon: GitCompare,
    color: "text-pink-400",
    iconBg: "bg-pink-900/50 border-pink-700",
    title: "Model Comparison",
    desc: "Side-by-side accuracy, coverage, and mass-error benchmarks across all three MOSAIC-GNN variants.",
  },
  {
    icon: Network,
    color: "text-emerald-400",
    iconBg: "bg-emerald-900/50 border-emerald-700",
    title: "Graph Neural Network",
    desc: "SMDG relational GNN + isotope transformer + FCL crowd-sourcing graphs power state-of-the-art formula scoring.",
  },
  {
    icon: FlaskConical,
    color: "text-orange-400",
    iconBg: "bg-orange-900/50 border-orange-700",
    title: "DOM & PFAS Ready",
    desc: "Pre-trained on dissolved organic matter and PFAS datasets. Covers CHO, CHON, CHOS, CHONS, CHONSP classes.",
  },
];

const STATS = [
  { label: "Element Classes", value: "7" },
  { label: "GNN Models",      value: "3" },
  { label: "File Formats",    value: "6" },
  { label: "Graph Layers",    value: "4" },
];

export default function HomePage() {
  return (
    <div className="dot-grid">
      {/* Hero */}
      <div className="relative flex flex-col items-center text-center pt-14 pb-10 px-6 overflow-hidden">
        {/* Glow orb */}
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[500px] h-[260px]
                        bg-violet-700/20 rounded-full blur-[90px] pointer-events-none" />

        <div className="relative flex items-center gap-2 mb-5 px-4 py-1.5 rounded-full
                        bg-violet-900/50 border border-violet-700 text-violet-300 text-sm">
          <Layers size={13} />
          Graph Neural Network Platform
        </div>

        <h1 className="relative text-5xl font-extrabold tracking-tight leading-tight max-w-2xl">
          <span className="text-white">Formula Assignment </span>
          <span className="grad-text">with MOSAIC-GNN</span>
        </h1>

        <p className="relative mt-4 text-slate-400 max-w-lg text-lg leading-relaxed">
          Upload FT-ICR MS data, assign molecular formulae via evidence-based graph neural
          networks, and explore results with interactive Van&nbsp;Krevelen diagnostics.
        </p>

        <div className="relative mt-7 flex items-center gap-4">
          <Link href="/upload" className="btn-primary text-base px-7 py-3.5 glow-violet">
            Get Started →
          </Link>
          <Link href="/compare" className="btn-secondary text-base">
            View Benchmarks
          </Link>
        </div>

        {/* Stats */}
        <div className="relative mt-10 flex flex-wrap justify-center gap-3">
          {STATS.map(s => (
            <div key={s.label} className="stat-pill">
              <span className="text-violet-400 font-bold text-base">{s.value}</span>
              <span>{s.label}</span>
            </div>
          ))}
        </div>
      </div>

      {/* How it works */}
      <div className="px-6 pb-10 max-w-5xl mx-auto">
        <p className="text-center text-xs font-semibold tracking-widest text-slate-600 uppercase mb-8">
          How it works
        </p>
        <div className="flex flex-col sm:flex-row gap-4 sm:gap-0 items-stretch sm:items-start">
          <div className="flow-step">
            <div className="flex items-center gap-3 mb-3">
              <div className="step-badge">1</div>
              <h3 className="font-bold text-slate-100 text-base">Upload</h3>
            </div>
            <p className="text-sm text-slate-400 leading-relaxed">
              Drop a CSV, XLSX, mzML, or mzXML file. Smart column detection handles any FT-ICR MS export format automatically.
            </p>
          </div>

          {/* Connector arrow */}
          <div className="hidden sm:flex items-center justify-center flex-shrink-0 w-9 pt-5">
            <svg width="28" height="12" viewBox="0 0 28 12" fill="none">
              <path d="M0 6H24M24 6L19 1M24 6L19 11" stroke="#475569" strokeWidth="1.5" strokeLinecap="round"/>
            </svg>
          </div>

          <div className="flow-step">
            <div className="flex items-center gap-3 mb-3">
              <div className="step-badge">2</div>
              <h3 className="font-bold text-slate-100 text-base">Configure</h3>
            </div>
            <p className="text-sm text-slate-400 leading-relaxed">
              Choose Model9, 10, or&nbsp;11. Set an optional m/z range and ion mode, then submit for GNN inference.
            </p>
          </div>

          <div className="hidden sm:flex items-center justify-center flex-shrink-0 w-9 pt-5">
            <svg width="28" height="12" viewBox="0 0 28 12" fill="none">
              <path d="M0 6H24M24 6L19 1M24 6L19 11" stroke="#475569" strokeWidth="1.5" strokeLinecap="round"/>
            </svg>
          </div>

          <div className="flow-step">
            <div className="flex items-center gap-3 mb-3">
              <div className="step-badge">3</div>
              <h3 className="font-bold text-slate-100 text-base">Analyse</h3>
            </div>
            <p className="text-sm text-slate-400 leading-relaxed">
              Explore Van Krevelen plots, element-class charts, m/z histograms, and export your annotated results to CSV.
            </p>
          </div>
        </div>
      </div>

      {/* Features */}
      <div className="px-6 pb-14 max-w-5xl mx-auto">
        <div className="h-px bg-slate-800 mb-10" />
        <p className="text-center text-xs font-semibold tracking-widest text-slate-600 uppercase mb-8">Platform Features</p>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {FEATURES.map(({ icon: Icon, color, iconBg, title, desc }) => (
            <div key={title} className="glass p-5 hover:bg-slate-700/60 transition-colors cursor-default">
              <div className={`w-10 h-10 rounded-xl border flex items-center justify-center mb-4 ${iconBg}`}>
                <Icon size={20} className={color} />
              </div>
              <h3 className="font-semibold text-slate-100 mb-1.5 text-sm">{title}</h3>
              <p className="text-sm text-slate-500 leading-relaxed">{desc}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

"use client";
import { useState, useEffect, Suspense } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import axios from "axios";
import { Zap, Loader2, Layers, Check, FileText, ChevronRight, AlertCircle, Columns, Droplets, FlaskConical } from "lucide-react";
import clsx from "clsx";

type ModelName  = "Model10" | "Model11b";
type SampleType = "DOM" | "PFAS";

function PredictForm() {
  const params = useSearchParams();
  const router = useRouter();
  const job_id  = params.get("job_id")  ?? "";
  const filename = params.get("filename") ?? "";

  const [sampleType, setSampleType] = useState<SampleType>("DOM");
  const model: ModelName = sampleType === "PFAS" ? "Model11b" : "Model10";
  const [mzMin, setMzMin]   = useState("");
  const [mzMax, setMzMax]   = useState("");
  const [ionMode, setIonMode] = useState("both");
  const [loading, setLoading] = useState(false);
  const [error, setError]   = useState<string | null>(null);
  const [progress, setProgress] = useState("");

  // Column selector state
  const [columns, setColumns]         = useState<string[]>([]);
  const [fileType, setFileType]       = useState<"tabular" | "spectrum" | "">("");
  const [mzCol, setMzCol]             = useState("");
  const [intensityCol, setIntensityCol] = useState("");

  // Fetch column names from the uploaded file
  useEffect(() => {
    if (!job_id) return;
    axios.get(`/api/v1/upload/${job_id}/columns`)
      .then(({ data }) => {
        setFileType(data.file_type);
        if (data.file_type === "tabular" && data.columns.length > 0) {
          setColumns(data.columns);
          // Auto-select first likely m/z column
          const mzGuess = data.columns.find((c: string) =>
            /m.?z|mass/i.test(c)
          ) ?? data.columns[0];
          setMzCol(mzGuess);
          // Auto-select intensity column if obvious
          const intGuess = data.columns.find((c: string) =>
            /int(ensity)?|tic|abund/i.test(c)
          ) ?? "";
          setIntensityCol(intGuess);
        }
      })
      .catch(() => {/* non-critical — column picker hidden if fetch fails */});
  }, [job_id]);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setProgress("Submitting job…");
    try {
      await axios.post("/api/v1/predict/", {
        job_id,
        model,
        sample_type: sampleType,
        mass_range: mzMin && mzMax ? [parseFloat(mzMin), parseFloat(mzMax)] : null,
        ion_mode: ionMode,
        mz_column: mzCol || null,
        intensity_column: intensityCol || null,
      });
      setProgress("Running MOSAIC-GNN…");
      const poll = setInterval(async () => {
        try {
          const { data } = await axios.get(`/api/v1/predict/${job_id}/status`);
          if (data.status === "done") {
            clearInterval(poll);
            router.push(`/results?job_id=${job_id}`);
          } else if (data.status === "failed") {
            clearInterval(poll);
            setError(data.message ?? data.error ?? "Prediction failed — check backend logs");
            setLoading(false);
          }
        } catch {
          clearInterval(poll);
          setError("Lost connection to backend");
          setLoading(false);
        }
      }, 1500);
    } catch (err: unknown) {
      const msg = axios.isAxiosError(err) ? (err.response?.data?.detail ?? err.message) : "Request failed";
      setError(msg);
      setLoading(false);
    }
  }

  return (
    <div className="max-w-2xl mx-auto pt-8 pb-16 px-2">
      {/* Header */}
      <div className="mb-8">
        <p className="text-xs font-semibold text-violet-400 tracking-widest uppercase mb-2">Step 2 of 3</p>
        <h2 className="text-3xl font-bold text-white">Configure Prediction</h2>
        <div className="mt-2 flex items-center gap-2 text-slate-400 text-sm">
          <FileText size={14} className="text-slate-500" />
          <span className="font-mono text-slate-300">{decodeURIComponent(filename)}</span>
        </div>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Column selector — only shown for tabular files */}
        {fileType === "tabular" && columns.length > 0 && (
          <div className="glass p-4 rounded-xl border border-slate-700 space-y-4">
            <div className="flex items-center gap-2 mb-1">
              <Columns size={15} className="text-violet-400" />
              <p className="text-sm font-semibold text-slate-200">Input columns</p>
              <span className="ml-auto text-[10px] text-slate-500 font-mono">{columns.length} columns detected</span>
            </div>

            {/* m/z column — required */}
            <div>
              <label className="block text-xs font-medium text-slate-400 mb-1.5">
                m/z column <span className="text-red-400">*</span>
              </label>
              <select
                value={mzCol}
                onChange={e => setMzCol(e.target.value)}
                required
                className="w-full bg-slate-900 border border-slate-700 text-slate-200 rounded-lg
                           px-3 py-2 text-sm focus:outline-none focus:border-violet-500"
              >
                {columns.map(c => (
                  <option key={c} value={c}>{c}</option>
                ))}
              </select>
            </div>

            {/* Intensity column — optional */}
            <div>
              <label className="block text-xs font-medium text-slate-400 mb-1.5">
                Intensity column <span className="text-slate-600 font-normal">(optional)</span>
              </label>
              <select
                value={intensityCol}
                onChange={e => setIntensityCol(e.target.value)}
                className="w-full bg-slate-900 border border-slate-700 text-slate-200 rounded-lg
                           px-3 py-2 text-sm focus:outline-none focus:border-violet-500"
              >
                <option value="">— auto-detect —</option>
                {columns.map(c => (
                  <option key={c} value={c}>{c}</option>
                ))}
              </select>
            </div>
          </div>
        )}

        {/* Sample type selector */}
        <div>
          <p className="text-sm font-medium text-slate-300 mb-3">Sample type</p>
          <div className="grid grid-cols-2 gap-3">
            <button
              type="button"
              onClick={() => setSampleType("DOM")}
              className={clsx(
                "relative flex flex-col items-start p-4 rounded-xl border text-left transition-all",
                sampleType === "DOM"
                  ? "border-cyan-600 bg-cyan-900/30"
                  : "border-slate-700 bg-slate-900 hover:bg-slate-800 hover:border-slate-600"
              )}
            >
              {sampleType === "DOM" && (
                <div className="absolute top-3 right-3 w-4 h-4 rounded-full bg-cyan-500
                                flex items-center justify-center">
                  <Check size={10} className="text-white" />
                </div>
              )}
              <Droplets size={16} className={clsx("mb-2", sampleType === "DOM" ? "text-cyan-400" : "text-slate-500")} />
              <p className="font-semibold text-slate-100 text-sm">DOM / NOM</p>
              <p className="text-[10px] font-mono text-cyan-600 mt-0.5">Model 10 · 14-dim FCL</p>
              <p className="text-xs text-slate-500 mt-1 leading-relaxed">
                Dissolved / natural organic matter, environmental &amp; marine samples.
              </p>
            </button>
            <button
              type="button"
              onClick={() => setSampleType("PFAS")}
              className={clsx(
                "relative flex flex-col items-start p-4 rounded-xl border text-left transition-all",
                sampleType === "PFAS"
                  ? "border-amber-600 bg-amber-900/30"
                  : "border-slate-700 bg-slate-900 hover:bg-slate-800 hover:border-slate-600"
              )}
            >
              {sampleType === "PFAS" && (
                <div className="absolute top-3 right-3 w-4 h-4 rounded-full bg-amber-500
                                flex items-center justify-center">
                  <Check size={10} className="text-white" />
                </div>
              )}
              <FlaskConical size={16} className={clsx("mb-2", sampleType === "PFAS" ? "text-amber-400" : "text-slate-500")} />
              <p className="font-semibold text-slate-100 text-sm">PFAS</p>
              <p className="text-[10px] font-mono text-amber-600 mt-0.5">Model 11b · bypass · 78.9% Top-1</p>
              <p className="text-xs text-slate-500 mt-1 leading-relaxed">
                Per- &amp; polyfluoroalkyl substances, AFFF samples.
              </p>
            </button>
          </div>
        </div>

        {/* m/z range */}
        <div>
          <p className="text-sm font-medium text-slate-300 mb-3">m/z range <span className="text-slate-600 font-normal">(optional)</span></p>
          <div className="flex gap-3">
            <div className="flex-1">
              <input type="number" value={mzMin} onChange={e => setMzMin(e.target.value)}
                placeholder="Min (e.g. 100)" className="inp" />
            </div>
            <div className="flex items-center text-slate-600">—</div>
            <div className="flex-1">
              <input type="number" value={mzMax} onChange={e => setMzMax(e.target.value)}
                placeholder="Max (e.g. 800)" className="inp" />
            </div>
          </div>
        </div>

        {/* Ion mode */}
        <div>
          <p className="text-sm font-medium text-slate-300 mb-3">Ion mode</p>
          <div className="flex gap-2">
            {["both", "positive", "negative"].map(mode => (
              <button key={mode} type="button" onClick={() => setIonMode(mode)}
                className={clsx(
                  "flex-1 py-2.5 rounded-xl text-sm font-medium border transition-all capitalize",
                  ionMode === mode
                    ? "bg-violet-900/60 border-violet-600 text-violet-200"
                    : "bg-slate-900 border-slate-700 text-slate-400 hover:bg-slate-800"
                )}>
                {mode.charAt(0).toUpperCase() + mode.slice(1)}
              </button>
            ))}
          </div>
        </div>

        {error && (
          <div className="rounded-xl border border-red-800 bg-red-950/60 px-4 py-3 space-y-1">
            <div className="flex items-center gap-2 text-red-400 text-sm font-semibold">
              <AlertCircle size={14} className="flex-shrink-0" />
              Error
            </div>
            <pre className="text-xs text-red-300 whitespace-pre-wrap break-all font-mono">{error}</pre>
          </div>
        )}

        <button type="submit" disabled={loading || !job_id}
          className="btn-primary w-full text-base py-3.5 glow-violet">
          {loading ? (
            <><Loader2 className="animate-spin" size={18} /> {progress}</>
          ) : (
            <><Zap size={18} /> Run MOSAIC-GNN <ChevronRight size={16} /></>
          )}
        </button>
      </form>

      {loading && (
        <div className="mt-6 glass p-5 flex items-center gap-4">
          <div className="flex-shrink-0 w-10 h-10 rounded-full bg-violet-900/40 border border-violet-700
                          flex items-center justify-center">
            <Layers size={18} className="text-violet-400 animate-pulse" />
          </div>
          <div>
            <p className="text-sm font-medium text-slate-200">{progress}</p>
            <p className="text-xs text-slate-500 mt-0.5">Graph inference may take 10–60s depending on file size</p>
          </div>
        </div>
      )}
    </div>
  );
}

export default function PredictPage() {
  return <Suspense><PredictForm /></Suspense>;
}

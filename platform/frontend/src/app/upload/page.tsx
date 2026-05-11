"use client";
import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { useRouter } from "next/navigation";
import axios from "axios";
import {
  UploadCloud, FileText, FileSpreadsheet, FlaskConical,
  CheckCircle2, XCircle, Loader2, Info,
} from "lucide-react";
import clsx from "clsx";

const FORMATS = [
  { ext: "CSV",   icon: FileText,        color: "text-emerald-400", bg: "bg-emerald-900/60 border-emerald-700",  mime: "text/csv",                        exts: [".csv"] },
  { ext: "XLSX",  icon: FileSpreadsheet, color: "text-cyan-400",    bg: "bg-cyan-900/60 border-cyan-700",        mime: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", exts: [".xlsx", ".xls"] },
  { ext: "TSV",   icon: FileText,        color: "text-blue-400",    bg: "bg-blue-900/60 border-blue-700",        mime: "text/tab-separated-values",       exts: [".tsv"] },
  { ext: "TXT",   icon: FileText,        color: "text-slate-300",   bg: "bg-slate-700 border-slate-600",          mime: "text/plain",                      exts: [".txt"] },
  { ext: "mzML",  icon: FlaskConical,    color: "text-violet-400",  bg: "bg-violet-900/60 border-violet-700",    mime: "application/octet-stream",        exts: [".mzml"] },
  { ext: "mzXML", icon: FlaskConical,    color: "text-purple-400",  bg: "bg-purple-900/60 border-purple-700",    mime: "application/octet-stream",        exts: [".mzxml"] },
];

const ACCEPT = FORMATS.reduce((acc, f) => {
  acc[f.mime] = [...(acc[f.mime] ?? []), ...f.exts];
  return acc;
}, {} as Record<string, string[]>);

export default function UploadPage() {
  const router = useRouter();
  const [status, setStatus] = useState<"idle" | "uploading" | "done" | "error">("idle");
  const [error, setError] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const onDrop = useCallback(
    async (accepted: File[]) => {
      if (!accepted.length) return;
      const file = accepted[0];
      setSelectedFile(file);
      setStatus("uploading");
      setError(null);
      const form = new FormData();
      form.append("file", file);
      try {
        const { data } = await axios.post("/api/v1/upload/", form);
        setStatus("done");
        setTimeout(() => router.push(`/predict?job_id=${data.job_id}&filename=${encodeURIComponent(data.filename)}`), 600);
      } catch (err: unknown) {
        const msg = axios.isAxiosError(err)
          ? (err.response?.data?.detail ?? err.message)
          : "Upload failed";
        setError(msg);
        setStatus("error");
      }
    },
    [router]
  );

  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
    onDrop,
    accept: ACCEPT,
    maxFiles: 1,
  });

  const formatSize = (bytes: number) =>
    bytes < 1024 * 1024 ? `${(bytes / 1024).toFixed(1)} KB` : `${(bytes / 1024 / 1024).toFixed(1)} MB`;

  return (
    <div className="max-w-2xl mx-auto pt-8 pb-16 px-2">
      {/* Header */}
      <div className="mb-8">
        <p className="text-xs font-semibold text-violet-400 tracking-widest uppercase mb-2">Step 1 of 3</p>
        <h2 className="text-3xl font-bold text-white">Upload Mass-Spec Data</h2>
        <p className="mt-2 text-slate-400">Select a file containing your FT-ICR MS peaks to begin formula assignment.</p>
      </div>

      {/* Format badges */}
      <div className="flex flex-wrap gap-2 mb-6">
        {FORMATS.map(f => (
          <div key={f.ext} className={clsx(
            "flex items-center gap-1.5 px-3 py-1.5 rounded-lg border text-xs font-medium", f.bg, f.color
          )}>
            <f.icon size={12} />
            {f.ext}
          </div>
        ))}
      </div>

      {/* Drop zone */}
      <div
        {...getRootProps()}
        className={clsx(
          "relative border-2 border-dashed rounded-2xl p-12 text-center cursor-pointer transition-all",
          isDragReject && "border-red-500 bg-red-950/60",
          isDragActive && !isDragReject && "border-violet-500 bg-violet-900/25 glow-violet",
          status === "done"  && "border-emerald-500 bg-emerald-950/50",
          status === "error" && "border-red-500 bg-red-950/60",
          !isDragActive && status === "idle"  && "border-slate-700 hover:border-violet-500 hover:bg-slate-800/30",
        )}
      >
        <input {...getInputProps()} />

        {/* Glow on drag */}
        {isDragActive && !isDragReject && (
          <div className="absolute inset-0 rounded-2xl bg-violet-600/10 pointer-events-none" />
        )}

        <div className="flex flex-col items-center gap-4">
          {status === "uploading" ? (
            <Loader2 className="text-violet-400 animate-spin" size={44} />
          ) : status === "done" ? (
            <CheckCircle2 className="text-emerald-400" size={44} />
          ) : status === "error" ? (
            <XCircle className="text-red-400" size={44} />
          ) : isDragReject ? (
            <XCircle className="text-red-400" size={44} />
          ) : (
            <div className="relative">
              <div className="w-14 h-14 rounded-2xl bg-slate-800 border border-slate-700
                              flex items-center justify-center">
                <UploadCloud className="text-slate-400 group-hover:text-violet-400 transition-colors" size={28} />
              </div>
              {isDragActive && (
                <div className="absolute -inset-1 rounded-2xl border-2 border-violet-400 animate-pulse-slow" />
              )}
            </div>
          )}

          <div>
            <p className="text-slate-200 font-medium">
              {status === "uploading" ? "Uploading…" :
               status === "done"      ? "Upload complete — redirecting…" :
               status === "error"     ? "Upload failed" :
               isDragReject           ? "File type not supported" :
               isDragActive           ? "Drop it here!" :
               "Drag & drop or click to select"}
            </p>
            {status === "idle" && !isDragActive && (
              <p className="text-sm text-slate-500 mt-1">Max 200 MB · one file at a time</p>
            )}
          </div>
        </div>
      </div>

      {/* Selected file info */}
      {selectedFile && status !== "idle" && (
        <div className="mt-4 glass-sm px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <FileText size={16} className="text-slate-400 flex-shrink-0" />
            <div>
              <p className="text-sm font-medium text-slate-200">{selectedFile.name}</p>
              <p className="text-xs text-slate-500">{formatSize(selectedFile.size)}</p>
            </div>
          </div>
          {status === "done" && <CheckCircle2 size={16} className="text-emerald-400" />}
          {status === "uploading" && <Loader2 size={16} className="text-violet-400 animate-spin" />}
          {status === "error"   && <XCircle size={16} className="text-red-400" />}
        </div>
      )}

      {error && (
        <div className="mt-3 flex items-start gap-2 text-red-400 text-sm glass-sm px-4 py-3">
          <Info size={14} className="flex-shrink-0 mt-0.5" />
          {error}
        </div>
      )}

      {/* Hints */}
      <div className="mt-8 glass p-5">
        <p className="text-xs font-semibold text-slate-500 uppercase tracking-widest mb-3">Format notes</p>
        <ul className="space-y-1.5 text-sm text-slate-400">
          <li className="flex items-start gap-2"><span className="text-violet-400 mt-0.5">•</span>CSV/TSV/TXT: columns are auto-detected (<span className="font-mono text-xs">mz</span>, <span className="font-mono text-xs">intensity</span>, optional <span className="font-mono text-xs">formula</span>)</li>
          <li className="flex items-start gap-2"><span className="text-violet-400 mt-0.5">•</span>XLSX: first sheet is read; same column detection applies</li>
          <li className="flex items-start gap-2"><span className="text-violet-400 mt-0.5">•</span>mzML/mzXML: MS1 spectra only; peaks with intensity&nbsp;&gt;&nbsp;0 are extracted</li>
        </ul>
      </div>
    </div>
  );
}

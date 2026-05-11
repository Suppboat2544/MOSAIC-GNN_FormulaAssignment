"use client";
import { Suspense } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import axios from "axios";
import { GitCompare, Loader2 } from "lucide-react";
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  BarChart, Bar, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer,
} from "recharts";

function CompareView() {
  const params = useSearchParams();
  const job_id = params.get("job_id") ?? "";

  const { data, isLoading } = useQuery({
    queryKey: ["compare", job_id],
    queryFn: () => axios.get(`/api/v1/compare/${job_id}`).then(r => r.data),
    enabled: !!job_id,
  });

  const pageHeader = (
    <div className="mb-8">
      <p className="text-xs font-semibold text-violet-400 tracking-widest uppercase mb-1">Benchmarks</p>
      <h2 className="text-3xl font-bold text-white">Model Comparison</h2>
      <p className="text-slate-400 text-sm mt-1">
        Side-by-side accuracy, coverage, and mass-error benchmarks — M9 / M10 / M11.
      </p>
    </div>
  );

  if (isLoading) return (
    <div className="space-y-8 pb-10">
      {pageHeader}
      <div className="flex flex-col items-center justify-center h-56 gap-4 text-slate-400">
        <Loader2 className="animate-spin" size={32} />
        <p className="text-sm">Loading comparison…</p>
      </div>
    </div>
  );

  if (!data) return (
    <div className="space-y-8 pb-10">
      {pageHeader}
      <div className="glass p-10 flex flex-col items-center text-center gap-5">
        <div className="w-14 h-14 rounded-xl bg-slate-800 border border-slate-700
                        flex items-center justify-center">
          <GitCompare size={24} className="text-slate-500" />
        </div>
        <div>
          <p className="font-semibold text-slate-300 mb-2">No comparison data yet</p>
          <p className="text-sm text-slate-500 max-w-md leading-relaxed">
            Run a prediction first, then return here with a{" "}
            <code className="text-violet-400 bg-violet-950/60 px-1.5 py-0.5 rounded text-xs font-mono">
              ?job_id=…
            </code>{" "}
            query string to view benchmarks across all three models.
          </p>
        </div>
        <Link href="/upload" className="btn-primary px-6 py-2.5 text-sm mt-1">
          Start with Upload →
        </Link>
      </div>
    </div>
  );

  const metrics = data.metrics as Array<{
    model: string; top1: number; top5: number; mpe: number; coverage: number;
  }>;

  return (
    <div className="space-y-8 pb-10">
      {pageHeader}

      <div className="glass p-6">
        <h3 className="text-base font-semibold text-slate-200 mb-5">Top-1 / Top-5 Accuracy (%)</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={metrics}>
            <XAxis dataKey="model" stroke="#94a3b8" tick={{ fontSize: 12 }} />
            <YAxis stroke="#94a3b8" tick={{ fontSize: 12 }} />
            <Tooltip
              contentStyle={{ backgroundColor: "#1e293b", border: "1px solid #334155", borderRadius: "0.5rem" }}
              labelStyle={{ color: "#f1f5f9" }}
            />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            <Bar dataKey="top1" name="Top-1 %" fill="#7c3aed" radius={[4, 4, 0, 0]} />
            <Bar dataKey="top5" name="Top-5 %" fill="#a78bfa" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="glass p-6">
        <h3 className="text-base font-semibold text-slate-200 mb-5">Overall Metrics Radar</h3>
        <ResponsiveContainer width="100%" height={320}>
          <RadarChart data={metrics}>
            <PolarGrid stroke="#334155" />
            <PolarAngleAxis dataKey="model" stroke="#94a3b8" tick={{ fontSize: 12 }} />
            <PolarRadiusAxis stroke="#475569" tick={{ fontSize: 10 }} />
            <Radar name="Top-1" dataKey="top1" stroke="#7c3aed" fill="#7c3aed" fillOpacity={0.3} />
            <Radar name="Coverage" dataKey="coverage" stroke="#06b6d4" fill="#06b6d4" fillOpacity={0.2} />
            <Legend wrapperStyle={{ fontSize: 12 }} />
          </RadarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

export default function ComparePage() {
  return <Suspense><CompareView /></Suspense>;
}


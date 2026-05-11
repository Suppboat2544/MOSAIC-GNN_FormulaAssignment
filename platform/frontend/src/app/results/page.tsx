"use client";
import { Suspense } from "react";
import { useSearchParams } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import axios from "axios";
import { ElementClassChart } from "@/components/charts/ElementClassChart";
import { MassRangeHistogram } from "@/components/charts/MassRangeHistogram";
import { VanKrevelenPlot } from "@/components/charts/VanKrevelenPlot";
import { ResultsTable } from "@/components/charts/ResultsTable";
import { Download, BarChart2, Loader2, Network, TableProperties } from "lucide-react";
import clsx from "clsx";

function StatCard({
  label, value, sub, accent = "",
}: {
  label: string; value: string | number; sub?: string; accent?: string;
}) {
  return (
    <div className={`metric-card ${accent}`}>
      <p className="text-2xl font-bold text-white">{value}</p>
      <p className="text-xs font-medium text-slate-400 mt-0.5">{label}</p>
      {sub && <p className="text-[10px] text-slate-600 mt-0.5">{sub}</p>}
    </div>
  );
}

function SectionHeader({ icon: Icon, title }: { icon: React.ElementType; title: string }) {
  return (
    <div className="flex items-center gap-2.5 mb-4">
      <div className="w-7 h-7 rounded-lg bg-violet-900/50 border border-violet-700
                      flex items-center justify-center">
        <Icon size={14} className="text-violet-400" />
      </div>
      <h3 className="font-semibold text-slate-200">{title}</h3>
    </div>
  );
}

function ResultsView() {
  const params  = useSearchParams();
  const job_id  = params.get("job_id") ?? "";

  const { data, isLoading } = useQuery({
    queryKey: ["results", job_id],
    queryFn:  () => axios.get(`/api/v1/results/${job_id}`).then(r => r.data),
    enabled:  !!job_id,
  });
  const { data: vkData } = useQuery({
    queryKey: ["vankrevelen", job_id],
    queryFn:  () => axios.get(`/api/v1/vankrevelen/${job_id}`).then(r => r.data),
    enabled:  !!job_id,
  });
  const { data: classData } = useQuery({
    queryKey: ["element_classes", job_id],
    queryFn:  () => axios.get(`/api/v1/results/${job_id}/element_classes`).then(r => r.data),
    enabled:  !!job_id,
  });

  if (isLoading) return (
    <div className="flex flex-col items-center justify-center h-64 gap-4 text-slate-400">
      <Loader2 className="animate-spin" size={36} />
      <p className="text-sm">Loading results…</p>
    </div>
  );
  if (!data) return null;

  const results: Array<{mz?: number; formula?: string; confidence?: number}> = data.results ?? [];
  const total = results.length;
  const avgConf = total
    ? (results.reduce((s, r) => s + (r.confidence ?? 0), 0) / total * 100).toFixed(1)
    : "N/A";
  const uniqueFormulas = new Set(results.map(r => r.formula).filter(Boolean)).size;

  return (
    <div className="space-y-8 pb-10">
      {/* Top bar */}
      <div className="flex items-center justify-between">
        <div>
          <p className="text-xs font-semibold text-violet-400 tracking-widest uppercase mb-1">Step 3 of 3</p>
          <h2 className="text-3xl font-bold text-white">Results</h2>
        </div>
        <a
          href={`/api/v1/results/${job_id}/download`}
          className="btn-secondary gap-2"
        >
          <Download size={15} /> Download CSV
        </a>
      </div>

      {/* Stat cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <StatCard label="Total features"   value={total.toLocaleString()}                   accent="metric-card-violet" />
        <StatCard label="Unique formulas"  value={uniqueFormulas.toLocaleString()}           accent="metric-card-cyan" />
        <StatCard label="Avg confidence"   value={`${avgConf}%`}                             accent="metric-card-emerald" />
        <StatCard label="Element classes"  value={classData?.element_classes?.length ?? "—"} accent="metric-card-amber" />
      </div>

      {/* Charts row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        <div className="glass p-5">
          <SectionHeader icon={BarChart2} title="Element Class Distribution" />
          {classData ? <ElementClassChart data={classData.element_classes} /> : <p className="text-slate-500 text-sm">No class data</p>}
        </div>
        <div className="glass p-5">
          <SectionHeader icon={BarChart2} title="m/z Histogram" />
          {data ? <MassRangeHistogram data={data.results} /> : <p className="text-slate-500 text-sm">No data</p>}
        </div>
      </div>

      {/* Van Krevelen */}
      {vkData && (
        <div className="glass p-5">
          <SectionHeader icon={Network} title="Van Krevelen Plot" />
          <VanKrevelenPlot points={vkData.points} />
        </div>
      )}

      {/* Table */}
      {data && (
        <div className="glass p-5">
          <SectionHeader icon={TableProperties} title="Formula Assignments" />
          <ResultsTable rows={data.results} />
        </div>
      )}
    </div>
  );
}

export default function ResultsPage() {
  return <Suspense><ResultsView /></Suspense>;
}

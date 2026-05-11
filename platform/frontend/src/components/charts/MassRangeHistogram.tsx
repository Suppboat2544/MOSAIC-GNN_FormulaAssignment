"use client";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
} from "recharts";
import { useMemo } from "react";

interface Row { mz: number; formula: string }
interface Props { data: Row[] }

export function MassRangeHistogram({ data }: Props) {
  const bins = useMemo(() => {
    if (!data.length) return [];
    const mzs = data.map(r => r.mz);
    const min = Math.floor(Math.min(...mzs) / 50) * 50;
    const max = Math.ceil(Math.max(...mzs) / 50) * 50;
    const buckets: Record<string, number> = {};
    for (let b = min; b < max; b += 50) buckets[`${b}–${b + 50}`] = 0;
    data.forEach(r => {
      const key = Math.floor(r.mz / 50) * 50;
      buckets[`${key}–${key + 50}`] = (buckets[`${key}–${key + 50}`] ?? 0) + 1;
    });
    return Object.entries(buckets).map(([range, count]) => ({ range, count }));
  }, [data]);

  return (
    <div className="bg-slate-800 rounded-2xl p-5">
      <h3 className="text-base font-semibold mb-4">Assignments per Mass Range (Δ50 Da)</h3>
      <ResponsiveContainer width="100%" height={260}>
        <BarChart data={bins}>
          <XAxis dataKey="range" stroke="#94a3b8" fontSize={10} interval="preserveStartEnd" />
          <YAxis stroke="#94a3b8" fontSize={11} />
          <Tooltip contentStyle={{ backgroundColor: "#1e293b", border: "none", borderRadius: 8 }} />
          <Bar dataKey="count" fill="#7c3aed" radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

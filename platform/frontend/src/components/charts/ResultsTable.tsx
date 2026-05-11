"use client";
import { useState } from "react";

interface Row {
  mz: number;
  formula: string;
  element_class: string;
  mass_error_ppm: number;
  confidence: number;
}

interface Props { rows: Row[] }

const PAGE_SIZE = 50;

export function ResultsTable({ rows }: Props) {
  const [page, setPage] = useState(0);
  const total = Math.ceil(rows.length / PAGE_SIZE);
  const slice = rows.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);

  return (
    <div className="bg-slate-800 rounded-2xl p-5 space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-base font-semibold">Formula Assignments ({rows.length.toLocaleString()})</h3>
        <div className="flex gap-2 text-sm">
          <button disabled={page === 0} onClick={() => setPage(p => p - 1)}
            className="px-3 py-1 rounded-lg bg-slate-700 disabled:opacity-40">‹ Prev</button>
          <span className="px-2 py-1 text-slate-400">{page + 1} / {total}</span>
          <button disabled={page === total - 1} onClick={() => setPage(p => p + 1)}
            className="px-3 py-1 rounded-lg bg-slate-700 disabled:opacity-40">Next ›</button>
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-sm text-left">
          <thead>
            <tr className="text-slate-400 border-b border-slate-700">
              <th className="py-2 pr-4">m/z</th>
              <th className="py-2 pr-4">Formula</th>
              <th className="py-2 pr-4">Class</th>
              <th className="py-2 pr-4">Error (ppm)</th>
              <th className="py-2">Confidence</th>
            </tr>
          </thead>
          <tbody>
            {slice.map((r, i) => (
              <tr key={i} className="border-b border-slate-700/50 hover:bg-slate-700/40 transition-colors">
                <td className="py-1.5 pr-4 font-mono">{r.mz.toFixed(5)}</td>
                <td className="py-1.5 pr-4 font-mono text-violet-300">{r.formula}</td>
                <td className="py-1.5 pr-4">{r.element_class}</td>
                <td className="py-1.5 pr-4">{r.mass_error_ppm.toFixed(3)}</td>
                <td className="py-1.5">
                  <div className="flex items-center gap-2">
                    <div className="h-1.5 w-20 bg-slate-700 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-violet-500 rounded-full"
                        style={{ width: `${(r.confidence * 100).toFixed(0)}%` }}
                      />
                    </div>
                    <span className="text-slate-400 text-xs">{(r.confidence * 100).toFixed(1)}%</span>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

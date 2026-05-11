"use client";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
} from "recharts";

const COLORS = [
  "#7c3aed","#a78bfa","#06b6d4","#0ea5e9","#10b981","#f59e0b","#ef4444","#ec4899",
];

interface Props {
  data: Array<{ class: string; count: number }>;
}

export function ElementClassChart({ data }: Props) {
  return (
    <div className="bg-slate-800 rounded-2xl p-5">
      <h3 className="text-base font-semibold mb-4">Element-Class Distribution</h3>
      <ResponsiveContainer width="100%" height={260}>
        <BarChart data={data} layout="vertical" margin={{ left: 16 }}>
          <XAxis type="number" stroke="#94a3b8" fontSize={11} />
          <YAxis dataKey="class" type="category" stroke="#94a3b8" width={60} fontSize={11} />
          <Tooltip
            contentStyle={{ backgroundColor: "#1e293b", border: "none", borderRadius: 8 }}
            labelStyle={{ color: "#e2e8f0" }}
          />
          <Bar dataKey="count" radius={[0, 4, 4, 0]}>
            {data.map((_, i) => (
              <Cell key={i} fill={COLORS[i % COLORS.length]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

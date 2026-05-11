"use client";
import { useRef, useEffect, useState } from "react";
import * as d3 from "d3";

interface Point {
  mz: number;
  formula: string;
  element_class: string;
  H_C: number;
  O_C: number;
  confidence: number;
}

interface Props { points: Point[] }

const CLASS_COLORS: Record<string, string> = {
  CHO: "#7c3aed", CHON: "#0ea5e9", CHOS: "#10b981",
  CHONS: "#f59e0b", CHOP: "#ef4444", Other: "#94a3b8",
};

export function VanKrevelenPlot({ points }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [tooltip, setTooltip] = useState<{ x: number; y: number; d: Point } | null>(null);

  useEffect(() => {
    if (!svgRef.current || !points.length) return;
    const el = svgRef.current;
    const { width } = el.getBoundingClientRect();
    const height = 420;
    const margin = { top: 20, right: 20, bottom: 50, left: 55 };

    d3.select(el).selectAll("*").remove();
    const svg = d3.select(el).attr("height", height);

    const xScale = d3.scaleLinear().domain([0, d3.max(points, p => p.O_C)! + 0.05]).range([margin.left, width - margin.right]);
    const yScale = d3.scaleLinear().domain([0, d3.max(points, p => p.H_C)! + 0.1]).range([height - margin.bottom, margin.top]);

    // Axes
    svg.append("g").attr("transform", `translate(0,${height - margin.bottom})`)
      .call(d3.axisBottom(xScale).ticks(8))
      .call(g => g.select(".domain").attr("stroke", "#475569"))
      .call(g => g.selectAll("text").attr("fill", "#94a3b8").attr("font-size", 11));

    svg.append("g").attr("transform", `translate(${margin.left},0)`)
      .call(d3.axisLeft(yScale).ticks(8))
      .call(g => g.select(".domain").attr("stroke", "#475569"))
      .call(g => g.selectAll("text").attr("fill", "#94a3b8").attr("font-size", 11));

    // Axis labels
    svg.append("text").attr("x", width / 2).attr("y", height - 8)
      .attr("fill", "#94a3b8").attr("text-anchor", "middle").attr("font-size", 12).text("O/C");
    svg.append("text").attr("transform", "rotate(-90)")
      .attr("x", -height / 2).attr("y", 14)
      .attr("fill", "#94a3b8").attr("text-anchor", "middle").attr("font-size", 12).text("H/C");

    // Points
    svg.selectAll("circle").data(points).enter().append("circle")
      .attr("cx", p => xScale(p.O_C))
      .attr("cy", p => yScale(p.H_C))
      .attr("r", 3)
      .attr("fill", p => CLASS_COLORS[p.element_class] ?? CLASS_COLORS.Other)
      .attr("fill-opacity", 0.7)
      .on("mouseover", (event, d) => {
        const rect = el.getBoundingClientRect();
        setTooltip({ x: event.clientX - rect.left, y: event.clientY - rect.top, d });
      })
      .on("mouseout", () => setTooltip(null));
  }, [points]);

  return (
    <div className="bg-slate-800 rounded-2xl p-5 relative">
      <h3 className="text-base font-semibold mb-2">Van Krevelen Plot</h3>
      <div className="flex flex-wrap gap-3 mb-3">
        {Object.entries(CLASS_COLORS).map(([cls, col]) => (
          <span key={cls} className="flex items-center gap-1 text-xs text-slate-300">
            <span className="inline-block w-3 h-3 rounded-full" style={{ background: col }} />
            {cls}
          </span>
        ))}
      </div>
      <svg ref={svgRef} className="w-full" />
      {tooltip && (
        <div
          className="pointer-events-none absolute z-10 bg-slate-900 border border-slate-600 rounded-lg px-3 py-2 text-xs shadow-xl"
          style={{ left: tooltip.x + 10, top: tooltip.y - 10 }}
        >
          <p className="font-mono font-bold text-violet-300">{tooltip.d.formula}</p>
          <p>Class: {tooltip.d.element_class}</p>
          <p>m/z: {tooltip.d.mz.toFixed(5)}</p>
          <p>O/C: {tooltip.d.O_C.toFixed(3)} · H/C: {tooltip.d.H_C.toFixed(3)}</p>
          <p>Confidence: {(tooltip.d.confidence * 100).toFixed(1)}%</p>
        </div>
      )}
    </div>
  );
}

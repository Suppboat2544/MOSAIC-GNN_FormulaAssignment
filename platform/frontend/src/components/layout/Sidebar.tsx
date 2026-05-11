"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { UploadCloud, Zap, BarChart2, Home, GitCompare, Atom } from "lucide-react";
import clsx from "clsx";

// ── Workflow steps (ordered pipeline) ─────────────────────────────────────
const WORKFLOW = [
  { href: "/upload",  num: 1, label: "Upload",    sub: "Add data",     icon: UploadCloud },
  { href: "/predict", num: 2, label: "Configure", sub: "Select model", icon: Zap },
  { href: "/results", num: 3, label: "Analyse",   sub: "View results", icon: BarChart2 },
] as const;

// ── Extra navigation links ────────────────────────────────────────────────
const NAV = [
  { href: "/",        label: "Home",    icon: Home },
  { href: "/compare", label: "Compare", icon: GitCompare },
] as const;

const WORKFLOW_HREFS = WORKFLOW.map(s => s.href);

// ── Inline SVG check mark ─────────────────────────────────────────────────
function CheckMark() {
  return (
    <svg width="10" height="10" viewBox="0 0 10 10" fill="none">
      <path
        d="M1.5 5L3.8 7.5L8.5 2.5"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

// ── Single workflow step item ─────────────────────────────────────────────
function WorkflowStep({
  href, num, label, sub,
  active, done,
}: {
  href: string; num: number; label: string; sub: string;
  active: boolean; done: boolean;
}) {
  return (
    <Link
      href={href}
      className={clsx(
        "group relative flex items-center gap-3 px-3 py-2.5 rounded-xl transition-all duration-150",
        active
          ? "bg-violet-600/15 border border-violet-700/50"
          : "border border-transparent hover:bg-slate-800/60 hover:border-slate-700/40",
      )}
    >
      {/* Numbered / check circle */}
      <div
        className={clsx(
          "flex-shrink-0 w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold transition-colors duration-150",
          active
            ? "bg-violet-600 text-white shadow-md shadow-violet-900/60"
            : done
            ? "bg-violet-950 text-violet-400 border border-violet-800"
            : "bg-slate-800 text-slate-500 border border-slate-700",
        )}
      >
        {done ? <CheckMark /> : num}
      </div>

      {/* Labels */}
      <div className="flex-1 min-w-0">
        <p
          className={clsx(
            "text-[13px] font-semibold leading-none",
            active
              ? "text-slate-100"
              : done
              ? "text-slate-400"
              : "text-slate-500 group-hover:text-slate-300",
          )}
        >
          {label}
        </p>
        <p
          className={clsx(
            "text-[10px] mt-0.5 leading-none",
            active ? "text-violet-400" : "text-slate-700 group-hover:text-slate-500",
          )}
        >
          {sub}
        </p>
      </div>

      {/* Active dot */}
      {active && (
        <div className="w-1 h-1 rounded-full bg-violet-400 flex-shrink-0 animate-pulse" />
      )}
    </Link>
  );
}

// ── Sidebar component ─────────────────────────────────────────────────────
export function Sidebar() {
  const path = usePathname();
  const currentIdx = WORKFLOW_HREFS.indexOf(path as typeof WORKFLOW_HREFS[number]);

  return (
    <aside className="w-52 flex-shrink-0 flex flex-col bg-[#0c1526] border-r border-slate-800/80">
      {/* Logo area */}
      <div className="flex items-center gap-3 px-4 h-14 border-b border-slate-800/80 flex-shrink-0">
        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-violet-600 to-purple-700
                        flex items-center justify-center flex-shrink-0
                        shadow-lg shadow-violet-900/40">
          <Atom size={15} className="text-white" />
        </div>
        <div>
          <p className="text-[13px] font-bold tracking-wide text-white leading-none">MOSAIC</p>
          <p className="text-[9px] text-violet-400/80 tracking-widest uppercase leading-tight mt-0.5">
            GNN Platform
          </p>
        </div>
      </div>

      {/* ── Workflow steps ── */}
      <div className="px-3 pt-5 pb-2">
        <p className="px-2 mb-3 text-[9px] font-semibold text-slate-600 uppercase tracking-[0.14em]">
          Workflow
        </p>
        <div className="space-y-0.5">
          {WORKFLOW.map(({ href, num, label, sub }) => (
            <WorkflowStep
              key={href}
              href={href}
              num={num}
              label={label}
              sub={sub}
              active={path === href}
              done={currentIdx > WORKFLOW_HREFS.indexOf(href as typeof WORKFLOW_HREFS[number]) && currentIdx !== -1}
            />
          ))}
        </div>
      </div>

      <div className="h-px bg-slate-800 mx-3 my-3" />

      {/* ── General navigation ── */}
      <div className="px-3 pb-2">
        <p className="px-2 mb-2 text-[9px] font-semibold text-slate-600 uppercase tracking-[0.14em]">
          Navigation
        </p>
        {NAV.map(({ href, label, icon: Icon }) => {
          const active = path === href;
          return (
            <Link
              key={href}
              href={href}
              className={clsx(
                "group flex items-center gap-2.5 px-3 py-2 rounded-lg text-[13px] transition-colors duration-150",
                active
                  ? "bg-slate-800 text-slate-200"
                  : "text-slate-500 hover:bg-slate-800/60 hover:text-slate-300",
              )}
            >
              <Icon
                size={14}
                className={clsx(
                  active ? "text-slate-300" : "text-slate-600 group-hover:text-slate-400",
                )}
              />
              {label}
            </Link>
          );
        })}
      </div>

      {/* Spacer */}
      <div className="flex-1" />

      {/* Footer */}
      <div className="px-4 py-4 border-t border-slate-800">
        <div className="flex items-center gap-2 mb-1">
          <div className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
          <p className="text-[10px] text-slate-500">API ready</p>
        </div>
        <p className="text-[9px] text-slate-700 font-mono">v1.0 · M9 · M10 · M11</p>
      </div>
    </aside>
  );
}

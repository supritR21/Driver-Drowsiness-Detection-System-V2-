// frontend/src/app/monitor/page.tsx

import Link from "next/link";
import LiveMonitor from "@/components/LiveMonitor";

export default function MonitorPage() {
  return (
    <main
      className="h-screen overflow-hidden bg-[#060a10] px-4 py-4 text-white"
      style={{ fontFamily: "'JetBrains Mono', 'Fira Code', monospace" }}
    >
      <div className="mx-auto flex h-full max-w-400 flex-col gap-3">

        {/* ── Header ── */}
        <div className="flex items-center justify-between gap-4 px-1">
          <div className="flex items-center gap-5">
            {/* Status pill */}
            <div className="hidden sm:flex items-center gap-2 rounded-full border border-slate-700 bg-slate-900 px-3 py-1">
              <span className="h-1.5 w-1.5 rounded-full bg-cyan-400 animate-pulse" />
              <span className="text-[9px] tracking-[0.3em] uppercase text-slate-400">System Online</span>
            </div>

            <div>
              <p className="text-[9px] uppercase tracking-[0.3em] text-cyan-500 mb-0.5">
                Driver Drowsiness Detection System
              </p>
              <h1 className="text-xl font-bold tracking-tight text-white">
                Real-time Monitoring Dashboard
              </h1>
            </div>
          </div>

          <Link
            href="/dashboard"
            className="rounded-xl border border-slate-700 bg-slate-900 px-4 py-2 text-[11px] font-semibold tracking-wider uppercase text-slate-300 transition hover:bg-slate-800 hover:text-white"
          >
            Dashboard →
          </Link>
        </div>

        {/* ── Monitor (fills remaining height) ── */}
        <div className="min-h-0 flex-1">
          <LiveMonitor />
        </div>
      </div>
    </main>
  );
}
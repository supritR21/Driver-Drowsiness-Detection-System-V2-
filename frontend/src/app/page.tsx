import Link from "next/link";

export default function HomePage() {
  return (
    <main className="flex min-h-screen items-center justify-center bg-slate-950 px-4 text-white">
      <div className="max-w-2xl rounded-3xl border border-slate-800 bg-slate-900 p-8 text-center shadow-2xl">
        <p className="text-sm uppercase tracking-[0.3em] text-cyan-400">
          Semester Project
        </p>
        <h1 className="mt-4 text-4xl font-bold">Driver Drowsiness Detection System</h1>
        <p className="mt-4 text-slate-400">
          Live webcam monitoring, temporal drowsiness prediction, and multi-level alerting.
        </p>

        <div className="mt-8 flex items-center justify-center gap-4">
          <Link
            href="/monitor"
            className="rounded-2xl bg-white px-5 py-3 font-medium text-slate-950 transition hover:opacity-90"
          >
            Open Monitor
          </Link>
        </div>
      </div>
    </main>
  );
}
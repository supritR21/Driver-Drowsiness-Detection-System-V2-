import LiveMonitor from "@/components/LiveMonitor";

export default function MonitorPage() {
  return (
    <main className="min-h-screen bg-slate-950 px-4 py-8 text-white">
      <div className="mx-auto max-w-7xl">
        <div className="mb-8">
          <p className="text-sm uppercase tracking-[0.3em] text-cyan-400">
            Driver Drowsiness Detection System
          </p>
          <h1 className="mt-2 text-4xl font-bold">Real-time Monitoring Dashboard</h1>
          <p className="mt-3 max-w-3xl text-slate-400">
            Live webcam analysis, drowsiness scoring, and alert escalation built on FastAPI and
            PyTorch.
          </p>
        </div>

        <LiveMonitor />
      </div>
    </main>
  );
}
"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { fetchDashboard } from "@/lib/api";
import { clearToken, getToken } from "@/lib/auth";

type DashboardData = {
  user: {
    id: number;
    email: string;
    full_name: string | null;
    role: string;
  };
  summary: {
    total_sessions: number;
    total_alerts: number;
    average_score: number | null;
    max_score: number | null;
    alert_counts: Record<string, number>;
  };
  recent_sessions: Array<{
    id: number;
    session_name: string | null;
    started_at: string;
    ended_at: string | null;
    avg_score: number | null;
    max_score: number | null;
  }>;
  recent_alerts: Array<{
    id: number;
    alert_level: string;
    prediction: string;
    fatigue_score: number;
    message: string | null;
    frame_index: number | null;
    created_at: string;
  }>;
};

export default function DashboardPage() {
  const [data, setData] = useState<DashboardData | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const token = getToken();

    if (!token) {
      setError("No login session found. Please log in first.");
      setLoading(false);
      return;
    }

    fetchDashboard()
      .then((res) => setData(res))
      .catch((err) => setError(err instanceof Error ? err.message : "Failed to load"))
      .finally(() => setLoading(false));
  }, []);

  const handleLogout = () => {
    clearToken();
    window.location.href = "/login";
  };

  if (loading) {
    return <main className="min-h-screen bg-slate-950 p-8 text-white">Loading dashboard...</main>;
  }

  if (error || !data) {
    return (
      <main className="min-h-screen bg-slate-950 p-8 text-white">
        <div className="mx-auto max-w-2xl rounded-3xl border border-slate-800 bg-slate-900 p-6">
          <p className="text-red-300">{error}</p>
          <div className="mt-4 flex gap-3">
            <Link href="/login" className="rounded-2xl bg-white px-4 py-2 text-slate-950">
              Go to Login
            </Link>
          </div>
        </div>
      </main>
    );
  }

  return (
    <main className="min-h-screen bg-slate-950 px-4 py-8 text-white">
      <div className="mx-auto max-w-7xl">
        <div className="mb-8 flex items-center justify-between gap-4">
          <div>
            <p className="text-sm uppercase tracking-[0.3em] text-cyan-400">
              Personalized Dashboard
            </p>
            <h1 className="mt-2 text-4xl font-bold">
              Welcome, {data.user.full_name || data.user.email}
            </h1>
            <p className="mt-2 text-slate-400">{data.user.email}</p>
          </div>

          <div className="flex gap-3">
            <Link
              href="/monitor"
              className="rounded-2xl bg-white px-4 py-2 font-medium text-slate-950"
            >
              Open Monitor
            </Link>
            <button
              onClick={handleLogout}
              className="rounded-2xl border border-slate-700 px-4 py-2 font-medium text-white"
            >
              Logout
            </button>
          </div>
        </div>

        <div className="grid gap-4 md:grid-cols-4">
          <StatCard title="Sessions" value={data.summary.total_sessions.toString()} />
          <StatCard title="Alerts" value={data.summary.total_alerts.toString()} />
          <StatCard
            title="Avg Score"
            value={data.summary.average_score ? data.summary.average_score.toFixed(2) : "—"}
          />
          <StatCard
            title="Max Score"
            value={data.summary.max_score ? data.summary.max_score.toFixed(2) : "—"}
          />
        </div>

        <div className="mt-8 grid gap-6 lg:grid-cols-2">
          <section className="rounded-3xl border border-slate-800 bg-slate-900 p-5">
            <h2 className="mb-4 text-xl font-semibold">Recent Sessions</h2>
            <div className="space-y-3">
              {data.recent_sessions.length === 0 ? (
                <p className="text-slate-400">No sessions recorded yet.</p>
              ) : (
                data.recent_sessions.map((session) => (
                  <div
                    key={session.id}
                    className="rounded-2xl border border-slate-800 bg-slate-950 p-4"
                  >
                    <div className="flex items-center justify-between">
                      <p className="font-medium">
                        {session.session_name || `Session #${session.id}`}
                      </p>
                      <p className="text-sm text-slate-400">
                        {new Date(session.started_at).toLocaleString()}
                      </p>
                    </div>
                    <div className="mt-2 grid grid-cols-3 gap-3 text-sm text-slate-300">
                      <div>Avg: {session.avg_score?.toFixed(2) ?? "—"}</div>
                      <div>Max: {session.max_score?.toFixed(2) ?? "—"}</div>
                      <div>ID: {session.id}</div>
                    </div>
                  </div>
                ))
              )}
            </div>
          </section>

          <section className="rounded-3xl border border-slate-800 bg-slate-900 p-5">
            <h2 className="mb-4 text-xl font-semibold">Recent Alerts</h2>
            <div className="space-y-3">
              {data.recent_alerts.length === 0 ? (
                <p className="text-slate-400">No alerts recorded yet.</p>
              ) : (
                data.recent_alerts.map((alert) => (
                  <div
                    key={alert.id}
                    className="rounded-2xl border border-slate-800 bg-slate-950 p-4"
                  >
                    <div className="flex items-center justify-between">
                      <p className="font-medium capitalize">{alert.alert_level}</p>
                      <p className="text-sm text-slate-400">
                        {new Date(alert.created_at).toLocaleString()}
                      </p>
                    </div>
                    <p className="mt-2 text-sm text-slate-300">{alert.message}</p>
                    <div className="mt-2 text-xs text-slate-400">
                      Prediction: {alert.prediction} | Score: {alert.fatigue_score.toFixed(2)}
                    </div>
                  </div>
                ))
              )}
            </div>
          </section>
        </div>
      </div>
    </main>
  );
}

function StatCard({ title, value }: { title: string; value: string }) {
  return (
    <div className="rounded-3xl border border-slate-800 bg-slate-900 p-5">
      <p className="text-sm text-slate-400">{title}</p>
      <p className="mt-2 text-3xl font-bold">{value}</p>
    </div>
  );
}
"use client";

import { useEffect, useRef, useState } from "react";
import {
  AlertTriangle,
  Camera,
  CircleStop,
  Play,
  Radio,
  ShieldAlert,
} from "lucide-react";
import { sendFrame } from "@/lib/api";
import type { InferenceResponse } from "@/types/inference";

type StatusColor = "safe" | "soft" | "warning" | "danger" | "idle";

function statusColor(level: InferenceResponse["level"] | "idle"): string {
  switch (level) {
    case "safe":
      return "bg-emerald-500";
    case "soft":
      return "bg-yellow-500";
    case "warning":
      return "bg-orange-500";
    case "danger":
      return "bg-red-500";
    default:
      return "bg-slate-500";
  }
}

function statusBorder(level: InferenceResponse["level"] | "idle"): string {
  switch (level) {
    case "safe":
      return "border-emerald-500/40";
    case "soft":
      return "border-yellow-500/40";
    case "warning":
      return "border-orange-500/40";
    case "danger":
      return "border-red-500/40";
    default:
      return "border-slate-700";
  }
}

export default function LiveMonitor() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const intervalRef = useRef<number | null>(null);

  const audioContextRef = useRef<AudioContext | null>(null);
  const alarmTimerRef = useRef<number | null>(null);

  const [sessionId] = useState("demo-session");
  const [isRunning, setIsRunning] = useState(false);
  const [isStarting, setIsStarting] = useState(false);
  const [backendStatus, setBackendStatus] = useState<string>("idle");
  const [sequenceLength, setSequenceLength] = useState<number>(0);
  const [score, setScore] = useState<number | null>(null);
  const [level, setLevel] = useState<InferenceResponse["level"] | "idle">("idle");
  const [prediction, setPrediction] = useState<string>("—");
  const [message, setMessage] = useState<string>("Start camera to begin monitoring.");
  const [source, setSource] = useState<string>("—");
  const [permissionError, setPermissionError] = useState<string>("");

  // ---------- AUDIO ALARM SYSTEM ----------
  const ensureAudioContext = async () => {
    if (typeof window === "undefined") return null;

    if (!audioContextRef.current) {
      const AudioCtx = window.AudioContext || (window as any).webkitAudioContext;
      audioContextRef.current = new AudioCtx();
    }

    if (audioContextRef.current.state === "suspended") {
      await audioContextRef.current.resume();
    }

    return audioContextRef.current;
  };

  const clearAlarmTimer = () => {
    if (alarmTimerRef.current) {
      window.clearTimeout(alarmTimerRef.current);
      alarmTimerRef.current = null;
    }
  };

  const stopAlarm = () => {
    clearAlarmTimer();
  };

  const playBeep = (frequency: number, duration = 0.15, volume = 0.08) => {
    const ctx = audioContextRef.current;
    if (!ctx) return;

    const oscillator = ctx.createOscillator();
    const gainNode = ctx.createGain();

    oscillator.type = "sine";
    oscillator.frequency.value = frequency;

    oscillator.connect(gainNode);
    gainNode.connect(ctx.destination);

    const now = ctx.currentTime;

    gainNode.gain.setValueAtTime(Math.max(0.0001, volume), now);
    gainNode.gain.exponentialRampToValueAtTime(0.0001, now + duration);

    oscillator.start(now);
    oscillator.stop(now + duration);
  };

  const getAlarmVolume = (scoreValue: number | null, levelValue: InferenceResponse["level"] | "idle") => {
    if (scoreValue === null || !levelValue || levelValue === "idle" || levelValue === "safe") {
      return 0;
    }

    const s = Math.max(0, Math.min(100, scoreValue));

    // Base volume rises with score:
    // 30 -> low, 50 -> medium, 75+ -> loud
    let volume = 0.03 + (s / 100) * 0.12;

    // Make louder for more serious levels
    if (levelValue === "soft") volume *= 0.8;
    if (levelValue === "warning") volume *= 1.05;
    if (levelValue === "danger") volume *= 1.2;

    return Math.min(0.15, Math.max(0.02, volume));
  };

  const triggerAlarm = async (
    levelValue: InferenceResponse["level"] | "idle",
    scoreValue: number | null
  ) => {
    stopAlarm();

    if (
      scoreValue === null ||
      !levelValue ||
      levelValue === "idle" ||
      levelValue === "safe"
    ) {
      return;
    }

    await ensureAudioContext();

    const s = Math.max(0, Math.min(100, scoreValue));
    const volume = getAlarmVolume(scoreValue, levelValue);

    // Higher score => higher tone
    const baseFreq = 320 + s * 8; // approx 320Hz to 1120Hz

    const pattern =
      levelValue === "soft"
        ? { beeps: 1, repeat: 2400, gap: 220 }
        : levelValue === "warning"
        ? { beeps: 2, repeat: 1400, gap: 170 }
        : { beeps: 3, repeat: 750, gap: 130 }; // danger

    const playPattern = () => {
      for (let i = 0; i < pattern.beeps; i++) {
        window.setTimeout(() => {
          playBeep(baseFreq + i * 70, 0.14, volume);
        }, i * pattern.gap);
      }

      alarmTimerRef.current = window.setTimeout(playPattern, pattern.repeat);
    };

    playPattern();
  };

  // ---------- CAMERA ----------
  const stopCamera = () => {
    if (intervalRef.current) {
      window.clearInterval(intervalRef.current);
      intervalRef.current = null;
    }

    stopAlarm();

    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    setIsRunning(false);
  };

  const captureAndSend = async () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.videoWidth === 0 || video.videoHeight === 0) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const frameBase64 = canvas.toDataURL("image/jpeg", 0.8);

    try {
      const result = await sendFrame(sessionId, frameBase64);

      setBackendStatus(result.status);
      setSequenceLength(result.sequence_length);
      setScore(result.score);
      setLevel(result.level ?? "idle");
      setPrediction(result.prediction ?? "—");
      setMessage(result.message ?? "No message");
      setSource(result.source ?? "—");

      await triggerAlarm(result.level, result.score);
    } catch (error) {
      setBackendStatus("error");
      setMessage(error instanceof Error ? error.message : "Unknown error");
      stopAlarm();
    }
  };

  const startCamera = async () => {
    setPermissionError("");
    setIsStarting(true);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user",
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      });

      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      await ensureAudioContext();

      setIsRunning(true);
      setBackendStatus("running");
      setMessage("Camera started. Sending frames to backend...");

      intervalRef.current = window.setInterval(() => {
        void captureAndSend();
      }, 400);
    } catch (error) {
      setPermissionError(
        error instanceof Error
          ? error.message
          : "Could not access camera. Please allow camera permission."
      );
    } finally {
      setIsStarting(false);
    }
  };

  useEffect(() => {
    return () => {
      stopCamera();
      if (audioContextRef.current) {
        audioContextRef.current.close().catch(() => {});
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const badgeClass = statusColor(level);
  const borderClass = statusBorder(level);

  return (
    <div className="grid gap-6 lg:grid-cols-[1.6fr_1fr]">
      <div className={`rounded-3xl border ${borderClass} bg-slate-950 p-4 shadow-xl`}>
        <div className="mb-4 flex items-center justify-between gap-3">
          <div>
            <h2 className="text-xl font-semibold text-white">Live Driver Monitoring</h2>
            <p className="text-sm text-slate-400">
              Webcam capture → feature extraction → temporal inference
            </p>
          </div>

          <div className="flex items-center gap-2">
            {!isRunning ? (
              <button
                onClick={startCamera}
                disabled={isStarting}
                className="inline-flex items-center gap-2 rounded-2xl bg-white px-4 py-2 text-sm font-medium text-slate-950 transition hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-60"
              >
                <Play className="h-4 w-4" />
                {isStarting ? "Starting..." : "Start Camera"}
              </button>
            ) : (
              <button
                onClick={stopCamera}
                className="inline-flex items-center gap-2 rounded-2xl bg-slate-800 px-4 py-2 text-sm font-medium text-white transition hover:bg-slate-700"
              >
                <CircleStop className="h-4 w-4" />
                Stop
              </button>
            )}
          </div>
        </div>

        <div className="relative overflow-hidden rounded-3xl border border-slate-800 bg-black">
          <video
            ref={videoRef}
            className="aspect-video w-full object-cover"
            playsInline
            muted
            autoPlay
          />
          {!isRunning && (
            <div className="absolute inset-0 flex items-center justify-center bg-black/70">
              <div className="text-center">
                <Camera className="mx-auto mb-3 h-10 w-10 text-slate-300" />
                <p className="text-sm text-slate-300">Camera is off</p>
              </div>
            </div>
          )}
        </div>

        <canvas ref={canvasRef} className="hidden" />

        {permissionError ? (
          <div className="mt-4 rounded-2xl border border-red-500/30 bg-red-500/10 p-3 text-sm text-red-200">
            {permissionError}
          </div>
        ) : null}
      </div>

      <div className="grid gap-4">
        <div className="rounded-3xl border border-slate-800 bg-slate-950 p-4 shadow-xl">
          <div className="mb-4 flex items-center gap-2">
            <Radio className="h-5 w-5 text-cyan-400" />
            <h3 className="text-lg font-semibold text-white">Live Status</h3>
          </div>

          <div className="grid gap-3">
            <Stat label="Session ID" value={sessionId} />
            <Stat label="Backend" value={backendStatus} />
            <Stat label="Sequence" value={`${sequenceLength} / 30`} />
            <Stat label="Score" value={score === null ? "—" : score.toFixed(2)} />
            <Stat label="Prediction" value={prediction} />
            <Stat label="Source" value={source} />
          </div>
        </div>

        <div className={`rounded-3xl border ${borderClass} bg-slate-950 p-4 shadow-xl`}>
          <div className="mb-3 flex items-center gap-2">
            <ShieldAlert className="h-5 w-5 text-white" />
            <h3 className="text-lg font-semibold text-white">Alert State</h3>
          </div>

          <div
            className={`inline-flex items-center gap-2 rounded-full px-3 py-1 text-sm font-medium text-white ${badgeClass}`}
          >
            <AlertTriangle className="h-4 w-4" />
            {level === "idle" ? "idle" : level}
          </div>

          <p className="mt-4 text-sm leading-6 text-slate-300">{message}</p>

          <div className="mt-4 rounded-2xl border border-slate-800 bg-slate-900 p-3 text-xs text-slate-400">
            The system sends one frame per second. After 30 frames, the BiLSTM pipeline starts returning
            real predictions.
          </div>
        </div>
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between rounded-2xl border border-slate-800 bg-slate-900 px-4 py-3">
      <span className="text-sm text-slate-400">{label}</span>
      <span className="text-sm font-medium text-white">{value}</span>
    </div>
  );
}
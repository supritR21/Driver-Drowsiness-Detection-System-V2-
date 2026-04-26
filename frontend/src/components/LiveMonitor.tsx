"use client";

import { useEffect, useRef, useState } from "react";
import {
  AlertTriangle,
  Camera,
  CircleStop,
  Download,
  Play,
  Radio,
  ShieldAlert,
  SquareDot,
  SquareStop,
} from "lucide-react";
import { sendFrame } from "@/lib/api";
import type { InferenceResponse } from "@/types/inference";
import FaceMeshBox from "@/components/FaceMeshBox";

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

  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);

  const audioContextRef = useRef<AudioContext | null>(null);
  const alarmTimerRef = useRef<number | null>(null);

  const [sessionId] = useState("demo-session");
  const [isRunning, setIsRunning] = useState(false);
  const [isStarting, setIsStarting] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [downloadUrl, setDownloadUrl] = useState<string>("");

  const [backendStatus, setBackendStatus] = useState<string>("idle");
  const [sequenceLength, setSequenceLength] = useState<number>(0);
  const [score, setScore] = useState<number | null>(null);
  const [level, setLevel] = useState<InferenceResponse["level"] | "idle">("idle");
  const [prediction, setPrediction] = useState<string>("—");
  const [message, setMessage] = useState<string>("Start camera to begin monitoring.");
  const [source, setSource] = useState<string>("—");
  const [permissionError, setPermissionError] = useState<string>("");

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

  const getAlarmVolume = (
    scoreValue: number | null,
    levelValue: InferenceResponse["level"] | "idle"
  ) => {
    if (scoreValue === null || !levelValue || levelValue === "idle" || levelValue === "safe") {
      return 0;
    }

    const s = Math.max(0, Math.min(100, scoreValue));
    let volume = 0.03 + (s / 100) * 0.12;

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
    const baseFreq = 320 + s * 8;

    const pattern =
      levelValue === "soft"
        ? { beeps: 1, repeat: 2400, gap: 220 }
        : levelValue === "warning"
        ? { beeps: 2, repeat: 1400, gap: 170 }
        : { beeps: 3, repeat: 750, gap: 130 };

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

  const startRecording = () => {
    if (!streamRef.current) return;

    try {
      chunksRef.current = [];
      const recorder = new MediaRecorder(streamRef.current, {
        mimeType: "video/webm;codecs=vp8",
      });

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: "video/webm" });
        const url = URL.createObjectURL(blob);
        setDownloadUrl(url);
      };

      recorder.start();
      recorderRef.current = recorder;
      setIsRecording(true);
    } catch {
      setPermissionError("Recording is not supported in this browser.");
    }
  };

  const stopRecording = () => {
    if (recorderRef.current && recorderRef.current.state !== "inactive") {
      recorderRef.current.stop();
    }
    recorderRef.current = null;
    setIsRecording(false);
  };

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

    if (isRecording) {
      stopRecording();
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

      if (downloadUrl) {
        URL.revokeObjectURL(downloadUrl);
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const badgeClass = statusColor(level);
  const borderClass = statusBorder(level);

  return (
    <div className="grid h-full min-h-0 gap-4 xl:grid-cols-[1.1fr_1fr_0.9fr]">
      <section className={`flex min-h-0 flex-col rounded-3xl border ${borderClass} bg-slate-950 p-4 shadow-xl`}>
        <div className="mb-3 flex items-center justify-between gap-3">
          <div>
            <h2 className="text-xl font-semibold text-white">Live Driver Monitoring</h2>
            <p className="text-sm text-slate-400">Clear webcam feed</p>
          </div>

          <div className="flex flex-wrap items-center gap-2">
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

            {isRunning && !isRecording ? (
              <button
                onClick={startRecording}
                className="inline-flex items-center gap-2 rounded-2xl bg-red-500 px-4 py-2 text-sm font-medium text-white transition hover:bg-red-400"
              >
                <SquareDot className="h-4 w-4" />
                Record
              </button>
            ) : null}

            {isRecording ? (
              <button
                onClick={stopRecording}
                className="inline-flex items-center gap-2 rounded-2xl bg-red-700 px-4 py-2 text-sm font-medium text-white transition hover:bg-red-600"
              >
                <SquareStop className="h-4 w-4" />
                Stop Recording
              </button>
            ) : null}
          </div>
        </div>

        <div className="relative min-h-80 flex-1 overflow-hidden rounded-3xl border border-slate-800 bg-black">
          <video
            ref={videoRef}
            className="h-full w-full object-contain bg-black"
            playsInline
            muted
            autoPlay
          />

          <div className="absolute left-4 top-4 rounded-2xl border border-slate-700 bg-slate-950/80 px-4 py-3 backdrop-blur">
            <div className="text-xs uppercase tracking-[0.2em] text-slate-400">Score</div>
            <div className="text-3xl font-bold text-white">
              {score === null ? "—" : score.toFixed(1)}
            </div>
            <div className="mt-1 text-xs text-slate-400 capitalize">
              {level === "idle" ? "idle" : level}
            </div>
          </div>

          <div className="absolute right-4 top-4 flex items-center gap-2 rounded-full border border-slate-700 bg-slate-950/80 px-3 py-2 text-xs text-slate-200 backdrop-blur">
            <span
              className={`h-2.5 w-2.5 rounded-full ${
                isRecording ? "bg-red-500" : "bg-slate-500"
              }`}
            />
            {isRecording ? "Recording" : "Not recording"}
          </div>

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

        {downloadUrl ? (
          <div className="mt-3 flex items-center gap-3 rounded-2xl border border-slate-800 bg-slate-900 p-3">
            <Download className="h-4 w-4 text-cyan-400" />
            <a
              href={downloadUrl}
              download="drowsiness-recording.webm"
              className="text-sm font-medium text-cyan-400 hover:underline"
            >
              Download last recording
            </a>
          </div>
        ) : null}

        {permissionError ? (
          <div className="mt-3 rounded-2xl border border-red-500/30 bg-red-500/10 p-3 text-sm text-red-200">
            {permissionError}
          </div>
        ) : null}
      </section>

      <section className="flex min-h-0 flex-col rounded-3xl border border-slate-800 bg-slate-950 p-4 shadow-xl">
        <div className="mb-3 flex items-center gap-2">
          <Radio className="h-5 w-5 text-cyan-400" />
          <h3 className="text-lg font-semibold text-white">Live Status</h3>
        </div>

        <div className="grid gap-2">
          <Stat label="Session ID" value={sessionId} />
          <Stat label="Backend" value={backendStatus} />
          <Stat label="Sequence" value={`${sequenceLength} / 45`} />
          <Stat label="Score" value={score === null ? "—" : score.toFixed(2)} />
          <Stat label="Prediction" value={prediction} />
          <Stat label="Source" value={source} />
        </div>

        <div className={`mt-4 rounded-3xl border ${borderClass} bg-slate-900 p-4`}>
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

          <p className="mt-3 text-sm leading-6 text-slate-300">{message}</p>

          <div className="mt-3 rounded-2xl border border-slate-800 bg-slate-950 p-3 text-xs text-slate-400">
            Frames are sent every 0.4 seconds. The score updates live after the temporal window fills.
          </div>
        </div>
      </section>

      <section className="flex min-h-0 flex-col">
        <FaceMeshBox videoRef={videoRef} active={isRunning} />
      </section>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between rounded-2xl border border-slate-800 bg-slate-900 px-4 py-2.5">
      <span className="text-sm text-slate-400">{label}</span>
      <span className="text-sm font-medium text-white">{value}</span>
    </div>
  );
}
"use client";

import {
  useEffect,
  useRef,
  useState,
} from "react";

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

import {
  getTransformerWebSocketUrl,
} from "@/lib/api";

import type {
  InferenceLevel,
  InferenceResponse,
} from "@/types/inference";

import FaceMeshBox from "@/components/FaceMeshBox";


const TARGET_FRAME_INTERVAL_MS = 75;

const INFERENCE_WIDTH = 640;


function statusColor(
  level: InferenceLevel | "idle"
): string {

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


function statusBorder(
  level: InferenceLevel | "idle"
): string {

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

  // ==========================================
  // VIDEO / CANVAS
  // ==========================================

  const videoRef =
    useRef<HTMLVideoElement | null>(
      null
    );

  const canvasRef =
    useRef<HTMLCanvasElement | null>(
      null
    );

  const streamRef =
    useRef<MediaStream | null>(
      null
    );


  // ==========================================
  // TRANSFORMER WEBSOCKET
  // ==========================================

  const socketRef =
    useRef<WebSocket | null>(
      null
    );

  const animationFrameRef =
    useRef<number | null>(
      null
    );

  const awaitingResponseRef =
    useRef(false);

  const lastFrameSentAtRef =
    useRef(0);


  // ==========================================
  // RECORDING
  // ==========================================

  const recorderRef =
    useRef<MediaRecorder | null>(
      null
    );

  const chunksRef =
    useRef<BlobPart[]>(
      []
    );


  // ==========================================
  // ALARM
  // ==========================================

  const audioContextRef =
    useRef<AudioContext | null>(
      null
    );

  const alarmTimerRef =
    useRef<number | null>(
      null
    );

  const activeAlarmLevelRef =
    useRef<
      InferenceLevel | "idle"
    >(
      "idle"
    );


  // ==========================================
  // UI STATE
  // ==========================================

  const [sessionId] =
    useState(
      "demo-session"
    );

  const [
    isRunning,
    setIsRunning,
  ] = useState(false);

  const [
    isStarting,
    setIsStarting,
  ] = useState(false);

  const [
    isRecording,
    setIsRecording,
  ] = useState(false);

  const [
    downloadUrl,
    setDownloadUrl,
  ] = useState("");


  const [
    backendStatus,
    setBackendStatus,
  ] = useState("idle");


  const [
    sequenceLength,
    setSequenceLength,
  ] = useState(0);


  const [
    score,
    setScore,
  ] = useState<number | null>(
    null
  );


  const [
    level,
    setLevel,
  ] = useState<
    InferenceLevel | "idle"
  >(
    "idle"
  );


  const [
    prediction,
    setPrediction,
  ] = useState("—");


  const [
    message,
    setMessage,
  ] = useState(
    "Start camera to begin monitoring."
  );


  const [
    source,
    setSource,
  ] = useState("—");


  const [
    rawProbability,
    setRawProbability,
  ] = useState<number | null>(
    null
  );


  const [
    smoothedProbability,
    setSmoothedProbability,
  ] = useState<number | null>(
    null
  );


  const [
    voteRatio,
    setVoteRatio,
  ] = useState<number | null>(
    null
  );


  const [
    coverageSeconds,
    setCoverageSeconds,
  ] = useState(0);


  const [
    sourceSamples,
    setSourceSamples,
  ] = useState(0);


  const [
    permissionError,
    setPermissionError,
  ] = useState("");


  // ==========================================
  // AUDIO
  // ==========================================

  const ensureAudioContext =
    async () => {

      if (
        typeof window ===
        "undefined"
      ) {
        return null;
      }

      if (
        !audioContextRef.current
      ) {

        const AudioCtx =
          window.AudioContext ||
          (window as any)
            .webkitAudioContext;

        audioContextRef.current =
          new AudioCtx();
      }

      if (
        audioContextRef.current
          .state === "suspended"
      ) {

        await (
          audioContextRef.current
            .resume()
        );
      }

      return (
        audioContextRef.current
      );
    };


  const clearAlarmTimer =
    () => {

      if (
        alarmTimerRef.current
        !== null
      ) {

        window.clearTimeout(
          alarmTimerRef.current
        );

        alarmTimerRef.current =
          null;
      }
    };


  const stopAlarm =
    () => {

      clearAlarmTimer();

      activeAlarmLevelRef.current =
        "idle";
    };


  const playBeep = (
    frequency: number,
    duration = 0.15,
    volume = 0.08
  ) => {

    const ctx =
      audioContextRef.current;

    if (!ctx) {
      return;
    }

    const oscillator =
      ctx.createOscillator();

    const gainNode =
      ctx.createGain();

    oscillator.type =
      "sine";

    oscillator.frequency.value =
      frequency;

    oscillator.connect(
      gainNode
    );

    gainNode.connect(
      ctx.destination
    );

    const now =
      ctx.currentTime;

    gainNode.gain.setValueAtTime(
      Math.max(
        0.0001,
        volume
      ),
      now
    );

    gainNode.gain
      .exponentialRampToValueAtTime(
        0.0001,
        now + duration
      );

    oscillator.start(
      now
    );

    oscillator.stop(
      now + duration
    );
  };


  const getAlarmVolume = (
    scoreValue: number | null,
    levelValue:
      | InferenceLevel
      | "idle"
  ) => {

    if (
      scoreValue === null ||
      levelValue === "idle" ||
      levelValue === "safe"
    ) {

      return 0;
    }

    const s =
      Math.max(
        0,
        Math.min(
          100,
          scoreValue
        )
      );

    let volume =
      0.03 +
      (
        s / 100
      ) * 0.12;

    if (
      levelValue === "soft"
    ) {
      volume *= 0.8;
    }

    if (
      levelValue === "warning"
    ) {
      volume *= 1.05;
    }

    if (
      levelValue === "danger"
    ) {
      volume *= 1.2;
    }

    return Math.min(
      0.15,
      Math.max(
        0.02,
        volume
      )
    );
  };


  const updateAlarm =
    async (
      levelValue:
        | InferenceLevel
        | null,
      scoreValue:
        | number
        | null
    ) => {

      if (
        levelValue === null ||
        levelValue === "safe" ||
        scoreValue === null
      ) {

        stopAlarm();

        return;
      }


      // Do not restart the repeating
      // alarm every WebSocket response.
      if (
        activeAlarmLevelRef.current
        === levelValue
      ) {
        return;
      }


      clearAlarmTimer();

      activeAlarmLevelRef.current =
        levelValue;


      await ensureAudioContext();


      const s =
        Math.max(
          0,
          Math.min(
            100,
            scoreValue
          )
        );


      const volume =
        getAlarmVolume(
          scoreValue,
          levelValue
        );


      const baseFreq =
        320 + s * 8;


      const pattern =
        levelValue === "soft"

          ? {
              beeps: 1,
              repeat: 2400,
              gap: 220,
            }

          : levelValue ===
            "warning"

          ? {
              beeps: 2,
              repeat: 1400,
              gap: 170,
            }

          : {
              beeps: 3,
              repeat: 750,
              gap: 130,
            };


      const currentLevel =
        levelValue;


      const playPattern =
        () => {

          if (
            activeAlarmLevelRef
              .current
            !== currentLevel
          ) {
            return;
          }


          for (
            let i = 0;
            i < pattern.beeps;
            i++
          ) {

            window.setTimeout(
              () => {

                if (
                  activeAlarmLevelRef
                    .current
                  !== currentLevel
                ) {
                  return;
                }

                playBeep(
                  baseFreq
                    + i * 70,

                  0.14,

                  volume
                );

              },

              i * pattern.gap
            );
          }


          alarmTimerRef.current =
            window.setTimeout(
              playPattern,
              pattern.repeat
            );
        };


      playPattern();
    };


  // ==========================================
  // RECORDING
  // ==========================================

  const startRecording =
    () => {

      if (
        !streamRef.current
      ) {
        return;
      }

      try {

        chunksRef.current =
          [];


        const recorder =
          new MediaRecorder(
            streamRef.current,
            {
              mimeType:
                "video/webm;codecs=vp8",
            }
          );


        recorder.ondataavailable =
          (event) => {

            if (
              event.data.size > 0
            ) {

              chunksRef.current.push(
                event.data
              );
            }
          };


        recorder.onstop =
          () => {

            const blob =
              new Blob(
                chunksRef.current,
                {
                  type:
                    "video/webm",
                }
              );

            const url =
              URL.createObjectURL(
                blob
              );

            setDownloadUrl(
              url
            );
          };


        recorder.start();

        recorderRef.current =
          recorder;

        setIsRecording(
          true
        );

      } catch {

        setPermissionError(
          "Recording is not supported in this browser."
        );
      }
    };


  const stopRecording =
    () => {

      if (
        recorderRef.current &&
        recorderRef.current.state
          !== "inactive"
      ) {

        recorderRef.current.stop();
      }

      recorderRef.current =
        null;

      setIsRecording(
        false
      );
    };


  // ==========================================
  // FRAME CAPTURE
  // ==========================================

  const captureFrameBase64 =
    (): string | null => {

      const video =
        videoRef.current;

      const canvas =
        canvasRef.current;


      if (
        !video ||
        !canvas ||
        video.videoWidth === 0 ||
        video.videoHeight === 0
      ) {
        return null;
      }


      const aspectRatio =
        video.videoHeight
        / video.videoWidth;


      const width =
        INFERENCE_WIDTH;

      const height =
        Math.round(
          width * aspectRatio
        );


      canvas.width =
        width;

      canvas.height =
        height;


      const ctx =
        canvas.getContext(
          "2d"
        );


      if (!ctx) {
        return null;
      }


      ctx.drawImage(
        video,
        0,
        0,
        width,
        height
      );


      return canvas.toDataURL(
        "image/jpeg",
        0.65
      );
    };


  // ==========================================
  // BACKEND RESPONSE
  // ==========================================

  const applyInferenceResult =
    (
      result:
        InferenceResponse
    ) => {

      setBackendStatus(
        result.status
      );

      setSequenceLength(
        result.sequence_length
          ?? 0
      );

      setScore(
        result.score
          ?? null
      );

      setLevel(
        result.level
          ?? "idle"
      );

      setPrediction(
        result.prediction
          ?? "—"
      );

      setMessage(
        result.message
          ?? "No message"
      );

      setSource(
        result.source
          ?? "—"
      );

      setRawProbability(
        result.raw_probability
          ?? null
      );

      setSmoothedProbability(
        result.smoothed_probability
          ?? null
      );

      setVoteRatio(
        result.vote_ratio
          ?? null
      );

      setCoverageSeconds(
        result.coverage_seconds
          ?? 0
      );

      setSourceSamples(
        result.source_samples
          ?? 0
      );


      void updateAlarm(
        result.level,
        result.score
      );
    };


  // ==========================================
  // BACKPRESSURE FRAME LOOP
  // ==========================================

  const startFrameLoop =
    () => {

      lastFrameSentAtRef.current =
        0;


      const loop =
        (
          now: number
        ) => {

          const socket =
            socketRef.current;


          if (
            socket &&
            socket.readyState
              === WebSocket.OPEN &&
            !awaitingResponseRef.current &&
            (
              now
              - lastFrameSentAtRef.current
            )
              >= TARGET_FRAME_INTERVAL_MS
          ) {

            const frameBase64 =
              captureFrameBase64();


            if (frameBase64) {

              awaitingResponseRef.current =
                true;

              lastFrameSentAtRef.current =
                now;


              socket.send(
                JSON.stringify(
                  {
                    frame_base64:
                      frameBase64,
                  }
                )
              );
            }
          }


          animationFrameRef.current =
            window.requestAnimationFrame(
              loop
            );
        };


      animationFrameRef.current =
        window.requestAnimationFrame(
          loop
        );
    };


  // ==========================================
  // STOP CAMERA
  // ==========================================

  const stopCamera =
    () => {

      if (
        animationFrameRef.current
        !== null
      ) {

        window.cancelAnimationFrame(
          animationFrameRef.current
        );

        animationFrameRef.current =
          null;
      }


      awaitingResponseRef.current =
        false;


      if (
        socketRef.current
      ) {

        try {

          socketRef.current.close();

        } catch {
          // Ignore shutdown error.
        }

        socketRef.current =
          null;
      }


      stopAlarm();


      streamRef.current
        ?.getTracks()
        .forEach(
          (
            track
          ) =>
            track.stop()
        );


      streamRef.current =
        null;


      if (
        videoRef.current
      ) {

        videoRef.current.srcObject =
          null;
      }


      if (
        recorderRef.current &&
        recorderRef.current.state
          !== "inactive"
      ) {

        stopRecording();
      }


      setIsRunning(
        false
      );

      setBackendStatus(
        "stopped"
      );

      setSequenceLength(
        0
      );

      setCoverageSeconds(
        0
      );

      setSourceSamples(
        0
      );

      setRawProbability(
        null
      );

      setSmoothedProbability(
        null
      );

      setVoteRatio(
        null
      );

      setScore(
        null
      );

      setLevel(
        "idle"
      );

      setPrediction(
        "—"
      );

      setSource(
        "—"
      );

      setMessage(
        "Monitoring stopped."
      );
    };


  // ==========================================
  // START CAMERA + WEBSOCKET
  // ==========================================

  const startCamera =
    async () => {

      setPermissionError(
        ""
      );

      setIsStarting(
        true
      );


      try {

        const stream =
          await navigator
            .mediaDevices
            .getUserMedia(
              {
                video: {
                  facingMode:
                    "user",

                  width: {
                    ideal: 1280,
                  },

                  height: {
                    ideal: 720,
                  },
                },

                audio: false,
              }
            );


        streamRef.current =
          stream;


        if (
          videoRef.current
        ) {

          videoRef.current.srcObject =
            stream;

          await (
            videoRef.current.play()
          );
        }


        await ensureAudioContext();


        const socketUrl =
          getTransformerWebSocketUrl(
            sessionId
          );


        const socket =
          new WebSocket(
            socketUrl
          );


        socketRef.current =
          socket;


        setIsRunning(
          true
        );

        setBackendStatus(
          "connecting"
        );

        setMessage(
          "Connecting to Transformer inference..."
        );


        socket.onopen =
          () => {

            setBackendStatus(
              "connected"
            );

            setMessage(
              "Connected. Collecting approximately 2 seconds of temporal features..."
            );

            startFrameLoop();
          };


        socket.onmessage =
          (
            event
          ) => {

            // This is the core
            // backpressure release.
            awaitingResponseRef.current =
              false;


            try {

              const result =
                JSON.parse(
                  event.data
                ) as InferenceResponse;


              if (
                result.status
                === "error"
              ) {

                setBackendStatus(
                  "error"
                );

                setMessage(
                  result.message
                    ?? "Backend inference error."
                );

                stopAlarm();

                return;
              }


              applyInferenceResult(
                result
              );

            } catch {

              setBackendStatus(
                "error"
              );

              setMessage(
                "Invalid response from inference server."
              );

              stopAlarm();
            }
          };


        socket.onerror =
          () => {

            awaitingResponseRef.current =
              false;

            setBackendStatus(
              "error"
            );

            setMessage(
              "Transformer WebSocket connection error."
            );

            stopAlarm();
          };


        socket.onclose =
          () => {

            awaitingResponseRef.current =
              false;

            if (
              streamRef.current
            ) {

              setBackendStatus(
                "disconnected"
              );

              setMessage(
                "Transformer WebSocket disconnected."
              );
            }

            stopAlarm();
          };


      } catch (
        error
      ) {

        setPermissionError(
          error instanceof Error
            ? error.message
            : (
                "Could not access camera. "
                + "Please allow camera permission."
              )
        );


        streamRef.current
          ?.getTracks()
          .forEach(
            (
              track
            ) =>
              track.stop()
          );


        streamRef.current =
          null;

        setIsRunning(
          false
        );

      } finally {

        setIsStarting(
          false
        );
      }
    };


  // ==========================================
  // CLEANUP
  // ==========================================

  useEffect(
    () => {

      return () => {

        if (
          animationFrameRef.current
          !== null
        ) {

          window.cancelAnimationFrame(
            animationFrameRef.current
          );
        }


        socketRef.current
          ?.close();


        streamRef.current
          ?.getTracks()
          .forEach(
            (
              track
            ) =>
              track.stop()
          );


        clearAlarmTimer();


        if (
          audioContextRef.current
        ) {

          audioContextRef.current
            .close()
            .catch(
              () => {}
            );
        }
      };
    },
    []
  );


  const badgeClass =
    statusColor(
      level
    );


  const borderClass =
    statusBorder(
      level
    );


  return (

    <div className="grid h-full min-h-0 gap-4 xl:grid-cols-[1.1fr_1fr_0.9fr]">

      {/* ================================== */}
      {/* VIDEO */}
      {/* ================================== */}

      <section
        className={
          `flex min-h-0 flex-col ` +
          `rounded-3xl border ` +
          `${borderClass} ` +
          `bg-slate-950 p-4 shadow-xl`
        }
      >

        <div className="mb-3 flex items-center justify-between gap-3">

          <div>

            <h2 className="text-xl font-semibold text-white">
              Live Driver Monitoring
            </h2>

            <p className="text-sm text-slate-400">
              BiLSTM + Transformer temporal monitoring
            </p>

          </div>


          <div className="flex flex-wrap items-center gap-2">

            {!isRunning ? (

              <button
                onClick={
                  startCamera
                }
                disabled={
                  isStarting
                }
                className="inline-flex items-center gap-2 rounded-2xl bg-white px-4 py-2 text-sm font-medium text-slate-950 transition hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-60"
              >

                <Play className="h-4 w-4" />

                {
                  isStarting
                    ? "Starting..."
                    : "Start Camera"
                }

              </button>

            ) : (

              <button
                onClick={
                  stopCamera
                }
                className="inline-flex items-center gap-2 rounded-2xl bg-slate-800 px-4 py-2 text-sm font-medium text-white transition hover:bg-slate-700"
              >

                <CircleStop className="h-4 w-4" />

                Stop

              </button>
            )}


            {
              isRunning &&
              !isRecording
                ? (

                  <button
                    onClick={
                      startRecording
                    }
                    className="inline-flex items-center gap-2 rounded-2xl bg-red-500 px-4 py-2 text-sm font-medium text-white transition hover:bg-red-400"
                  >

                    <SquareDot className="h-4 w-4" />

                    Record

                  </button>

                )
                : null
            }


            {
              isRecording
                ? (

                  <button
                    onClick={
                      stopRecording
                    }
                    className="inline-flex items-center gap-2 rounded-2xl bg-red-700 px-4 py-2 text-sm font-medium text-white transition hover:bg-red-600"
                  >

                    <SquareStop className="h-4 w-4" />

                    Stop Recording

                  </button>

                )
                : null
            }

          </div>

        </div>


        <div className="relative min-h-80 flex-1 overflow-hidden rounded-3xl border border-slate-800 bg-black">

          <video
            ref={
              videoRef
            }
            className="h-full w-full object-contain bg-black"
            playsInline
            muted
            autoPlay
          />


          <div className="absolute left-4 top-4 rounded-2xl border border-slate-700 bg-slate-950/80 px-4 py-3 backdrop-blur">

            <div className="text-xs uppercase tracking-[0.2em] text-slate-400">
              Score
            </div>

            <div className="text-3xl font-bold text-white">

              {
                score === null
                  ? "—"
                  : score.toFixed(
                      1
                    )
              }

            </div>

            <div className="mt-1 text-xs text-slate-400 capitalize">

              {
                level === "idle"
                  ? "idle"
                  : level
              }

            </div>

          </div>


          <div className="absolute right-4 top-4 flex items-center gap-2 rounded-full border border-slate-700 bg-slate-950/80 px-3 py-2 text-xs text-slate-200 backdrop-blur">

            <span
              className={
                `h-2.5 w-2.5 rounded-full ${
                  isRecording
                    ? "bg-red-500"
                    : "bg-slate-500"
                }`
              }
            />

            {
              isRecording
                ? "Recording"
                : "Not recording"
            }

          </div>


          {!isRunning && (

            <div className="absolute inset-0 flex items-center justify-center bg-black/70">

              <div className="text-center">

                <Camera className="mx-auto mb-3 h-10 w-10 text-slate-300" />

                <p className="text-sm text-slate-300">
                  Camera is off
                </p>

              </div>

            </div>
          )}

        </div>


        <canvas
          ref={
            canvasRef
          }
          className="hidden"
        />


        {
          downloadUrl
            ? (

              <div className="mt-3 flex items-center gap-3 rounded-2xl border border-slate-800 bg-slate-900 p-3">

                <Download className="h-4 w-4 text-cyan-400" />

                <a
                  href={
                    downloadUrl
                  }
                  download="drowsiness-recording.webm"
                  className="text-sm font-medium text-cyan-400 hover:underline"
                >
                  Download last recording
                </a>

              </div>

            )
            : null
        }


        {
          permissionError
            ? (

              <div className="mt-3 rounded-2xl border border-red-500/30 bg-red-500/10 p-3 text-sm text-red-200">

                {
                  permissionError
                }

              </div>

            )
            : null
        }

      </section>


      {/* ================================== */}
      {/* TRANSFORMER STATUS */}
      {/* ================================== */}

      <section className="flex min-h-0 flex-col rounded-3xl border border-slate-800 bg-slate-950 p-4 shadow-xl">

        <div className="mb-3 flex items-center gap-2">

          <Radio className="h-5 w-5 text-cyan-400" />

          <h3 className="text-lg font-semibold text-white">
            Transformer Status
          </h3>

        </div>


        <div className="grid gap-2">

          <Stat
            label="Session"
            value={
              sessionId
            }
          />

          <Stat
            label="Backend"
            value={
              backendStatus
            }
          />

          <Stat
            label="Sequence"
            value={
              `${sequenceLength} / 60`
            }
          />

          <Stat
            label="Temporal coverage"
            value={
              `${coverageSeconds.toFixed(2)} s`
            }
          />

          <Stat
            label="Source frames"
            value={
              sourceSamples.toString()
            }
          />

          <Stat
            label="Raw probability"
            value={
              rawProbability === null
                ? "—"
                : rawProbability
                    .toFixed(
                      4
                    )
            }
          />

          <Stat
            label="EMA probability"
            value={
              smoothedProbability === null
                ? "—"
                : smoothedProbability
                    .toFixed(
                      4
                    )
            }
          />

          <Stat
            label="Vote ratio"
            value={
              voteRatio === null
                ? "—"
                : (
                    voteRatio
                    * 100
                  ).toFixed(
                    0
                  )
                  + "%"
            }
          />

          <Stat
            label="Score"
            value={
              score === null
                ? "—"
                : score.toFixed(
                    2
                  )
            }
          />

          <Stat
            label="Prediction"
            value={
              prediction
            }
          />

          <Stat
            label="Source"
            value={
              source
            }
          />

        </div>


        <div
          className={
            `mt-4 rounded-3xl border ` +
            `${borderClass} ` +
            `bg-slate-900 p-4`
          }
        >

          <div className="mb-3 flex items-center gap-2">

            <ShieldAlert className="h-5 w-5 text-white" />

            <h3 className="text-lg font-semibold text-white">
              Alert State
            </h3>

          </div>


          <div
            className={
              `inline-flex items-center gap-2 ` +
              `rounded-full px-3 py-1 ` +
              `text-sm font-medium text-white ` +
              `${badgeClass}`
            }
          >

            <AlertTriangle className="h-4 w-4" />

            {
              level === "idle"
                ? "idle"
                : level
            }

          </div>


          <p className="mt-3 text-sm leading-6 text-slate-300">

            {
              message
            }

          </p>


          <div className="mt-3 rounded-2xl border border-slate-800 bg-slate-950 p-3 text-xs leading-5 text-slate-400">

            Frames use a persistent WebSocket with backpressure.
            The backend builds an approximately 2-second temporal
            window, resamples it to 60 steps, and applies
            Transformer inference, EMA and temporal voting.

          </div>

        </div>

      </section>


      {/* ================================== */}
      {/* FACE LANDMARK VIEW */}
      {/* ================================== */}

      <section className="flex min-h-0 flex-col">

        <FaceMeshBox
          videoRef={
            videoRef
          }
          active={
            isRunning
          }
        />

      </section>

    </div>
  );
}


function Stat({
  label,
  value,
}: {
  label: string;
  value: string;
}) {

  return (

    <div className="flex items-center justify-between rounded-2xl border border-slate-800 bg-slate-900 px-4 py-2.5">

      <span className="text-sm text-slate-400">
        {label}
      </span>

      <span className="text-sm font-medium text-white">
        {value}
      </span>

    </div>
  );
}
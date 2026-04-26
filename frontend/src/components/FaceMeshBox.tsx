"use client";

import { RefObject, useEffect, useRef, useState } from "react";

type Props = {
  videoRef: RefObject<HTMLVideoElement | null>;
  active: boolean;
};

export default function FaceMeshBox({ videoRef, active }: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const [status, setStatus] = useState("idle");
  const [landmarkCount, setLandmarkCount] = useState(0);

  useEffect(() => {
    if (!active || !videoRef.current || !canvasRef.current) return;

    let cancelled = false;
    let rafId = 0;
    let faceMesh: any = null;

    const start = async () => {
      const faceMeshModule = await import("@mediapipe/face_mesh");
      const drawingModule = await import("@mediapipe/drawing_utils");

      const FaceMesh = faceMeshModule.FaceMesh;
      const { drawLandmarks } = drawingModule;

      const video = videoRef.current!;
      const canvas = canvasRef.current!;
      const ctx = canvas.getContext("2d");

      if (!ctx) return;

      faceMesh = new FaceMesh({
        locateFile: (file: string) =>
          `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`,
      });

      faceMesh.setOptions({
        maxNumFaces: 1,
        refineLandmarks: true,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
        selfieMode: true,
      });

      faceMesh.onResults((results: any) => {
        const w = video.videoWidth || 640;
        const h = video.videoHeight || 480;

        canvas.width = w;
        canvas.height = h;

        ctx.clearRect(0, 0, w, h);

        if (results.image) {
          ctx.drawImage(results.image, 0, 0, w, h);
        }

        if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
          const landmarks = results.multiFaceLandmarks[0];

          setStatus("face detected");
          setLandmarkCount(landmarks.length);

          drawLandmarks(ctx, landmarks, {
            color: "#22c55e",
            lineWidth: 1,
            radius: 1.5,
          });
        } else {
          setStatus("no face");
          setLandmarkCount(0);
        }
      });

      const loop = async () => {
        if (cancelled) return;

        try {
          if (video.readyState >= 2) {
            await faceMesh.send({ image: video });
          }
        } catch {}

        rafId = requestAnimationFrame(loop);
      };

      loop();
    };

    start();

    return () => {
      cancelled = true;
      cancelAnimationFrame(rafId);
      if (faceMesh) {
        faceMesh.close();
      }
    };
  }, [active, videoRef]);

  return (
    <div className="flex h-full min-h-0 flex-col rounded-3xl border border-slate-800 bg-slate-950 p-4 shadow-xl">
      <div className="mb-3 flex items-center justify-between">
        <div>
          <h3 className="text-lg font-semibold text-white">Face Points View</h3>
          <p className="text-sm text-slate-400">Real-time facial landmarks</p>
        </div>

        <div className="rounded-full border border-slate-700 px-3 py-1 text-xs text-slate-300">
          {status}
        </div>
      </div>

      <div className="relative min-h-80 flex-1 overflow-hidden rounded-3xl border border-slate-800 bg-black">
        <canvas
          ref={canvasRef}
          className="h-full w-full object-contain"
          style={{ display: "block" }}
        />
      </div>

      <div className="mt-3 grid grid-cols-2 gap-3">
        <InfoCard label="Landmarks" value={String(landmarkCount)} />
        <InfoCard label="Mode" value="MediaPipe" />
      </div>

      <div className="mt-3 rounded-2xl border border-slate-800 bg-slate-900 p-3 text-xs text-slate-400">
        Green points are live landmarks used to estimate drowsiness.
      </div>
    </div>
  );
}

function InfoCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-2xl border border-slate-800 bg-slate-900 px-4 py-3">
      <div className="text-xs text-slate-400">{label}</div>
      <div className="mt-1 text-sm font-semibold text-white">{value}</div>
    </div>
  );
}
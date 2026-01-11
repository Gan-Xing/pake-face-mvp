"use client";

import dynamic from "next/dynamic";

const FaceDemo = dynamic(() => import("./face-demo"), { ssr: false });

export default function FacePage() {
  return (
    <main className="page">
      <section className="panel header">
        <p className="tag">Local-only AI</p>
        <h1 className="title">Face Register / Check in</h1>
        <p className="subtitle">
          SCRFD 找脸 + ArcFace embedding + IndexedDB。
        </p>
      </section>
      <section className="panel">
        <FaceDemo />
      </section>
      <p className="footer">
        模型放在 <code>public/models</code>（人脸检测 + embedding）。
      </p>
    </main>
  );
}

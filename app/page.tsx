import Link from "next/link";

export default function Home() {
  return (
    <main className="page" style={{ justifyContent: "center" }}>
      <div className="panel" style={{ textAlign: "center", maxWidth: "600px" }}>
        <h1 className="title">Face Attendance App</h1>
        <p className="subtitle" style={{ marginBottom: "40px" }}>
          基于 Electron + MediaPipe + ArcFace 的离线人脸识别系统
        </p>

        <div className="grid" style={{ gridTemplateColumns: "1fr 1fr" }}>
          <Link href="/face">
            <button style={{ width: "100%", padding: "20px", fontSize: "18px" }}>
              进入 Demo 1<br />
              <span style={{ fontSize: "14px", opacity: 0.7 }}>(原有版本)</span>
            </button>
          </Link>

          <Link href="/demo2">
            <button className="primary" style={{ width: "100%", padding: "20px", fontSize: "18px" }}>
              进入 Demo 2<br />
              <span style={{ fontSize: "14px", opacity: 0.7 }}>(重构极简版)</span>
            </button>
          </Link>
        </div>
      </div>
    </main>
  );
}
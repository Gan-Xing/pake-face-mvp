export default function HomePage() {
  return (
    <main className="page">
      <section className="panel header">
        <p className="tag">Offline Face Demo</p>
        <h1 className="title">Next.js + WebView 离线人脸识别</h1>
        <p className="subtitle">
          进入演示页面开始录入两个人的 face embedding，再试试打卡识别。
        </p>
        <div className="row">
          <a className="tag" href="/face">
            打开 /face
          </a>
        </div>
      </section>
    </main>
  );
}

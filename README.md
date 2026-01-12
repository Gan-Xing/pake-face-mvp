# Pake Face MVP - 离线人脸识别考勤系统

这是一个基于 **Electron** + **Next.js** + **Hybrid AI (MediaPipe + ArcFace)** 构建的高性能、完全离线的桌面端人脸识别考勤应用。

## ✨ 项目特性

*   **双引擎架构**：结合 MediaPipe（前端快速检测/活体）与 ArcFace（后端高精度识别），兼顾速度与准确率。
*   **完全离线**：数据与模型均存储在本地，保护隐私，无网络依赖。
*   **活体检测**：内置眨眼检测 (EAR) 算法，有效防御照片/视频攻击。
*   **自动化考勤**：识别成功自动打卡，支持抓拍存档与考勤记录导出。

## 🛠 环境要求

*   **Node.js**: v18 或更高版本
*   **包管理器**: pnpm (主项目), npm (Electron子项目)
*   **操作系统**: macOS (推荐 M1/M2), Windows, Linux

---

## 🚀 快速上手 (Installation)

本项目采用混合依赖管理策略，请严格按照以下步骤安装。

### 1. 安装项目依赖

**第一步：安装前端依赖 (根目录)**
```bash
pnpm install
```

**第二步：安装 Electron 依赖**
*注意：Electron 目录下必须使用 `npm` 以确保原生模块打包正确。*
```bash
cd electron
npm install
cd ..
```

### 2. 下载 AI 模型
运行脚本自动下载所需的人脸检测与识别模型。
```bash
pnpm run download:models
```

### 3. 启动开发环境
启动包含热重载的开发模式：
```bash
pnpm run dev:build
```
应用启动后，点击主页的 **"Demo 2"** 即可体验完整功能。

---

## 📦 打包指南 (Build)

请按顺序执行以下命令生成生产环境安装包。

### 1. 构建前端
编译 Next.js 静态资源：
```bash
pnpm run build
```

### 2. 同步资源
将前端资源复制到 Electron 容器：
```bash
npx cap copy electron
```

### 3. 编译主进程与打包
进入 Electron 目录，编译 TypeScript 并生成应用包：

```bash
cd electron

# 编译主进程代码
npx tsc

# 打包为 macOS 应用 (.app)
npx electron-builder build --mac --dir -c ./electron-builder.config.json -p never

# (可选) 打包为安装镜像 (.dmg)
# npx electron-builder build --mac dmg -c ./electron-builder.config.json -p never
```

打包完成后，可执行文件位于 `electron/dist/mac-arm64/` (Apple Silicon) 或 `electron/dist/mac/` (Intel) 目录下。

---

## 📂 项目结构

*   `app/demo2` - **前端核心业务** (React/Next.js)
    *   `hooks/useCamera.ts` - 摄像头控制逻辑
    *   `hooks/useFaceDetection.ts` - MediaPipe 前端检测逻辑
*   `electron/` - **桌面主进程**
    *   `src/face/native-face.ts` - ArcFace 后端推理引擎 (ONNX)
    *   `entitlements.mac.plist` - macOS 硬件权限授权文件
*   `public/models` - AI 模型文件

## 📄 License

MIT
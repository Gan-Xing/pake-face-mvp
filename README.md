# Pake Face MVP - 离线人脸识别考勤系统

这是一个基于 **Electron** + **Next.js** + **Google MediaPipe** 构建的现代化、离线人脸识别考勤系统演示 (MVP)。

它展示了如何使用纯前端技术栈（在 Electron 容器中）实现高性能的人脸检测、特征提取、活体检测和考勤记录管理，所有数据均存储在本地 IndexedDB 中，无需联网。

![Demo Screenshot](https://via.placeholder.com/800x450?text=Face+Demo+Screenshot)

## ✨ 核心功能

*   **👥 多角度人脸注册**
    *   支持摄像头抓拍和本地图片上传。
    *   **质量检测**：实时分析人脸清晰度、角度、距离，引导用户录入高质量人脸。
    *   **特征融合**：自动融合多张照片的特征向量，提高识别准确率。

*   **⚡️ 实时自动考勤**
    *   **活体检测**：通过检测眨眼动作（Eye Aspect Ratio）防止照片/视频攻击。
    *   **自动打卡**：识别成功后自动记录考勤，无需手动点击。
    *   **防刷机制**：内置冷却时间（如 30秒），防止短时间内重复记录。

*   **📊 数据与记录管理**
    *   **实时抓拍**：每次考勤成功都会抓拍当前画面存档。
    *   **记录查询**：查看考勤时间、人员、相似度及抓拍大图。
    *   **用户管理**：支持修改用户名、删除用户。

## 🛠 技术栈

*   **框架**: [Electron](https://www.electronjs.org/), [Next.js](https://nextjs.org/), [React](https://react.dev/)
*   **AI 模型**: [Google MediaPipe](https://developers.google.com/mediapipe) (Face Mesh & Face Detection)
*   **数据库**: IndexedDB (本地离线存储)
*   **样式**: CSS Modules (响应式布局)
*   **语言**: TypeScript

## 🚀 快速开始

### 1. 环境准备
确保您的电脑上安装了 [Node.js](https://nodejs.org/) (推荐 v18+)。

### 2. 安装依赖
```bash
# 使用 pnpm (推荐)
pnpm install

# 或者 npm
npm install
```

### 3. 下载 AI 模型 (关键步骤！)
本项目依赖 MediaPipe 的模型文件，为了减小仓库体积，模型文件未包含在 git 中。请运行以下命令从服务器下载：

```bash
pnpm run download:models
```
*(该脚本会将模型下载到 `public/mediapipe` 目录)*

### 4. 启动开发环境 (Electron 模式)
这是开发桌面应用的推荐方式，包含热重载：

```bash
pnpm run dev:build
```
*(该命令会先编译 Next.js，然后启动 Electron 窗口)*

如果您只想在浏览器中调试网页版逻辑，可以使用：
```bash
pnpm dev
```

*   **Demo 入口**: 点击主页的 **"Demo 2: 完整考勤系统"** 进入核心功能演示。

## 📦 构建与打包

构建生产环境版本（生成可执行文件）：

```bash
pnpm run build
```
构建产物将位于 `dist/` 或 `electron/dist/` 目录下（取决于 electron-builder 配置）。

## 📂 项目结构

*   `app/demo2/` - **核心业务代码**
    *   `page.tsx` - 考勤主页（逻辑入口）
    *   `register/` - 注册页面
    *   `components/` - UI 组件 (控制面板, 列表, 弹窗等)
    *   `hooks/` - 核心逻辑 Hooks (`useFaceDetection`, `useLiveness` 等)
*   `lib/face/` - **人脸算法库**
    *   `utils.ts` - 几何计算、EAR算法、质量评估
    *   `storage.ts` - IndexedDB 数据库操作
    *   `similarity.ts` - 余弦相似度计算
*   `public/mediapipe/` - AI 模型文件 (需下载)
*   `electron/` - Electron 主进程代码

## ⚠️ 注意事项

1.  **模型加载**：首次进入页面时需要加载 WASM 和模型文件，可能需要几秒钟，请耐心等待“就绪”状态。
2.  **摄像头权限**：Electron 或浏览器会请求摄像头权限，请务必允许。
3.  **性能**：为了平衡功耗，考勤检测帧率默认限制在 10 FPS。

## 📄 License

MIT

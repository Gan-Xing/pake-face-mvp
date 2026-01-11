const fs = require('fs');
const path = require('path');
const https = require('https');

const BASE_URL = 'https://static.byganxing.com/mediapipe';
const DEST_DIR = path.join(__dirname, '../public/mediapipe');

const FILES = {
  face_detection: [
    'face_detection_full_range_sparse.tflite',
    'face_detection_full_range.tflite',
    'face_detection_full.binarypb',
    'face_detection_short_range.tflite',
    'face_detection_short.binarypb',
    'face_detection_solution_simd_wasm_bin.data',
    'face_detection_solution_simd_wasm_bin.js',
    'face_detection_solution_simd_wasm_bin.wasm', // Added
    'face_detection_solution_wasm_bin.js',
    'face_detection_solution_wasm_bin.wasm', // Added
    'face_detection.js',
    'index.d.ts',
    'package.json',
    'README.md'
  ],
  face_mesh: [
    'face_mesh_solution_packed_assets_loader.js',
    'face_mesh_solution_packed_assets.data', // Added back
    'face_mesh_solution_simd_wasm_bin.data',
    'face_mesh_solution_simd_wasm_bin.js',
    'face_mesh_solution_simd_wasm_bin.wasm', // Added
    'face_mesh_solution_wasm_bin.js',
    'face_mesh_solution_wasm_bin.wasm', // Added
    'face_mesh.binarypb',
    'face_mesh.js',
    'index.d.ts',
    'package.json',
    'README.md'
  ]
};

async function downloadFile(url, dest) {
  return new Promise((resolve, reject) => {
    const file = fs.createWriteStream(dest);
    https.get(url, (response) => {
      if (response.statusCode !== 200) {
        reject(new Error(`Failed to download ${url}: Status Code ${response.statusCode}`));
        return;
      }
      response.pipe(file);
      file.on('finish', () => {
        file.close();
        resolve();
      });
    }).on('error', (err) => {
      fs.unlink(dest, () => {}); // Delete partial file
      reject(err);
    });
  });
}

async function main() {
  console.log('🚀 Starting MediaPipe model download...');

  for (const [folder, files] of Object.entries(FILES)) {
    const folderPath = path.join(DEST_DIR, folder);
    if (!fs.existsSync(folderPath)) {
      fs.mkdirSync(folderPath, { recursive: true });
      console.log(`Created directory: ${folderPath}`);
    }

    for (const file of files) {
      const url = `${BASE_URL}/${folder}/${file}`;
      const dest = path.join(folderPath, file);
      
      if (fs.existsSync(dest)) {
        // Optional: Skip if exists? Or overwrite? 
        // For now, let's overwrite to ensure latest version, 
        // or we could check file size. Overwriting is safer for correctness.
        // console.log(`  - File exists (overwriting): ${file}`);
      }

      try {
        process.stdout.write(`  ⬇️  Downloading ${folder}/${file}... `);
        await downloadFile(url, dest);
        console.log('✅ Done');
      } catch (err) {
        console.error(`\n❌ Error downloading ${file}:`, err.message);
        // process.exit(1); // Don't exit, try next file
      }
    }
  }

  console.log('\n🎉 All models downloaded successfully!');
}

main();

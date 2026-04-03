/**
 * electron-builder configuration
 * Run: npm run build
 */
module.exports = {
  appId: "com.visionaryai.desktop",
  productName: "VisionaryAI",
  copyright: "Copyright © 2026 VisionaryAI",

  directories: {
    output: "dist-app",
    buildResources: "assets",
  },

  files: [
    "electron/**/*",
    "frontend/dist/**/*",
    "node_modules/better-sqlite3/**/*",
    "!node_modules/better-sqlite3/build/Release/test*",
  ],

  extraResources: [
    {
      from: "backend",
      to: "backend",
      filter: [
        "**/*",
        "!**/__pycache__/**",
        "!**/*.pyc",
        "!**/*.pyo",
        "!**/logs/**",
        "!**/training_curves.png",
        "!**/confusion_matrix.png",
      ],
    },
  ],

  win: {
    target: [{ target: "nsis", arch: ["x64"] }],
    icon: "assets/icon.ico",
    requestedExecutionLevel: "asInvoker",
  },

  nsis: {
    oneClick: false,
    allowToChangeInstallationDirectory: true,
    createDesktopShortcut: true,
    createStartMenuShortcut: true,
    shortcutName: "VisionaryAI",
    installerIcon: "assets/icon.ico",
    uninstallerIcon: "assets/icon.ico",
  },

  mac: {
    target: [{ target: "dmg", arch: ["x64", "arm64"] }],
    icon: "assets/icon.icns",
    category: "public.app-category.medical",
  },

  linux: {
    target: [{ target: "AppImage", arch: ["x64"] }],
    icon: "assets/icon.png",
    category: "Science",
  },
};

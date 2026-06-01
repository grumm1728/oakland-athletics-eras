# AGENTS.md

## Project Overview

This repo contains a React + D3 Vite app for exploring Oakland Athletics roster eras from 1968 through 2024. It combines Lahman roster/playing-time data with Baseball-Reference bWAR data, then serves interactive views plus static exports.

## Common Commands

```powershell
npm.cmd install
npm.cmd run build
npm.cmd run dev -- --port 5173
npm.cmd run serve:dist
```

Data pipeline commands:

```powershell
npm.cmd run pipeline
npm.cmd run pipeline:download
```

Use `pipeline` when raw data is already cached. Use `pipeline:download` when refreshing or recovering missing source data.

## Important Paths

- `src/main.jsx`: React app and visualization components.
- `src/styles.css`: App styling.
- `scripts/build_oakland_eras.py`: Data pipeline and export generation.
- `data/raw/`: Cached source CSV/text data.
- `data/processed/`: Generated analysis CSV/JSON/Markdown outputs.
- `public/data/`: Browser-served data files.
- `public/exports/`: Browser-served static SVG/PNG exports.
- `.github/workflows/deploy-pages.yml`: GitHub Pages deployment workflow.
- `vite.config.js`: Vite config, including the GitHub Pages base path.

## Deployment Notes

GitHub Pages is deployed through GitHub Actions on pushes to `main`.

The app is served at:

```text
https://grumm1728.github.io/oakland-athletics-eras/
```

Because this is a project Pages site, public asset links must respect Vite's `import.meta.env.BASE_URL`. Avoid hard-coded root paths like `/data/app_data.json` or `/exports/file.svg` in app code.

## Guardrails

- Do not commit `node_modules/`, `dist/`, Vite caches, Python caches, or preview logs.
- Keep generated browser data mirrored in `public/data/` when the app needs it.
- Run `npm.cmd run build` before pushing app or deployment changes.
- Treat `data/raw/` as cached source data; avoid changing it unless intentionally refreshing source inputs.
- Keep changes focused. The data pipeline, visualization code, and deployment workflow are separate concerns.

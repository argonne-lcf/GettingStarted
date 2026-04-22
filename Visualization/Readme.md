# ParaView Blood Flow Visualization — ALCF Tutorial

This repository contains resources for the **ParaView Interactive Visualization & Automation** tutorial developed for ALCF (Argonne Leadership Computing Facility) systems. The materials demonstrate how to use ParaView's Python scripting interface to visualize blood flow simulation data, including red blood cells, artery geometry, and fluid particles.

---

## Files

### 📖 Tutorial: [ParaviewPython-3.md](./ParaviewPython-3.md)

A step-by-step tutorial covering ParaView scientific visualization on ALCF systems (Polaris, Sophia, Aurora). By following this guide you will learn how to:

1. **Set up an interactive ParaView session** on an ALCF compute node — launching a job, loading the ParaView module (up to v6.0.1-EGL), starting `pvserver`, and connecting via SSH tunnel.
2. **Save and reload visualization states** as portable Python `.py` state files.
3. **Capture GUI interactions as Python code** using ParaView's built-in trace feature, which translates every click into `paraview.simple` API calls.
4. **Automate camera rotation animations** — both via the GUI's Animation Panel and via custom Python scripts for full control over orbit path and viewpoint.
5. **Export frame-by-frame animations** and stitch them into a video using `ffmpeg`.

---

### 🐍 Example Script

> ⬇️ [Download blood_test_01.py](./blood_test_01.py?raw=true)

A Python script example generated using ParaView 6.0.0. It demonstrates a complete visualization pipeline for blood flow simulation data, including:
- Loading time-series VTP (red blood cells — `rbcbad` and `rbcgood`) and VTU (artery mesh and fluid particles) datasets over 100 timesteps
- Configuring surface representations with custom colors (red for healthy RBCs, cyan for abnormal RBCs, semi-transparent artery)
- Applying a Glyph filter with sphere glyphs to particle data, colored by velocity magnitude
- Setting up a 3D render view with camera, lighting, and colormap configurations

---

### 🗂️ ParaView State File

> ⬇️ [Download dev_session_blood_01-2.pvsm](./dev_session_blood_01-2.pvsm?raw=true)

A ParaView state file (`.pvsm` XML format) encoding a complete visualization session for the blood flow dataset. Load it in ParaView via **File → Load State** to instantly restore the full pipeline, including:
- Four data sources: artery mesh (VTU), fluid particles (VTU), healthy RBCs (VTP), and abnormal RBCs (VTP)
- Sphere glyph representation for particles colored by velocity
- Velocity scalar bar and custom colormaps
- Camera position and lighting setup

---

## Requirements

- [ParaView](https://www.paraview.org/) 6.0.0 or later
- Blood flow tutorial dataset (VTP/VTU files) from the ALCF training data directory


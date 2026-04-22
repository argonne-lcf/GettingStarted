# Visualization at ALCF: Beyond the Basics

This repository contains resources for the **Beyond the Basics** tutorial developed for ALCF systems.

---

## Files

### Tutorial: [ParaViewPython.md](./ParaViewPython.md)

A step-by-step tutorial covering ParaView scientific visualization on ALCF systems (Polaris, Sophia, Aurora, Crux). By following this guide you will learn how to:

1. **Set up an interactive ParaView session** on an ALCF compute node — launching a job, loading the ParaView module, starting `pvserver`, and connecting via SSH tunnel.
2. **Save and reload visualization states** as portable Python `.py` state files.
3. **Capture GUI interactions as Python code** using ParaView's built-in trace feature, which translates every click into `paraview.simple` API calls.
4. **Automate camera rotation animations** — both via the GUI's Animation Panel and via custom Python scripts for full control over orbit path and viewpoint.
5. **Export frame-by-frame animations** and stitch them into a video using `ffmpeg`.

---

### ParaView State File

> [dev_session_blood_01.pvsm](./dev_session_blood_01.pvsm)

A ParaView state file (`.pvsm` XML format) encoding a complete visualization session for the blood flow dataset. Load it in ParaView via **File → Load State** to instantly restore the full pipeline, including:
- Four data sources: artery mesh (VTU), fluid particles (VTU), healthy RBCs (VTP), and abnormal RBCs (VTP)
- Sphere glyph representation for particles colored by velocity
- Velocity scalar bar and custom colormaps
- Camera position and lighting setup

---

### Example Python Batch Script

> [blood_test_01.py](./blood_test_01.py)

A Python script example generated using ParaView 6.0.0. It demonstrates a complete visualization pipeline for blood flow simulation data, including:
- Loading time-series VTP (red blood cells — `rbcbad` and `rbcgood`) and VTU (artery mesh and fluid particles) datasets over 100 timesteps
- Configuring surface representations with custom colors (red for healthy RBCs, cyan for abnormal RBCs, semi-transparent artery)
- Applying a Glyph filter with sphere glyphs to particle data, colored by velocity magnitude
- Setting up a 3D render view with camera, lighting, and colormap configurations

This script can be used to render a requested set of frames in batch.

---

### Presentation Slides

> [ALCF-DevSession_visualization_2026.pdf](./ALCF-DevSession_visualization_2026.pdf)

Slide deck from the ALCF Developer Session on ParaView visualization presented on April 22, 2026.

---

## Additional Resources

- [ParaView](https://www.paraview.org/) 6.0.0 or later
- [Dataset](https://web.cels.anl.gov/projects/alcf_vis_internal/MISC/BLOODFLOW_TUTORIAL_DATA.tar.gz) for blood flow tutorial


# persistent-light-flame-wind-wave-one-field-inference-engine

A lean AI architecture replacing transformer layers with **10 coupled nonlinear PDE "thought fields"**.

This repository contains a **Python neural-field engine** that evolves these fields via reaction-diffusion dynamics, weak low-rank tensor coupling, and a small spiking readout.

A minimal **desktop interface** is provided to send prompts to the engine and read attractor-state summaries as inference outputs.

## 📦 What’s inside

- `atlantean/atlantean_engine.py`: core engine CLI, lightweight inference loop, and training helpers
- `atlantean_fieldnet*.py`: fieldnet implementations (baseline, extended, integrators)
- `desktop_app.py`: simple Tkinter desktop UI that sends prompts and displays results
- `services/atlanteanEngine.js`: Node.js helper to spawn the Python engine (for Electron integrations)

## 🚀 Getting started

1) Install Python dependencies:

```bash
python -m pip install -r requirements.txt
```

2) Install Node.js dependencies (for Electron app):

```bash
cd electron && npm install
```

3) Run the Electron desktop app:

```bash
cd electron && npm start
```

This opens a native desktop window with a modern dark-themed GUI, featuring prompt input, real-time field visualization, and output display.

---

## 🧠 Atlantean IDE (Advanced GUI)

A more advanced IDE-style interface is included under `atlantean-ide/`.

To preview it in a browser (or wrap it in Electron), open:

```bash
open atlantean-ide/index.html
```

It provides draggable panels, a WebGL field view, and a console/editor layout.

4) (Alternative) Run the Python Tkinter UI:

```bash
python desktop_app.py
```

5) (Optional) Build a distributable app:

```bash
cd electron && npm run dist
```

This creates a bundled desktop app in `electron/dist/`.

## 🧠 Notes

- The engine is intentionally lightweight and does not use transformers. It evolves a set of coupled PDE-like fields and reads out simple scalar summaries.
- For experiments, see `atlantean/train_and_visualize.py` which trains the coupling using a spectral objective and saves sample visuals.

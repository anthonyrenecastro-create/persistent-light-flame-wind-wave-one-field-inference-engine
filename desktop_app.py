#!/usr/bin/env python3
"""Simple desktop UI for the Atlantean inference engine.

This UI sends prompts to the neural-field engine and displays a small summary of the
``attractor state`` (mean field values + spectral energy fraction) as the inference result.

Usage:
  python desktop_app.py

Dependencies: torch, numpy
"""

import threading
import time
import tkinter as tk
from tkinter import scrolledtext, messagebox

from atlantean import lightweight_infer, AtlanteanFieldNetAdvanced


class AtlanteanDesktop(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Atlantean Field Inference")
        self.geometry("640x520")

        self._build_ui()
        self._init_engine()

    def _build_ui(self):
        frm_top = tk.Frame(self)
        frm_top.pack(fill="x", padx=10, pady=8)

        tk.Label(frm_top, text="Prompt:").pack(side="left")
        self.prompt_var = tk.StringVar(value="hello")
        tk.Entry(frm_top, textvariable=self.prompt_var, width=50).pack(side="left", padx=(8, 0))
        tk.Button(frm_top, text="Run", command=self.on_run).pack(side="left", padx=8)
        tk.Button(frm_top, text="Reset", command=self.on_reset).pack(side="left")

        self.log = scrolledtext.ScrolledText(self, state="disabled", wrap="word", height=24)
        self.log.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    def _init_engine(self):
        self._log("Initializing neural-field engine (this may take a moment)...")
        self.engine = AtlanteanFieldNetAdvanced(H=48, W=48, nf=10, rank=2, spec_rank=3, use_spectral=True)
        self._log("Engine initialized. Enter a prompt and press Run.")

    def _log(self, msg: str):
        self.log.configure(state="normal")
        self.log.insert("end", f"{time.strftime('%H:%M:%S')} - {msg}\n")
        self.log.configure(state="disabled")
        self.log.see("end")

    def on_run(self):
        prompt = self.prompt_var.get().strip()
        if not prompt:
            messagebox.showwarning("Empty prompt", "Please type a prompt.")
            return
        self._set_ui_enabled(False)
        threading.Thread(target=self._run_inference, args=(prompt,), daemon=True).start()

    def on_reset(self):
        self._set_ui_enabled(False)
        threading.Thread(target=self._reset_engine, daemon=True).start()

    def _set_ui_enabled(self, enabled: bool):
        level = "normal" if enabled else "disabled"
        for widget in self.winfo_children():
            if isinstance(widget, tk.Frame):
                for child in widget.winfo_children():
                    child.configure(state=level)

    def _run_inference(self, prompt: str):
        try:
            start = time.time()
            res = lightweight_infer(self.engine, prompt, steps=120, dt=0.04, pde_steps=1, integrator="etdrk4")
            elapsed = time.time() - start
            self._log(f"Prompt: {prompt}")
            self._log(f"Result: {res}")
            self._log(f"Elapsed: {elapsed:.2f}s")
        except Exception as e:
            self._log(f"Error: {e}")
        finally:
            self._set_ui_enabled(True)

    def _reset_engine(self):
        try:
            self._log("Resetting engine state...")
            self.engine = AtlanteanFieldNetAdvanced(H=48, W=48, nf=10, rank=2, spec_rank=3, use_spectral=True)
            self._log("Engine reset.")
        except Exception as e:
            self._log(f"Error resetting engine: {e}")
        finally:
            self._set_ui_enabled(True)


if __name__ == "__main__":
    app = AtlanteanDesktop()
    app.mainloop()

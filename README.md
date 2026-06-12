---
title: Retinal Disease Detection
emoji: 🔬
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
---

# Retinal Disease Detection System

**Undergraduate Thesis — Department of Biomedical Engineering — 2025**

## What this does

This system detects 4 retinal diseases simultaneously from a single fundus photograph:
- **DR** — Diabetic Retinopathy
- **Glaucoma**
- **HR** — Hypertensive Retinopathy
- **RVO** — Retinal Vein Occlusion

It also demonstrates **AGPT (Attention-Guided Patch Transmission)** — a novel mechanism that uses the model's attention maps to select only disease-relevant image patches for transmission, achieving **70.3% bandwidth reduction** for rural telemedicine.

## Results

| Disease | AUC |
|---|---|
| DR | 0.9663 |
| Glaucoma | 0.9993 |
| HR | 0.9999 |
| RVO | 0.9462 |
| **Macro** | **0.9779** |

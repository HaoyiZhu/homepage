---
title: "SANA-WM: Efficient Minute-Scale World Modeling with Hybrid Linear Diffusion Transformer"
subtitle: "SANA-WM"
date: 2026-05-14T00:00:00Z
venue_short: "arXiv 2026"
publication_types: ["3"]
publication: "*arXiv preprint, 2026*"
publication_short: ""
summary: An efficient 2.6B open-source world model that generates minute-scale, 720p videos with precise camera control on a single GPU — up to 36× higher throughput than prior open-source baselines. Reposted by NVIDIA AI on X.
draft: false
featured: true
selected: true
authors:
  - admin
  - Haozhe Liu
  - Yuyang Zhao
  - Tian Ye
  - Junsong Chen
  - Jincheng Yu
  - Tong He
  - Song Han
  - Enze Xie
author_notes:
  - "Equal contribution"
  - "Equal contribution"
  - "Equal contribution"
  - "Equal contribution"
  - "Equal contribution"
  - ""
  - ""
  - ""
  - ""
tags:
  - World Model
  - Computer Vision
categories:
  - World Model
  - Computer Vision
external_link: https://nvlabs.github.io/Sana/WM/
links:
  - url: https://nvlabs.github.io/Sana/WM/
    name: Website
  - url: https://arxiv.org/abs/2605.15178
    name: arXiv
  - url: https://arxiv.org/pdf/2605.15178
    name: PDF
  - url: https://github.com/NVlabs/SANA
    name: GitHub
  - url: https://huggingface.co/papers/2605.15178
    name: HuggingFace
  - url: https://x.com/HaoyiZhu/status/2055132732339126764
    name: X/Twitter
  - url: http://xhslink.cn/o/7h2cBXxTdmx
    name: Xiaohongshu
image:
  filename: teaser.png
  focal_point: Smart
  preview_only: false
video: teaser_480p.mp4
---

**Abstract:**

We introduce SANA-WM, an efficient 2.6B-parameter open-source world model natively trained for one-minute generation, synthesizing high-fidelity, 720p, minute-scale videos with precise camera control. SANA-WM achieves visual quality comparable to large-scale industrial baselines such as LingBot-World and HY-WorldPlay, while significantly improving efficiency. Four core designs drive our architecture: (1) Hybrid Linear Attention combines frame-wise Gated DeltaNet (GDN) with softmax attention for memory-efficient long-context modeling. (2) Dual-Branch Camera Control ensures precise 6-DoF trajectory adherence. (3) Two-Stage Generation Pipeline applies a long-video refiner to stage-1 outputs, improving quality and consistency across sequences. (4) Robust Annotation Pipeline extracts accurate metric-scale 6-DoF camera poses from public videos to yield high-quality, spatiotemporally consistent action labels. Driven by these designs, SANA-WM demonstrates remarkable efficiency across data, training compute, and inference hardware: it uses only ~213K public video clips with metric-scale pose supervision, completes training in 15 days on 64 H100s, and generates each 60s clip on a single GPU; its distilled variant can be deployed on a single RTX 5090 with NVFP4 quantization to denoise a 60s 720p clip in 34s. On our one-minute world-model benchmark, SANA-WM demonstrates stronger action-following accuracy than prior open-source baselines and achieves comparable visual quality at 36× higher throughput for scalable world modeling.

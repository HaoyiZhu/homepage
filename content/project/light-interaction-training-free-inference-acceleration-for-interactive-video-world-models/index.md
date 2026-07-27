---
title: "Light Interaction: Training-Free Inference Acceleration for Interactive Video World Models"
subtitle: "*arXiv 2026*"
date: 2026-05-29T00:00:00Z
venue_short: "arXiv 2026"
publication_types: ["1"]
publication: "*arXiv preprint, 2026*"
publication_short: ""
draft: false
featured: false
authors:
  - Jiacheng Lu
  - admin
  - Sipei Yi
  - Enze Xie
  - Yu Li
  - Cheng Zhuo
tags:
  - Computer Vision
categories:
  - Computer Vision
external_link: https://arxiv.org/abs/2605.31158
links:
  - url: https://arxiv.org/abs/2605.31158
    name: arXiv
  - url: https://arxiv.org/pdf/2605.31158
    name: PDF
image:
  filename: teaser.png
  focal_point: Smart
  preview_only: false
---

**Abstract:**

Interactive video world models generate video chunk by chunk in response to user-controlled camera movements, enabling applications such as real-time game simulation, virtual scene navigation, and embodied AI training. However, scaling to long interactive trajectories is prohibitively expensive due to growing context memory, quadratic attention complexity, and repeated denoising steps. We present Light Interaction, a training-free inference acceleration framework for interactive video world models. Our key insight is that interaction naturally enables trajectory-dependent adaptive computation: retrieved spatial memory can be discarded during novel exploration, temporal context can be adjusted according to local latent dynamics, and early-step model outputs can be reused when the camera revisits familiar regions. Based on this insight, Light Interaction combines adaptive context management, denoising cache acceleration, and hardware-software co-designed 3D block sparse attention with fused Triton kernels. Evaluated on HY-WorldPlay and Matrix-Game-3.0, Light Interaction achieves up to 2.59x speedup without model retraining while maintaining competitive visual quality.

---
title: "Learning Primitive Embodied World Models: Towards Scalable Robotic Learning"
subtitle: "*arXiv 2025*"
date: 2025-08-28T00:00:00Z
venue_short: "arXiv 2025"
publication_types: ["1"]
publication: "*arXiv preprint, 2025*"
publication_short: ""
draft: false
featured: false
authors:
  - Qiao Sun
  - Liujia Yang
  - Wei Tang
  - Wei Huang
  - Kaixin Xu
  - Yongchao Chen
  - Mingyu Liu
  - Jiange Yang
  - admin
  - Yating Wang
  - Tong He
  - Yilun Chen
  - Xili Dai
  - Nanyang Ye
  - Qinying Gu
tags:
  - Computer Vision
categories:
  - Computer Vision
external_link: https://arxiv.org/abs/2508.20840
links:
  - url: https://arxiv.org/abs/2508.20840
    name: arXiv
  - url: https://arxiv.org/pdf/2508.20840
    name: PDF
image:
  filename: teaser.png
  focal_point: Smart
  preview_only: false
---

**Abstract:**

While video-generation-based embodied world models have gained increasing attention, their reliance on large-scale embodied interaction data remains a key bottleneck. The scarcity, difficulty of collection, and high dimensionality of embodied data fundamentally limit the alignment granularity between language and actions and exacerbate the challenge of long-horizon video generation--hindering generative models from achieving a"GPT moment"in the embodied domain. There is a naive observation: the diversity of embodied data far exceeds the relatively small space of possible primitive motions. Based on this insight, we propose a novel paradigm for world modeling--Primitive Embodied World Models (PEWM). By restricting video generation to fixed short horizons, our approach 1) enables fine-grained alignment between linguistic concepts and visual representations of robotic actions, 2) reduces learning complexity, 3) improves data efficiency in embodied data collection, and 4) decreases inference latency. By equipping with a modular Vision-Language Model (VLM) planner and a Start-Goal heatmap Guidance mechanism (SGG), PEWM further enables flexible closed-loop control and supports compositional generalization of primitive-level policies over extended, complex tasks. Our framework leverages the spatiotemporal vision priors in video models and the semantic awareness of VLMs to bridge the gap between fine-grained physical interaction and high-level reasoning, paving the way toward scalable, interpretable, and general-purpose embodied intelligence.

---
title: "GAE: Unleashing Physical Potential of VLM with Generalizable Action Expert"
subtitle: "*ICML 2026*"
date: 2025-10-04T00:00:00Z
venue_short: "ICML 2026"
publication_types: ["1"]
publication: "*The Forty-third International Conference on Machine Learning (ICML), 2026*"
publication_short: ""
draft: false
featured: false
authors:
  - Mingyu Liu
  - Zheng Huang
  - Xiaoyi Lin
  - Muzhi Zhu
  - Canyu Zhao
  - Yating Wang
  - admin
  - Hao Chen
  - Chunhua Shen
tags:
  - Computer Vision
categories:
  - Computer Vision
external_link: https://arxiv.org/abs/2510.03896
links:
  - url: https://arxiv.org/abs/2510.03896
    name: arXiv
  - url: https://arxiv.org/pdf/2510.03896
    name: PDF
image:
  filename: teaser.png
  focal_point: Smart
  preview_only: false
---

**Abstract:**

Vision-language models demonstrate strong reasoning and planning abilities, yet grounding these predictions into precise robot actions remains a central challenge. Existing Vision-Language-Action methods typically entangle reasoning and action generation, leading to limited generalization. We propose Generalizable Action Expert (GAE), a task-agnostic model that converts sparse geometric plans into dense robot actions. Our approach introduces a sparse geometric interface: the VLM predicts sparse 3D waypoints representing high-level intention, while GAE maps these waypoints together with real-time point cloud observations to continuous action trajectories. GAE is pretrained on a large-scale pointcloud-trajectory dataset comprising 150k trajectories from both simulation and real-world robots. To further improve efficiency and generalization, we introduce an Action Pre-training, Pointcloud Fine-tuning (APPF) scheme that decouples learning action dynamics from geometry grounding. After pretraining, GAE is frozen and reused across downstream tasks, requiring only lightweight fine-tuning of the VLM to produce the sparse interface. Experiments show that our method achieves strong performance and generalization across diverse visual domains, camera viewpoints, and natural language instructions.

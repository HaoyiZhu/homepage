---
title: "Tra-MoE: Learning Trajectory Prediction Model from Multiple Domains for Adaptive Policy Conditioning"
subtitle: "*CVPR 2025*"
date: 2025-02-28T00:00:00Z

# Publication type.
# Legend: 0 = Uncategorized; 1 = Conference paper; 2 = Journal article;
# 3 = Preprint / Working Paper; 4 = Report; 5 = Book; 6 = Book section;
# 7 = Thesis; 8 = Patent
publication_types: ["1"]

# Publication name and optional abbreviated publication name.
publication: "*Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.*"
publication_short: ""

draft: false
featured: false
authors:
  - Jiange Yang
  - admin
  - Yating Wang
  - Gangshan Wu
  - Tong He
  - Limin Wang
tags:
  - Embodied AI
  - Robot Learning
categories:
  - Embodied AI
  - Robot Learning
external_link: https://arxiv.org/abs/2411.14519
links:
  - url: https://arxiv.org/pdf/2411.14519
    name: PDF
  - url: https://arxiv.org/abs/2411.14519
    name: Arxiv
image:
  filename: pipeline.png
  focal_point: Smart
  preview_only: false
---

**Abstract:**

Learning from multiple domains is a primary factor that influences the generalization of a single unified robot system. In this paper, we aim to learn the trajectory prediction model by using broad out-of-domain data to improve its performance and generalization ability. Trajectory model is designed to predict any-point trajectories in the current frame given an instruction and can provide detailed control guidance for robotic policy learning. To handle the diverse out-of-domain data distribution, we propose a sparsely-gated MoE (Top-1 gating strategy) architecture for trajectory model, coined as Tra-MoE. The sparse activation design enables good balance between parameter cooperation and specialization, effectively benefiting from large-scale out-of-domain data while maintaining constant FLOPs per token. In addition, we further introduce an adaptive policy conditioning technique by learning 2D mask representations for predicted
trajectories, which is explicitly aligned with image observations to guide action prediction more flexibly. We perform extensive experiments on both simulation and real-world scenarios to verify the effectiveness of Tra-MoE and adaptive policy conditioning technique. We also conduct a comprehensive empirical study to train Tra-MoE, demonstrating that our Tra-MoE consistently exhibits superior performance compared to the dense baseline model, even when the latter is scaled to match Tra-MoE’s parameter count.
---
title: "Point Cloud Matters: Rethinking the Impact of Different Observation Spaces on Robot Learning"
subtitle: "Point Cloud Matters"
summary: Extensive experiments prove that point cloud observations are beneficial for robot learning.
date: 2024-09-26T00:00:00Z

# Publication type.
# Legend: 0 = Uncategorized; 1 = Conference paper; 2 = Journal article;
# 3 = Preprint / Working Paper; 4 = Report; 5 = Book; 6 = Book section;
# 7 = Thesis; 8 = Patent
publication_types: ["3"]

# Publication name and optional abbreviated publication name.
publication: "*Neural Information Processing Systems (NeurIPS) Dataset & Benchmark, 2024*"
publication_short: ""

draft: false
featured: false
authors:
  - admin
  - Yating Wang
  - Di Huang
  - Weicai Ye
  - Wanli Ouyang
  - Tong He
author_notes:
  - ""
  - ""
  - ""
  - ""
  - ""
  - ""
tags:
  - Embodied AI
  - Robot Learning
categories:
  - Embodied AI
  - Robot Learning
external_link: https://haoyizhu.github.io/PointCloudMatters.github.io/
links:
  - url: https://haoyizhu.github.io/PointCloudMatters.github.io/
    name: Project
  - url: https://github.com/HaoyiZhu/PointCloudMatters
    name: Github
  - url: https://arxiv.org/pdf/2402.02500
    name: PDF
  - url: https://arxiv.org/abs/2402.02500
    name: Arxiv
image:
  filename: overview.png
  focal_point: Smart
  preview_only: false
---

**Abstract:**

In robot learning, the observation space is crucial due to the distinct characteristics of different modalities, which can potentially become a bottleneck alongside policy design. In this study, we explore the influence of various observation spaces on robot learning, focusing on three predominant modalities: RGB, RGB-D, and point cloud. We introduce OBSBench, a benchmark comprising two simulators and 125 tasks, along with standardized pipelines for various encoders and policy baselines. Extensive experiments on diverse contact-rich manipulation tasks reveal a notable trend: point cloud-based methods, even those with the simplest designs, frequently outperform their RGB and RGB-D counterparts. This trend persists in both scenarios: training from scratch and utilizing pre-training. Furthermore, our findings demonstrate that point cloud observations often yield better policy performance and significantly stronger generalization capabilities across various geometric and visual conditions. These outcomes suggest that the 3D point cloud is a valuable observation modality for intricate robotic tasks. We also suggest that incorporating both appearance and coordinate information can enhance the performance of point cloud methods. We hope our work provides valuable insights and guidance for designing more generalizable and robust robotic models.

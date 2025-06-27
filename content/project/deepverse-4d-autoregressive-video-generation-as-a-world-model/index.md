---
title: "DeepVerse: 4D Autoregressive Video Generation as a World Model"
subtitle: "*arXiv preprint, 2025*"
date: 2025-06-01T00:00:00Z

# Publication type.
# Legend: 0 = Uncategorized; 1 = Conference paper; 2 = Journal article;
# 3 = Preprint / Working Paper; 4 = Report; 5 = Book; 6 = Book section;
# 7 = Thesis; 8 = Patent
publication_types: ["1"]

# Publication name and optional abbreviated publication name.
publication: "*arXiv preprint, 2025*"
publication_short: ""

draft: false
featured: false
authors:
  - Junyi Chen
  - admin
  - Xianglong He
  - Yifan Wang
  - Jianjun Zhou
  - Wenzheng Chang
  - Yang Zhou
  - Zizun Li
  - Zhoujie Fu
  - Jiangmiao Pang
  - Tong He
tags:
  - Embodied AI
  - Computer Vision
categories:
  - Embodied AI
  - Robot Learning
external_link: https://sotamak1r.github.io/deepverse/
links:
  - url: https://sotamak1r.github.io/deepverse/
    name: Website
  - url: https://github.com/SOTAMak1r/DeepVerse
    name: GitHub
  - url: https://arxiv.org/pdf/2506.01103
    name: PDF
  - url: https://arxiv.org/abs/2506.01103
    name: arXiv
image:
  filename: teaser.png
  focal_point: Smart
  preview_only: false
---

**Abstract:**
World models serve as essential building blocks toward Artificial General Intelligence (AGI), enabling intelligent agents to predict future states and plan actions by simulating complex physical interactions. However, existing interactive models primarily predict visual observations, thereby neglecting crucial hidden states like geometric structures and spatial coherence. This leads to rapid error accumulation and temporal inconsistency. To address these limitations, we introduce DeepVerse, a novel 4D interactive world model explicitly incorporating geometric predictions from previous timesteps into current predictions conditioned on actions. Experiments demonstrate that by incorporating explicit geometric constraints, DeepVerse captures richer spatio-temporal relationships and underlying physical dynamics. This capability significantly reduces drift and enhances temporal consistency, enabling the model to reliably generate extended future sequences and achieve substantial improvements in prediction accuracy, visual realism, and scene rationality. Furthermore, our method provides an effective solution for geometry-aware memory retrieval, effectively preserving long-term spatial consistency. We validate the effectiveness of DeepVerse across diverse scenarios, establishing its capacity for high-fidelity, long-horizon predictions grounded in geometry-aware dynamics.
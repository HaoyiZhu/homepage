---
title: "PonderV2: Pave the Way for 3D Foundation Model with A Universal Pre-training Paradigm"
subtitle: "PonderV2"
date: 2023-10-13T00:00:00Z

# Publication type.
# Legend: 0 = Uncategorized; 1 = Conference paper; 2 = Journal article;
# 3 = Preprint / Working Paper; 4 = Report; 5 = Book; 6 = Book section;
# 7 = Thesis; 8 = Patent
publication_types: ["3"]

# Publication name and optional abbreviated publication name.
publication: "*arXiv preprint arXiv:2310.08586, 2023.*"
publication_short: "arXiv preprint"

draft: false
featured: true
authors:
  - admin
  - Honghui Yang
  - Xiaoyang Wu
  - Di Huang
  - Sha Zhang
  - Xianglong He
  - Tong He
  - Hengshuang Zhao
  - Chunhua Shen
  - Yu Qiao
  - Wanli Ouyang
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
  - ""
  - ""
tags:
  - Computer Vision
categories:
  - Computer Vision
external_link: https://github.com/OpenGVLab/PonderV2
links:
  - url: https://github.com/OpenGVLab/PonderV2
    name: Github
  - url: https://arxiv.org/pdf/2310.08586.pdf
    name: PDF
  - url: https://arxiv.org/abs/2310.08586
    name: Arxiv
image:
  filename: ponderv2.png
  focal_point: Smart
  preview_only: false
---

**Abstract:**

In contrast to numerous NLP and 2D computer vision foundational models, the learning of a robust and highly generalized 3D foundational model poses considerably greater challenges. This is primarily due to the inherent data variability and the diversity of downstream tasks. In this paper, we introduce a comprehensive 3D pre-training framework designed to facilitate the acquisition of efficient 3D representations, thereby establishing a pathway to 3D foundational models. Motivated by the fact that informative 3D features should be able to encode rich geometry and appearance cues that can be utilized to render realistic images, we propose a novel universal paradigm to learn point cloud representations by differentiable neural rendering, serving as a bridge between 3D and 2D worlds. We train a point cloud encoder within a devised volumetric neural renderer by comparing the rendered images with the real images. Notably, our approach demonstrates the seamless integration of the learned 3D encoder into diverse downstream tasks. These tasks encompass not only high-level challenges such as 3D detection and segmentation but also low-level objectives like 3D reconstruction and image synthesis, spanning both indoor and outdoor scenarios. Besides, we also illustrate the capability of pre-training a 2D backbone using the proposed universal methodology, surpassing conventional pre-training methods by a large margin.  For the first time, PonderV2 achieves state-of-the-art performance on 11 indoor and outdoor benchmarks. The consistent improvements in various settings imply the effectiveness of the proposed method. Code and models will be made available at [https://github.com/OpenGVLab/PonderV2](https://github.com/OpenGVLab/PonderV2).


<p align="center">
    <img src="assets/pipeline.png" alt="pipeline" />
</p>
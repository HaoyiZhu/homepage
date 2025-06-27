---
draft: false
publication_types:
  - "1"
summary: A novel representation learning framework that emphasizes the importance of 3D spatial awareness in embodied AI.
authors:
  - admin
  - Honghui Yang
  - Yating Wang
  - Jiange Yang
  - Limin Wang
  - Tong He
publication_short: ""
publication: "*The Thirteenth International Conference on Learning Representations (ICLR), 2025*"
author_notes:
  - ""
  - ""
  - ""
  - ""
  - ""
  - ""
title: "SPA: 3D Spatial-Awareness Enables Effective Embodied Representation"
subtitle: "SPA"
date: 2024-10-11T00:00:00.000Z
featured: true
tags:
  - Embodied AI
  - Computer Vision
categories:
  - Embodied AI
  - Computer Vision
external_link: https://haoyizhu.github.io/spa/
links:
  - url: https://haoyizhu.github.io/spa/
    name: Project
  - url: https://arxiv.org/pdf/2410.08208
    name: PDF
  - url: https://arxiv.org/abs/2410.08208
    name: Arxiv
  - name: Twitter
    url: https://x.com/HaoyiZhu/status/1844675411760013471
  - name: Code
    url: https://github.com/HaoyiZhu/SPA
  - url: https://huggingface.co/HaoyiZhu/SPA
    name: HuggingFace Model
  - url: https://github.com/HaoyiZhu/RealRobot
    name: RealWorld Code
  - url: https://www.youtube.com/watch?v=LS2R-kBxxwY
    name: YouTube Video
image:
  filename: image.gif
  focal_point: Smart
  preview_only: false
video: spa_pretrain.mp4
  
---
**Abstract:**

In this paper, we introduce SPA, a novel representation learning framework that emphasizes the importance of 3D spatial awareness in embodied AI. Our approach leverages differentiable neural rendering on multi-view images to endow a vanilla Vision Transformer (ViT) with intrinsic spatial understanding. We present the most comprehensive evaluation of embodied representation learning to date, covering 268 tasks across 8 simulators with diverse policies in both single-task and language-conditioned multi-task scenarios. The results are compelling: SPA consistently outperforms more than 10 state-of-the-art representation methods, including those specifically designed for embodied AI, vision-centric tasks, and multi-modal applications, while using less training data. Furthermore, we conduct a series of real-world experiments to confirm its effectiveness in practical scenarios. These results highlight the critical role of 3D spatial awareness for embodied representation learning. Our strongest model takes more than 6000 GPU hours to train and we are committed to open-sourcing all code and model weights to foster future research in embodied representation learning.

<video controls autoplay loop muted>
  <source src="spa_pretrain.mp4" type="video/mp4">
</video>
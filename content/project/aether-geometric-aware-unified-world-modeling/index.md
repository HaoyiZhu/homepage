---
draft: false
publication_types:
  - "1"
summary: A geometric-aware unified world model, capable of 4D reconstruction, action-conditioned prediction, and visual planning.
authors:
  - admin
  - Yifan Wang
  - Jianjun Zhou
  - Wenzheng Chang
  - Yang Zhou
  - Zizun Li
  - Junyi Chen
  - Chunhua Shen
  - Jiangmiao Pang
  - Tong He
publication_short: ""
publication: "*International Conference on Computer Vision, (ICCV) 2025*"
author_notes:
  - ""
  - ""
  - ""
  - ""
  - ""
  - ""
  - ""
  - ""
  - ""
  - ""
title: "Aether: Geometric-Aware Unified World Modeling"
subtitle: "Aether"
date: 2025-03-25T00:00:00.000Z
featured: true
tags:
  - Embodied AI
  - Computer Vision
categories:
  - Embodied AI
  - Computer Vision
external_link: https://aether-world.github.io/
links:
  - url: https://aether-world.github.io/
    name: Project
  - url: https://aether-world.github.io/assets/aether.pdf
    name: PDF
  - url: https://arxiv.org/abs/2503.18945
    name: arXiv
  - name: Code
    url: https://github.com/OpenRobotLab/Aether
image:
  filename: teaser.png
  focal_point: Smart
  preview_only: false
video: teaser_480p.mp4
  
---
**Abstract:**

The integration of geometric reconstruction and generative modeling remains a critical challenge in developing AI systems capable of human-like spatial reasoning. This paper proposes Aether, a unified framework that enables geometry-aware reasoning in world models by jointly optimizing three core capabilities: (1) 4D dynamic reconstruction, (2) action-conditioned video prediction, and (3) goal-conditioned visual planning. Through task-interleaved feature learning, Aether achieves synergistic knowledge sharing across reconstruction, prediction, and planning objectives. Building upon video generation models, our framework demonstrates unprecedented synthetic-to-real generalization despite never observing real-world data during training. Furthermore, our approach achieves zero-shot generalization in both action following and reconstruction tasks, thanks to its intrinsic geometric modeling. Remarkably, even without real-world data, its reconstruction performance far exceeds that of domain-specific models. Additionally, Aether leverages a geometry-informed action space to seamlessly translate predictions into actions, enabling effective autonomous trajectory planning. We hope our work inspires the community to explore new frontiers in physically-reasonable world modeling and its applications.

<video controls autoplay loop muted>
  <source src="teaser_480p.mp4" type="video/mp4">
</video>
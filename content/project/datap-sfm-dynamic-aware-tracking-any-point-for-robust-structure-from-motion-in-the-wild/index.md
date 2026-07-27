---
title: "DATAP-SfM: Dynamic-Aware Tracking Any Point for Robust Structure from Motion in the Wild"
subtitle: "*arXiv 2024*"
date: 2024-11-20T00:00:00Z
venue_short: "arXiv 2024"
publication_types: ["1"]
publication: "*arXiv preprint, 2024*"
publication_short: ""
draft: false
featured: false
authors:
  - Weicai Ye
  - Xinyu Chen
  - Ruohao Zhan
  - Di Huang
  - Xiaoshui Huang
  - admin
  - Hujun Bao
  - Wanli Ouyang
  - Tong He
  - Guofeng Zhang
tags:
  - Computer Vision
categories:
  - Computer Vision
external_link: https://arxiv.org/abs/2411.13291
links:
  - url: https://arxiv.org/abs/2411.13291
    name: arXiv
  - url: https://arxiv.org/pdf/2411.13291
    name: PDF
image:
  filename: teaser.png
  focal_point: Smart
  preview_only: false
---

**Abstract:**

This paper proposes a concise, elegant, and robust pipeline to estimate smooth camera trajectories and obtain dense point clouds for casual videos in the wild. Traditional frameworks, such as ParticleSfM~\cite{zhao2022particlesfm}, address this problem by sequentially computing the optical flow between adjacent frames to obtain point trajectories. They then remove dynamic trajectories through motion segmentation and perform global bundle adjustment. However, the process of estimating optical flow between two adjacent frames and chaining the matches can introduce cumulative errors. Additionally, motion segmentation combined with single-view depth estimation often faces challenges related to scale ambiguity. To tackle these challenges, we propose a dynamic-aware tracking any point (DATAP) method that leverages consistent video depth and point tracking. Specifically, our DATAP addresses these issues by estimating dense point tracking across the video sequence and predicting the visibility and dynamics of each point. By incorporating the consistent video depth prior, the performance of motion segmentation is enhanced. With the integration of DATAP, it becomes possible to estimate and optimize all camera poses simultaneously by performing global bundle adjustments for point tracking classified as static and visible, rather than relying on incremental camera registration. Extensive experiments on dynamic sequences, e.g., Sintel and TUM RGBD dynamic sequences, and on the wild video, e.g., DAVIS, demonstrate that the proposed method achieves state-of-the-art performance in terms of camera pose estimation even in complex dynamic challenge scenes.

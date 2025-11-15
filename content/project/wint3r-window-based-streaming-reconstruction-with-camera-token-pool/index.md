---
title: "WinT3R: Window-Based Streaming Reconstruction With Camera Token Pool"
subtitle: "*arXiv 2025*"
date: 2025-09-05T00:00:00Z

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
  - Zizun Li
  - Jianjun Zhou
  - Yifan Wang
  - Haoyi Guo
  - Wenzheng Chang
  - Yang Zhou
  - admin
  - Junyi Chen
  - Chunhua Shen
  - Tong He
tags:
  - Computer Vision
categories:
  - Computer Vision
external_link: https://lizizun.github.io/WinT3R.github.io/
links:
  - url: https://lizizun.github.io/WinT3R.github.io/
    name: Website
  - url: https://arxiv.org/abs/2509.05296
    name: arXiv
  - url: https://github.com/LiZizun/WinT3R
    name: GitHub
image:
  filename: teaser.jpg
  focal_point: Smart
  preview_only: false
---

**Abstract:**

We present WinT3R, a feed-forward reconstruction model capable of online prediction of precise camera poses and high-quality point maps. Previous methods suffer from a trade-off between reconstruction quality and real-time performance. To address this, we first introduce a sliding window mechanism that ensures sufficient information exchange among frames within the window, thereby improving the quality of geometric predictions without large computation. In addition, we leverage a compact representation of cameras and maintain a global camera token pool, which enhances the reliability of camera pose estimation without sacrificing efficiency. These designs enable WinT3R to achieve state-of-the-art performance in terms of online reconstruction quality, camera pose estimation, and reconstruction speed, as validated by extensive experiments on diverse datasets.
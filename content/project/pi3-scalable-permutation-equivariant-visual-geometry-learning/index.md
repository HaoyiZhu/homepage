---
title: "$\pi^3$: Scalable Permutation-Equivariant Visual Geometry Learning"
subtitle: "*arXiv preprint, 2025*"
date: 2025-07-22T00:00:00Z

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
  - Yifan Wang
  - Jianjun Zhou
  - admin
  - Wenzheng Chang
  - Yang Zhou
  - Zizun Li
  - Junyi Chen
  - Jiangmiao Pang
  - Chunhua Shen
  - Tong He
tags:
  - Computer Vision
categories:
  - Computer Vision
external_link: https://yyfz.github.io/pi3/
links:
  - url: https://yyfz.github.io/pi3/
    name: Website
  - url: https://github.com/yyfz/Pi3
    name: GitHub
  - url: https://arxiv.org/pdf/2507.13347
    name: PDF
  - url: https://arxiv.org/abs/2507.13347
    name: arXiv
image:
  filename: teaser.png
  focal_point: Smart
  preview_only: false
---

**Abstract:**
We introduce $\pi^3$, a feed-forward neural network that offers a
novel approach to visual geometry reconstruction, breaking
the reliance on a conventional fixed reference view. Previous
methods often anchor their reconstructions to a designated
viewpoint, an inductive bias that can lead to instability and
failures if the reference is suboptimal. In contrast, $\pi^3$ employs a fully permutation-equivariant architecture to predict
affine-invariant camera poses and scale-invariant local point
maps without any reference frames. This design makes our
model inherently robust to input ordering and highly scalable.
These advantages enable our simple and bias-free approach
to achieve state-of-the-art performance on a wide range of
tasks, including camera pose estimation, monocular/video
depth estimation, and dense point map reconstruction. Code
and models are publicly available.
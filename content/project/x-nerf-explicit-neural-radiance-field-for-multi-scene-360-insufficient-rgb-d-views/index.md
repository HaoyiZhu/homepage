---
title: "X-NeRF: Explicit Neural Radiance Field for Multi-Scene 360 Insufficient
  RGB-D Views"
subtitle: "*WACV 2023*"
date: 2022-10-11T00:00:00Z

# Publication type.
# Legend: 0 = Uncategorized; 1 = Conference paper; 2 = Journal article;
# 3 = Preprint / Working Paper; 4 = Report; 5 = Book; 6 = Book section;
# 7 = Thesis; 8 = Patent
publication_types: ["1"]

# Publication name and optional abbreviated publication name.
publication: "*IEEE Winter Conference on Applications of Computer Vision (WACV), 2023*"
publication_short: ""

draft: false
featured: false
authors:
  - admin
  - Hao-Shu Fang
  - Cewu Lu
tags:
  - Computer Vision
categories:
  - Computer Vision
external_link: https://arxiv.org/abs/2210.05135
links:
  - url: https://arxiv.org/pdf/2210.05135.pdf
    name: PDF
  - url: https://arxiv.org/abs/2210.05135
    name: Arxiv
  - url: https://github.com/HaoyiZhu/XNeRF
    name: Code
image:
  filename: insufficient.jpg
  focal_point: Smart
  preview_only: false
---

**Abstract:**

Neural Radiance Fields (NeRFs), despite their outstanding performance on novel view synthesis, often need dense input views. Many papers train one model for each scene respectively and few of them explore incorporating multimodal data into this problem. In this paper, we focus on a rarely discussed but important setting: can we train one model that can represent multiple scenes, with 360◦ insufficient views and RGB-D images? We refer insufficient views to few extremely sparse and almost non-overlapping views. To deal with it, X-NeRF, a fully explicit approach which learns a general scene completion process instead of a coordinate-based mapping, is proposed. Given a few insufficient RGB-D input views, X-NeRF first transforms them to a sparse point cloud tensor and then applies a 3D sparse generative Convolutional Neural Network (CNN) to complete it to an explicit radiance field whose volumetric rendering can be conducted fast without running networks during inference. To avoid overfitting, besides common rendering loss, we apply perceptual loss as well as view augmentation through random rotation on point clouds. The proposed methodology significantly out-performs previous implicit methods in our setting, indicating the great potential of proposed problem and approach. Codes and data are available at [https://github.com/HaoyiZhu/XNeRF](https://github.com/HaoyiZhu/XNeRF).
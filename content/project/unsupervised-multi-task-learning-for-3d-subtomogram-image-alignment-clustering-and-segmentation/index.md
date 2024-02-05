---
title: Unsupervised Multi-Task Learning for 3D Subtomogram Image Alignment,
  Clustering and Segmentation
date: 2022-01-08T00:00:00Z

# Publication type.
# Legend: 0 = Uncategorized; 1 = Conference paper; 2 = Journal article;
# 3 = Preprint / Working Paper; 4 = Report; 5 = Book; 6 = Book section;
# 7 = Thesis; 8 = Patent
publication_types: ["1"]

# Publication name and optional abbreviated publication name.
publication: "*IEEE International Conference on Image Processing (ICIP), 2022*"
publication_short: ""

draft: false
featured: false
authors:
  - admin
  - Chuting Wang
  - Yuanxin Wang
  - Zhaoxin Fan
  - Mostofa Rafid Uddin
  - Xin Gao
  - Jing Zhang
  - Xiangrui Zeng
  - Min Xu
tags:
  - Computer Vision
  - Others
categories:
  - Computer Vision
  - Others
external_link: https://ieeexplore.ieee.org/abstract/document/9897919/
links:
  - url: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9897919&tag=1
    name: PDF
  - url: https://doi.org/10.1109/ICIP46576.2022.9897919
    name: DOI
  - url: https://ieeexplore.ieee.org/abstract/document/9897919
    name: ICIP 2022
image:
  filename: pipeline_00.jpg
  focal_point: Smart
  preview_only: false
---

**Abstract：** 

3D subtomogram image alignment, clustering, and segmentation are vital to macromolecular structure recognition in cryo-electron tomography (cryo-ET). However, acquiring ground-truth labels to train a unified deep learning model that can simultaneously deal with these tasks is unaffordable. To this end, we propose an end-to-end unified multi-task learning framework to simultaneously complete the three tasks, where models are trained in an unsupervised manner without using any labels. In particular, we have three parallel branches. In the alignment branch, we adopt a two-stage training scheme, i.e., self-supervised pretraining and constrained unsupervised training using our proposed skip correlation attention layer and constrained loss. Synchronously, in the clustering branch, the learned deep cluster features are utilized to iteratively cluster subtomograms into groups using pseudo-labels from an image-wise Gaussian Mixture Model (GMM). Meanwhile, in the segmentation branch, we use rough pseudo-labels generated from a voxel-wise GMM as supervision signals, and prior knowledge from humans is utilized to jointly learn how to correct these labels as well as predict reliable segmentation results. Benefiting from the end-to-end unified network architecture, our method achieves overall state-of-the-art performance on both simulated and real subtomogram processing benchmarks.
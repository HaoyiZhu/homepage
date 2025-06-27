---
title: "UniPAD: A Universal Pre-training Paradigm for Autonomous Driving"
subtitle: "UniPAD"
date: 2023-10-12T00:00:00Z

# Publication type.
# Legend: 0 = Uncategorized; 1 = Conference paper; 2 = Journal article;
# 3 = Preprint / Working Paper; 4 = Report; 5 = Book; 6 = Book section;
# 7 = Thesis; 8 = Patent
publication_types: ["3"]

# Publication name and optional abbreviated publication name.
publication: "*Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024*"
publication_short: ""

draft: false
featured: false
authors:
  - Honghui Yang
  - Sha Zhang
  - Di Huang
  - Xiaoyang Wu
  - admin
  - Tong He
  - Shixiang Tang
  - Hengshuang Zhao
  - Qibo Qiu
  - Binbin Lin
  - Xiaofei He
  - Wanli Ouyang
tags:
  - Computer Vision
categories:
  - Computer Vision
external_link: https://github.com/Nightmare-n/UniPAD
links:
  - url: https://github.com/Nightmare-n/UniPAD
    name: Github
  - url: https://arxiv.org/pdf/2310.08370.pdf
    name: PDF
  - url: https://arxiv.org/abs/2310.08370
    name: Arxiv
image:
  filename: unipad.png
  focal_point: Smart
  preview_only: false
---

**Abstract:**

In the context of autonomous driving, the significance of effective feature learning is widely acknowledged. While conventional 3D self-supervised pre-training methods have shown widespread success, most methods follow the ideas originally designed for 2D images. In this paper, we present UniPAD, a novel self-supervised learning paradigm applying 3D volumetric differentiable rendering. UniPAD implicitly encodes 3D space, facilitating the reconstruction of continuous 3D shape structures and the intricate appearance characteristics of their 2D projections. The flexibility of our method enables seamless integration into both 2D and 3D frameworks, enabling a more holistic comprehension of the scenes. We manifest the feasibility and effectiveness of UniPAD by conducting extensive experiments on various downstream 3D tasks. Our method significantly improves lidar-, camera-, and lidar-camera-based baseline by 9.1, 7.7, and 6.9 NDS, respectively. Notably, our pre-training pipeline achieves 73.2 NDS for 3D object detection and 79.4 mIoU for 3D semantic segmentation on the nuScenes validation set, achieving state-of-the-art results in comparison with previous methods. The code will be available at [https://github.com/Nightmare-n/UniPAD](https://github.com/Nightmare-n/UniPAD).
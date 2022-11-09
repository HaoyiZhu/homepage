---
title: "AlphaPose: Whole-Body Regional Multi-Person Pose Estimation and Tracking
  in Real-Time"

# Publication type.
# Legend: 0 = Uncategorized; 1 = Conference paper; 2 = Journal article;
# 3 = Preprint / Working Paper; 4 = Report; 5 = Book; 6 = Book section;
# 7 = Thesis; 8 = Patent
publication_types: ["2"]

# Publication name and optional abbreviated publication name.
publication: "*IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)*"
publication_short: ""

subtitle: "AlphaPose"
date: 2022-11-07T00:00:00Z
draft: false
featured: true
summary: 'An accurate real-time multi-person pose estimator and tracker.'
authors:
  - Hao-Shu Fang
  - Jiefeng Li
  - Hongyang Tang
  - Chao Xu
  - admin
  - Yuliang Xiu
  - Yong-Lu Li
  - Cewu Lu
author_notes:
  - '*Equal contribution'
  - '*Equal contribution'
  - ''
  - '$Equal contribution'
  - '$Equal contribution'
  - ''
  - ''
  - ''
tags:
  - Computer Vision
categories:
  - Computer Vision
external_link: https://github.com/MVIG-SJTU/AlphaPose
links:
  - url: http://mvig.org/research/alphapose.html
    name: Project
  - name: PDF
    url: https://arxiv.org/pdf/2211.03375.pdf
  - url: https://arxiv.org/abs/2211.03375
    name: Arxiv
  - name: Github
    url: https://github.com/MVIG-SJTU/AlphaPose
  - name: Model Zoo
    url: https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/MODEL_ZOO.md
  - name: Halpe Dataset
    url: https://github.com/Fang-Haoshu/Halpe-FullBody
image:
  filename: logo.jpg
  focal_point: Smart
  preview_only: false
video: alphapose_17.mp4
---

**Abstract：**

Accurate whole-body multi-person pose estimation and tracking is an important yet challenging topic in computer vision. To capture the subtle actions of humans for complex behavior analysis, whole-body pose estimation including the face, body, hand and foot is essential over conventional body-only pose estimation. In this paper, we present AlphaPose, a system that can perform accurate whole-body pose estimation and tracking jointly while running in realtime. To this end, we propose several new techniques: Symmetric Integral Keypoint Regression (SIKR) for fast and fine localization, Parametric Pose Non-Maximum-Suppression (P-NMS) for eliminating redundant human detections and Pose Aware Identity Embedding for jointly pose estimation and tracking. During training, we resort to Part-Guided Proposal Generator (PGPG) and multi-domain knowledge distillation to further improve the accuracy. Our method is able to localize whole-body keypoints accurately and tracks humans simultaneously given inaccurate bounding boxes and redundant detections. We show a significant improvement over current state-of-the-art methods in both speed and accuracy on COCO-wholebody, COCO, PoseTrack, and our proposed Halpe-FullBody pose estimation dataset. Our model, source codes and dataset are made publicly available at [https://github.com/MVIG-SJTU/AlphaPose](https://github.com/MVIG-SJTU/AlphaPose).

<div align="center">
    <img src="alphapose_17.gif", width="400" alt><br>
    COCO 17 keypoints
</div>
<div align="center">
    <img src="alphapose_26.gif", width="400" alt><br>
    <b><a href="https://github.com/Fang-Haoshu/Halpe-FullBody">Halpe 26 keypoints</a></b> + tracking
</div>
<div align="center">
    <img src="alphapose_136.gif", width="400"alt><br>
    <b><a href="https://github.com/Fang-Haoshu/Halpe-FullBody">Halpe 136 keypoints</a></b> + tracking
</div>
<div align="center">
    <img src="alphapose_hybrik_smpl.gif", width="400"alt><br>
    <b><a href="https://github.com/Jeff-sjtu/HybrIK">SMPL</a></b> + tracking
</div>



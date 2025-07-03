---
title: "DeepVerse: 4D Autoregressive Video Generation as a World Model"
subtitle: "*arXiv preprint, 2025*"
date: 2025-07-03T00:00:00Z

# Publication type.
# Legend: 0 = Uncategorized; 1 = Conference paper; 2 = Journal article;
# 3 = Preprint / Working Paper; 4 = Report; 5 = Book; 6 = Book section;
# 7 = Thesis; 8 = Patent
publication_types: ["1"]

# Publication name and optional abbreviated publication name.
publication: "*Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2025*"
publication_short: ""

draft: false
featured: false
authors:
  - Yating Wang
  - admin
  - Mingyu Liu
  - Jiange Yang
  - Hao-Shu Fang
  - Tong He
tags:
  - Embodied AI
  - Robot Learning
  - Computer Vision
categories:
  - Embodied AI
  - Robot Learning
  - Computer Vision
external_link: https://xiaoxiao0406.github.io/vqvla.github.io/
links:
  - url: https://xiaoxiao0406.github.io/vqvla.github.io/
    name: Website
  - url: https://github.com/xiaoxiao0406/VQ-VLA
    name: GitHub
  - url: https://xiaoxiao0406.github.io/vqvla.github.io/static/pdfs/VQ-VLA.pdf
    name: PDF
  - url: https://arxiv.org/abs/2507.01016
    name: arXiv
image:
  filename: vqvla_pipeline.png
  focal_point: Smart
  preview_only: false
---

**Abstract:**
In this paper, we introduce an innovative vector quantization based action tokenizer built upon the largest-scale action trajectory dataset to date, leveraging over 100 times more data than previous approaches. This extensive dataset enables our tokenizer to capture rich spatiotemporal dynamics, resulting in a model that not only accelerates inference but also generates smoother and more coherent action outputs. Once trained, the tokenizer can be seamlessly adapted to a wide range of downstream tasks in a zero-shot manner, from short-horizon reactive behaviors to long-horizon planning. A key finding of our work is that the domain gap between synthetic and real action trajectories is marginal, allowing us to effectively utilize a vast amount of synthetic data during training without compromising real-world performance. To validate our approach, we conducted extensive experiments in both simulated environments and on real robotic platforms. The results demonstrate that as the volume of synthetic trajectory data increases, the performance of our tokenizer on downstream tasks improves significantly--most notably, achieving up to a 30% higher success rate on two real-world tasks in long-horizon scenarios. These findings highlight the potential of our action tokenizer as a robust and scalable solution for real-time embodied intelligence systems, paving the way for more efficient and reliable robotic control in diverse application domains.
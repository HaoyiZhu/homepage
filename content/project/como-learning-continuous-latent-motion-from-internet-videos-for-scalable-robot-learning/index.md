---
title: "CoMo: Learning Continuous Latent Motion from Internet Videos for Scalable Robot Learning"
subtitle: "*arXiv preprint, 2025*"
date: 2025-05-30T00:00:00Z

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
  - Jiange Yang
  - Yansong Shi
  - admin
  - Mingyu Liu
  - Kaijing Ma
  - Yating Wang
  - Gangshan Wu
  - Tong He
  - Limin Wang
tags:
  - Embodied AI
  - Robot Learning
categories:
  - Embodied AI
  - Robot Learning
external_link: https://arxiv.org/abs/2505.17006
links:
  - url: https://arxiv.org/pdf/2505.17006
    name: PDF
  - url: https://arxiv.org/abs/2505.17006
    name: arXiv
image:
  filename: pipeline.png
  focal_point: Smart
  preview_only: false
---

**Abstract:**

Learning latent motion from Internet videos is crucial for building generalist robots.
However, existing discrete latent action methods suffer from information loss and
struggle with complex and fine-grained dynamics. We propose CoMo, which
aims to learn more informative continuous motion representations from diverse,
internet-scale videos. CoMo employs a early temporal feature difference mechanism to prevent model collapse and suppress static appearance noise, effectively
discouraging shortcut learning problem. Furthermore, guided by the information
bottleneck principle, we constrain the latent motion embedding dimensionality to
achieve a better balance between retaining sufficient action-relevant information
and minimizing the inclusion of action-irrelevant appearance noise. Additionally,
we also introduce two new metrics for more robustly and affordably evaluating
motion and guiding motion learning methods development: (i) the linear probing
MSE of action prediction, and (ii) the cosine similarity between past-to-current and
future-to-current motion embeddings. Critically, CoMo exhibits strong zero-shot
generalization, enabling it to generate continuous pseudo actions for previously
unseen video domains. This capability facilitates unified policy joint learning
using pseudo actions derived from various action-less video datasets (such as
cross-embodiment videos and, notably, human demonstration videos), potentially
augmented with limited labeled robot data. Extensive experiments show that policies co-trained with CoMo pseudo actions achieve superior performance with both
diffusion and autoregressive architectures in simulated and real-world settings.
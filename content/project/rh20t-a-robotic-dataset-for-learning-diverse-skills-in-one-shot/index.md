---
title: "RH20T: A Robotic Dataset for Learning Diverse Skills in One-Shot"
subtitle: "RH20T"
date: 2023-07-02T00:00:00Z

# Publication type.
# Legend: 0 = Uncategorized; 1 = Conference paper; 2 = Journal article;
# 3 = Preprint / Working Paper; 4 = Report; 5 = Book; 6 = Book section;
# 7 = Thesis; 8 = Patent
publication_types: ["1"]

# Publication name and optional abbreviated publication name.
publication: "*IEEE International Conference on Robotics and Automation (ICRA), 2024*"
publication_short: ""

draft: false
featured: false
authors:
  - Hao-Shu Fang
  - Hongjie Fang
  - Zhenyu Tang
  - Jirong Liu
  - Junbo Wang
  - admin
  - Cewu Lu
tags:
  - Robot Learning
  - Embodied AI
categories:
  - Robot Learning
external_link: https://rh20t.github.io/
links:
  - url: https://rh20t.github.io/
    name: Project
  - url: https://arxiv.org/ftp/arxiv/papers/2307/2307.00595.pdf
    name: PDF
  - url: https://arxiv.org/abs/2307.00595
    name: Arxiv
image:
  filename: rh20t.gif
  focal_point: Smart
  preview_only: false
---

**Abstract:**

A key challenge in robotic manipulation in open domains is how to acquire diverse and generalizable skills for robots. Recent research in one-shot imitation learning has shown promise in transferring trained policies to new tasks based on demonstrations. This feature is attractive for enabling robots to acquire new skills and improving task and motion planning. However, due to limitations in the training dataset, the current focus of the community has mainly been on simple cases, such as push or pick-place tasks, relying solely on visual guidance. In reality, there are many complex skills, some of which may even require both visual and tactile perception to solve.

This paper aims to unlock the potential for an agent to generalize to hundreds of real-world skills with multi-modal perception. To achieve this, we have collected a dataset comprising over 110,000 contact-rich robot manipulation sequences across diverse skills, contexts, robots, and camera viewpoints, all collected in the real world. Each sequence in the dataset includes visual, force, audio, and action information, along with a corresponding human demonstration video. We have invested significant efforts in calibrating all the sensors and ensuring a high-quality dataset. The dataset will be made publicly available.
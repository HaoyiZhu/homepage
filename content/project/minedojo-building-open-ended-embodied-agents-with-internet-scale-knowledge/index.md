---
title: "MineDojo: Building Open-Ended Embodied Agents with Internet-Scale Knowledge"
date: 2022-06-17T00:00:00Z

# Publication type.
# Legend: 0 = Uncategorized; 1 = Conference paper; 2 = Journal article;
# 3 = Preprint / Working Paper; 4 = Report; 5 = Book; 6 = Book section;
# 7 = Thesis; 8 = Patent
publication_types: ["1"]

# Publication name and optional abbreviated publication name.
publication: "*Neural Information Processing Systems (NeurIPS), 2022*"
publication_short: "*NeurIPS 2022*"

abstract: Autonomous agents have made great strides in specialist domains like Atari games and Go. However, they typically learn tabula rasa in isolated environments with limited and manually conceived objectives, thus failing to generalize across a wide spectrum of tasks and capabilities. Inspired by how humans continually learn and adapt in the open world, we advocate a trinity of ingredients for building generalist agents: 1) an environment that supports a multitude of tasks and goals, 2) a large-scale database of multimodal knowledge, and 3) a flexible and scalable agent architecture. We introduce MineDojo, a new framework built on the popular Minecraft game that features a simulation suite with thousands of diverse open-ended tasks and an internet-scale knowledge base with Minecraft videos, tutorials, wiki pages, and forum discussions. Using MineDojo's data, we propose a novel agent learning algorithm that leverages large pre-trained video-language models as a learned reward function. Our agent is able to solve a variety of open-ended tasks specified in free-form language without any manually designed dense shaping reward. We open-source the simulation suite and knowledge bases ([https://minedojo.org](https://minedojo.org)) to promote research towards the goal of generally capable embodied agents.
summary: "Building open-ended agents with internet-scale knowledge in Minecraft."
draft: false
featured: true
authors:
  - Linxi Fan
  - Guanzhi Wang
  - Yunfan Jiang
  - Ajay Mandlekar
  - Yuncong Yang
  - admin
  - Andrew Tang
  - De-An Huang
  - Yuke Zhu
  - Anima Anandkumar
author_notes:
  - ''
  - ''
  - ''
  - ''
  - ''
  - ''
  - ''
  - ''
  - 'Equal advising'
  - 'Equal advising'
tags:
  - Computer Vision
  - Robot Learning
  - Others
categories:
  - Computer Vision
  - Robot Learning
  - Others
external_link: https://minedojo.org/
links:
  - url: https://minedojo.org/
    name: Project
  - url: https://arxiv.org/pdf/2206.08853.pdf
    name: PDF
  - url: https://arxiv.org/abs/2206.08853
    name: Arxiv
  - name: Twitter
    url: https://twitter.com/DrJimFan/status/1540381991052247041
  - name: Code
    url: https://github.com/MineDojo/MineDojo
  - url: https://minedojo.org/knowledge_base.html
    name: Database
  - url: https://developer.nvidia.com/blog/building-generally-capable-ai-agents-with-minedojo/
    name: Blog
  - url: https://www.youtube.com/watch?v=5LL6z1Ganbw
    name: Video

image:
  filename: pull.gif
  focal_point: Smart
  preview_only: false
---

{{< video src="pull.mp4" controls="yes">}}


---
draft: false
publication_types:
  - "1"
summary: Building open-ended agents with internet-scale knowledge in Minecraft.
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
publication_short: ""
publication: "💫✨ ***Outstanding Paper Award*** ✨💫. *Neural Information Processing Systems (NeurIPS) Dataset & Benchmark, 2022*"
author_notes:
  - ""
  - ""
  - ""
  - ""
  - ""
  - ""
  - ""
  - ""
  - Equal advising
  - Equal advising
title: "MineDojo: Building Open-Ended Embodied Agents with Internet-Scale Knowledge"
subtitle: "MineDojo"
date: 2022-06-17T00:00:00.000Z
featured: true
tags:
  - Embodied AI
  - Computer Vision
categories:
  - Embodied AI
  - Computer Vision
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
video: minedojo.mp4
  
---
**Abstract:**

Autonomous agents have made great strides in specialist domains like Atari games and Go. However, they typically learn tabula rasa in isolated environments with limited and manually conceived objectives, thus failing to generalize across a wide spectrum of tasks and capabilities. Inspired by how humans continually learn and adapt in the open world, we advocate a trinity of ingredients for building generalist agents: 1) an environment that supports a multitude of tasks and goals, 2) a large-scale database of multimodal knowledge, and 3) a flexible and scalable agent architecture. We introduce MineDojo, a new framework built on the popular Minecraft game that features a simulation suite with thousands of diverse open-ended tasks and an internet-scale knowledge base with Minecraft videos, tutorials, wiki pages, and forum discussions. Using MineDojo's data, we propose a novel agent learning algorithm that leverages large pre-trained video-language models as a learned reward function. Our agent is able to solve a variety of open-ended tasks specified in free-form language without any manually designed dense shaping reward. We open-source the simulation suite and knowledge bases ([https://minedojo.org](https://minedojo.org/)) to promote research towards the goal of generally capable embodied agents.

<video controls autoplay loop muted>
  <source src="minedojo.mp4" type="video/mp4">
</video>
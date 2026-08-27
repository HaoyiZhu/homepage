---
title: "Sparse Linear Attention: 当稀疏遇上线性注意力（2）"
title_en: "Sparse Linear Attention: When Sparsity Meets Linear Attention (2)"
authors: "Zhu, Haoyi and Deng, Haoge and Yang, Yuhang and Wang, Wen"
date: 2026-08-27
draft: false
---

在写完上一篇 blog 之后，笔者和小伙伴们（[@皓戈](https://bitterdhg.github.io/) [@雨航](https://yyvhang.github.io/) [@文哥](https://encounter1997.github.io/)）进行了一番深入的交流，讨论过程中有了一些新的发现和感悟，于是在这里记录和分享一下。

## 书接上文

在之前的[博客](https://haoyizhu.site/blog/sparse-linear-attention/)里，我们介绍了一些把稀疏注意力和线性注意力在序列维度上混合的方案。以笔者的二阶修正方案 [PWT](https://github.com/HaoyiZhu/Piecewise-Taylor-Attention) 为例：对每个 query $\boldsymbol{q}_t$，先按块打分选出 top-K 个块 $\mathcal{S}_t$ 做精确的 softmax attention；剩下的块 $\mathcal{U}_t$ 不直接丢弃，而是在块质心 $\bar{\boldsymbol{k}}_j$ 处做泰勒展开，压缩成带二阶修正的"虚拟 token"来做补偿。把输出写成分子分母的形式 $\boldsymbol{o}_t=\mathcal{N}_t/\mathcal{D}_t$：

$$
\mathcal{D}_t=\underbrace{\sum_{j\in\mathcal{S}_t}\sum_{i\in\mathcal{B}_j} e^{\boldsymbol{q}_t\cdot\boldsymbol{k}_i}}_{\text{精确分支}}+\underbrace{\sum_{j\in\mathcal{U}_t} B\,\exp\big(\boldsymbol{q}_t\cdot\bar{\boldsymbol{k}}_j+w_{t,j}\big)}_{\text{泰勒近似分支}}
$$

$$
\mathcal{N}_t=\sum_{j\in\mathcal{S}_t}\sum_{i\in\mathcal{B}_j} e^{\boldsymbol{q}_t\cdot\boldsymbol{k}_i}\,\boldsymbol{v}_i+\sum_{j\in\mathcal{U}_t} \exp\big(\boldsymbol{q}_t\cdot\bar{\boldsymbol{k}}_j+w_{t,j}\big)\Big(\sum_{i\in\mathcal{B}_j}\boldsymbol{v}_i+\boldsymbol{H}_j\,\boldsymbol{q}_t\Big)
$$

其中 $w_{t,j}=\frac{1}{2}\boldsymbol{q}_t^{\top}\boldsymbol{C}_j\boldsymbol{q}_t$ 是块质量的二阶修正（$\boldsymbol{C}_j$ 为块内 key 的协方差），$\boldsymbol{H}_j=\sum_{i\in\mathcal{B}_j}\boldsymbol{v}_i(\boldsymbol{k}_i-\bar{\boldsymbol{k}}_j)^{\top}$ 是块内 key–value 的一阶交叉统计量，两个分支共享同一个分母，归一化天然一致。

可以发现，这一类结合稀疏和线性注意力的方案（PWT / SLA / PISA 等），主要由两部分组成：**Top-K 选取 + 长尾补偿**。前者决定哪些位置需要精确计算注意力，后者决定如何近似剩下的区域，比如用泰勒展开的线性注意力做补偿。上一篇主要讨论的是补偿的部分——怎么把"丢弃"升级成"近似"；这一篇我们来讨论另一半：Top-K 到底该怎么选。

## 选取误差

目前视频模型里主流的一些稀疏注意力方案（比如 [VSA](https://arxiv.org/abs/2505.13389)、[PISA](https://arxiv.org/abs/2602.01077)、[PWT](https://github.com/HaoyiZhu/Piecewise-Taylor-Attention)），用的都是 [MoBA](https://arxiv.org/abs/2502.13189) 式的选取方案：分块、取块内平均、用质心打分选 top-K 个块。这类方案简单无参数，但精度的损失是显然的。首先是平均本身带来的损失：块内平均会抹平局部的尖峰，block size 越大，真正的 top-K token 就越可能被"平均掉"而漏选。此外还有一个不太常被讨论的误差点：位置编码。RoPE 给位置 $i$ 的 key 施加一个依赖位置的旋转 $\boldsymbol{R}_i$，并满足相对位置性质

$$
(\boldsymbol{R}_m\boldsymbol{q})\cdot(\boldsymbol{R}_n\boldsymbol{k})=\boldsymbol{q}^{\top}\boldsymbol{R}_{n-m}\,\boldsymbol{k}
$$

即内积只依赖于相对距离 $n-m$。但如果先做 RoPE 再做分块平均，平均出来的"质心"是

$$
\widetilde{\boldsymbol{k}}_j^{\mathrm{rope}}=\frac{1}{B}\sum_{i\in\mathcal{B}_j}\boldsymbol{R}_i\,\boldsymbol{k}_i
$$

它是 $B$ 个不同旋转的混合，无法写成"先平均、再在单一位置做旋转"的形式 $\boldsymbol{R}_{c_j}\bar{\boldsymbol{k}}_j$（注意 $\boldsymbol{R}_{c_j}\bar{\boldsymbol{k}}_j=\frac{1}{B}\sum_{i\in\mathcal{B}_j}\boldsymbol{R}_{c_j}\boldsymbol{k}_i$，与上式差的就是每个 token 上的旋转从 $\boldsymbol{R}_i$ 换成了 $\boldsymbol{R}_{c_j}$），于是块打分变成

$$
(\boldsymbol{R}_t\boldsymbol{q}_t)\cdot\widetilde{\boldsymbol{k}}_j^{\mathrm{rope}}=\frac{1}{B}\sum_{i\in\mathcal{B}_j}\boldsymbol{q}_t^{\top}\boldsymbol{R}_{i-t}\,\boldsymbol{k}_i
$$

一堆不同相对位置的内积混在一起，RoPE 的相对位置结构就被破坏了，块越大这个失真越严重。

对于 RoPE 的问题，一种缓解方案是在 RoPE 之前先做平均，然后用块中心的 index $c_j$ 补上 RoPE，即块打分取

$$
(\boldsymbol{R}_t\boldsymbol{q}_t)\cdot\big(\boldsymbol{R}_{c_j}\bar{\boldsymbol{k}}_j\big)=\boldsymbol{q}_t^{\top}\boldsymbol{R}_{c_j-t}\,\bar{\boldsymbol{k}}_j
$$

这样块打分重新成为一个定义良好的、只依赖相对位置 $c_j-t$ 的内积，training-free 的时候或许会有一点帮助。但如果我们想要的是一个 trainable 的选取方案，这种修补就未免有点不本质了——它还是在给"平均"打补丁，而不是正面回答"应该选谁"。

实际上，在 LLM 领域，稀疏注意力的 Top-K 选取更有名的大规模应用是 DeepSeek-V3.2 的 [DSA](https://arxiv.org/abs/2512.02556)（作为对比，Kimi 提出的 MoBA 并没有在 Kimi 的正式模型上大规模落地，可能也侧面反映出块平均选取的性能是受限的）。回顾一下 DSA 的方案：它额外训练一个轻量的 lightning indexer，给每个 (query, key) 对打分，再按分数选出 top-K 个 token（K 取 2048）做精确 attention：

$$
I_{t,s}=\sum_{j=1}^{H^I} w^{I}_{t,j}\cdot\mathrm{ReLU}\big(\boldsymbol{q}^{I}_{t,j}\cdot\boldsymbol{k}^{I}_{s}\big)
$$

其中 $w^{I}_{t,j}$ 是由 query 算出的 head 权重。Indexer 是可学习的，但它和主模型的训练是解耦的：先在一个 dense warm-up 阶段冻结主模型，用 KL 散度让 indexer 的打分分布去对齐主模型的注意力分布；再进入联合稀疏训练，此时 indexer 只吃自己的对齐损失，主模型只吃 LM loss，两者在计算图上互相 detach。

可以看到，这个 indexer 有两个核心优势：一是**可学习**，打分的标准可以从数据里训出来，而不是依赖"平均即代表"的启发式；二是 **token-wise 精确**，直接在 token 粒度上选 top-K，不存在块平均带来的漏选。两者综合起来，它的选取质量比 MoBA 式的块平均显然要更好。但另一方面，indexer 的打分本身是对全序列做的：每个 query 都要和所有（因果场景下即它之前的）key 算一次 $I_{t,s}$，复杂度仍然是 $\mathcal{O}(L^2)$，只是常数比主 attention 小得多（head 数少、head 维度低、可以跑 FP8，甚至可以考虑跨层共享打分等技巧）。当 token 数非常多，比如 10M 甚至更多时，这个二次项依然是无法承担的。

## 从粗到细

回顾一下两类 Top-K 选取的方案：MoBA 式块平均粗糙，但是无参数化且便宜；DSA 的 indexer 精确可学习，但是长序列下成本贵。于是，一个很自然的想法就是把它们结合起来，变成一个 coarse-to-fine 的方案，综合两者的优点——姑且把这个方案叫做 HSA（Hierarchical Sparse Attention）。具体来说分两步：

1. **Coarse（粗选）**：先做分块平均，用质心打分选出 top-M 个块。这里 M 故意取得比较大，宁可多选、不可漏选；
2. **Fine（精选）**：只在选出来的 M 个块内部，跑 DSA 式的可学习 indexer，做 token-wise 的精确 top-K 选取。

这里的洞察是：分块平均虽然有误差，但误差的幅度是有限的——只要 M 设得足够大，粗选出来的块就极大概率覆盖了真正的 top-K token。举个例子，我们可以先粗糙地选取 top 10% 的块，再在里面选取 top 1% 的 token：块平均可能会让某些真实 top 1% token 所在的块排名偏离 top 1%，但这个偏差很难把它们推出 top 10% 之外。当然，实际的超参需要根据训练情况调整，但核心思想是不变的：**先以低成本粗选一个大概率包含最终 top-K 的候选子集，再在子集内做可学习的精确选取**。

成本上也很直观：粗选每个 query 只需和 $L/B$ 个质心做内积，比 token 级打分便宜 $B$ 倍，而且还是无参数的；精选的 indexer 只在 $MB$ 个候选 token 上打分。总复杂度从 DSA 全序列打分的 $\mathcal{O}(L^2)$（常数小但仍是二次）降到

$$
\mathcal{O}\Big(\frac{L^2}{B}+L\cdot MB\Big)
$$

的"粗 + 细"两级，且精选阶段可以完整继承 DSA 的工程技巧（少 head、低维、FP8）。当序列进入 10M 量级时，这一点常数级的差距往往就是"能跑"和"不能跑"的区别。进一步地，如果一层粗选还不够，这个层次化的结构自然可以递归成更多级，这里就不展开了。

## 总结一下

这篇 blog 讨论了稀疏注意力里容易被忽略的另一半问题——Top-K 怎么选：块平均方案便宜，但有平均和位置编码带来的误差；可学习 indexer 精确，但全序列打分在超长序列下太贵。一个自然的出路是用 coarse-to-fine 的层次化选取（HSA）把两者结合起来，用便宜的粗选保证召回，用可学习的精选保证精度。同时注意，Top-K 选取和上一篇博客里讨论的线性近似补偿是**正交**的两件事：HSA 决定"哪里精确算"，PWT 式的补偿决定"剩下的地方怎么近似"，两者完全可以组合使用。实际上，笔者认为随着 token 长度越来越长，哪怕是 LLM，目前 DSA 这种"选完就丢"的做法也会遇到瓶颈——如果未来 scale 到 100M token 还是只精确看 top 2048，直觉上丢失的信息就太多了——在超长序列下把线性近似补偿也结合进来，会是一个很有价值的方向。

## 参考链接

- [上一篇：Sparse Linear Attention——当稀疏遇上线性注意力](https://haoyizhu.site/blog/sparse-linear-attention/)
- [MoBA: Mixture of Block Attention for Long-Context LLMs](https://arxiv.org/abs/2502.13189)
- [DeepSeek-V3.2 (DSA)](https://arxiv.org/abs/2512.02556)
- [VSA: Faster Video Diffusion with Trainable Sparse Attention](https://arxiv.org/abs/2505.13389)
- [SLA: Beyond Sparsity in Diffusion Transformers via Fine-Tunable Sparse–Linear Attention](https://arxiv.org/abs/2509.24006)
- [PISA: Piecewise Sparse Attention Is Wiser for Efficient Diffusion Transformers](https://arxiv.org/abs/2602.01077)
- [PWT 代码：HaoyiZhu/Piecewise-Taylor-Attention](https://github.com/HaoyiZhu/Piecewise-Taylor-Attention)

---
title: "视频 DeltaNet 的反思"
title_en: "Reflections on Video DeltaNet"
date: 2026-08-08
draft: false
---

自从 [Qwen3.5](https://qwen.ai/blog?id=qwen3.5) 用上 [Gated DeltaNet](https://arxiv.org/abs/2412.06464)、最近的 [Kimi K3](https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf) 用上 [KDA](https://arxiv.org/abs/2510.26692)，这类基于 delta rule 的线性注意力受到了很多关注。不过它的 state 是逐 token 往下传的，这一点对语言很自然，对视频却未必：语言的 token 天生有先后，视频虽然时序上是因果的，但一帧图像里那 $n$ 个 token 谁在前谁在后，全看我们自己挑的扫描方式，raster、蛇形、Hilbert 曲线都行，帧内 token 并没有一个天然的顺序。

当然，直接把视频 token 拉平了往 KDA 里送也能训，[Chimera](https://papers.cool/arxiv/2607.28611) 就是这么做的；多模态 LLM 也普遍先用双向 attention 的 ViT 编码视觉输入，再把特征和文本 token 一起 causal 地送进 backbone。但笔者一直觉得，一个视频原生的算子，也就是帧内换一种扫描顺序、结果完全不变的算子，即便效果未必更好，也会更优雅，也更有趣。

笔者之前的工作 [SANA-WM](https://arxiv.org/abs/2605.15178) 往这个方向走过一步，把 GDN 从逐 token 改成了逐帧。不过当时笔者对线性注意力的认识还很浅薄，从后来的实验结果看效果也不算好。最近重新想这个问题的时候有了些新的感悟，在这里记录一下。

> 注：本文的数学推导，部分由笔者与 Claude 讨论完成。

## 前情提要

先约定一下记号。向量都是列向量，query/key 维度为 $d_k$、value 维度为 $d_v$，state 记为 $\boldsymbol{S}\in\mathbb{R}^{d_v\times d_k}$。回顾 delta rule 时 $t$ 是 token 的下标；之后 $t$ 表示帧，帧内的 $n$ 个 token 用 $i$ 编号，第 $t$ 帧的 key、value 按行堆叠成 $\boldsymbol{K}_t\in\mathbb{R}^{n\times d_k}$、$\boldsymbol{V}_t\in\mathbb{R}^{n\times d_v}$。按惯例，$\boldsymbol{k}$ 已做 L2 归一化，即 $\boldsymbol{k}^{\top}\boldsymbol{k}=1$。

线性注意力把历史的 KV 压成一个固定大小的 state $\boldsymbol{S}_t$，给定 query $\boldsymbol{q}_t$ 就输出 $\boldsymbol{o}_t=\boldsymbol{S}_t\boldsymbol{q}_t$。[最朴素的更新规则](https://arxiv.org/abs/2006.16236)是累加：

$$
\boldsymbol{S}_t=\boldsymbol{S}_{t-1}+\boldsymbol{v}_t\boldsymbol{k}_t^{\top}\tag{1}
$$

如果一次进来一批 token，这一批的累加就是 $\boldsymbol{S}\leftarrow\boldsymbol{S}+\boldsymbol{V}^{\top}\boldsymbol{K}$，一堆外积的和写成了一次矩阵乘法。
显然，$(1)$ 没有任何遗忘机制，序列一长，写进去的东西就会互相稀释和干扰。

要改进它，最好用的视角是 [TTT](https://arxiv.org/abs/2407.04620) 式的 online learning（[苏剑林的这篇博客](https://kexue.fm/archives/11033)也介绍过）：把 $(\boldsymbol{k}_1,\boldsymbol{v}_1),\cdots,(\boldsymbol{k}_t,\boldsymbol{v}_t)$ 看成语料，state 就是一个线性模型 $f(\boldsymbol{S};\boldsymbol{k})=\boldsymbol{S}\boldsymbol{k}$，读出则是 $\boldsymbol{o}_t=f(\boldsymbol{S}_t;\boldsymbol{q}_t)$。于是"写入"就变成了"拿新样本把这个模型训一步"。定义单样本损失

$$
\mathcal{L}_t(\boldsymbol{S})=\tfrac12\lVert\boldsymbol{S}\boldsymbol{k}_t-\boldsymbol{v}_t\rVert^2,\qquad
\nabla_{\boldsymbol{S}}\mathcal{L}_t=\big(\boldsymbol{S}\boldsymbol{k}_t-\boldsymbol{v}_t\big)\boldsymbol{k}_t^{\top}\tag{2}
$$

以 $\beta_t$ 为学习率走一步梯度下降：

$$
\boldsymbol{S}_t=\boldsymbol{S}_{t-1}-\beta_t\nabla_{\boldsymbol{S}}\mathcal{L}_t(\boldsymbol{S}_{t-1})
=\boldsymbol{S}_{t-1}\big(\boldsymbol{I}-\beta_t\boldsymbol{k}_t\boldsymbol{k}_t^{\top}\big)+\beta_t\boldsymbol{v}_t\boldsymbol{k}_t^{\top}\tag{3}
$$

这就是 [delta rule](https://arxiv.org/abs/2102.11174)。它比累加式多做的就是那个 $\boldsymbol{I}-\beta_t\boldsymbol{k}_t\boldsymbol{k}_t^{\top}$：先把 state 在 $\boldsymbol{k}_t$ 方向上的旧内容按比例擦掉，再写新的。$\beta_t=1$ 时，当前这个 key 的残差正好归零，新值把旧值完全替换掉；累加式则只会把两者叠在一起。再加一个遗忘门 $\alpha_t\in(0,1)$ 就是 Gated DeltaNet：

$$
\boldsymbol{S}_t=\boldsymbol{S}_{t-1}\,\alpha_t\big(\boldsymbol{I}-\beta_t\boldsymbol{k}_t\boldsymbol{k}_t^{\top}\big)+\beta_t\boldsymbol{v}_t\boldsymbol{k}_t^{\top}\tag{4}
$$

[KDA](https://arxiv.org/abs/2510.26692) 的改动则是把标量门换成通道级的对角门 $\mathrm{diag}(\boldsymbol{a}_t)$，具体形式可以参考原文。对后文而言，重要的只是它们都能写成

$$
\boldsymbol{S}_t=\boldsymbol{S}_{t-1}\boldsymbol{A}_t+\boldsymbol{b}_t\tag{5}
$$

以 GDN 为例，$\boldsymbol{A}_t=\alpha_t(\boldsymbol{I}-\beta_t\boldsymbol{k}_t\boldsymbol{k}_t^{\top})$、$\boldsymbol{b}_t=\beta_t\boldsymbol{v}_t\boldsymbol{k}_t^{\top}$。也就是说，每个 token 都在 state 上作用一个仿射变换，一段序列就是这些仿射变换的有序复合。

## 目标分析

我们希望一帧内的 $n$ 个 token 不管按什么顺序排，结果都一样。写成数学语言是两条：

- **置换不变**：把帧内 token 重排之后，传给下一帧的 $\boldsymbol{S}_t$ 完全不变；
- **置换等变**：每个 token 的输出跟着它自己一起重排，取值不变。

这两条得分开讲，因为前者推不出后者：即便 $\boldsymbol{S}_t$ 不变，只要帧内第 $i$ 个 token 读的还是它自己那一步的前缀 state，输出就照样依赖顺序。

一个自然的想法是让帧内各 token 的 key 两两正交，这样两个更新矩阵可交换，谁先谁后就无所谓了。但 $d_k$ 维空间里最多只有 $d_k$ 个两两正交的方向，这条路要求 $n\le d_k$，而一帧的 token 数通常远大于 $d_k$。

事实上，只要还是 GDN 这类逐 token 的 rank-1 仿射更新按序复合，对一般输入就没法从结构上保证置换不变。这本质上就是矩阵乘法没有交换律。把一帧的 $n$ 个仿射变换按顺序 $\pi$ 复合起来：

$$
\boldsymbol{S}_t=\boldsymbol{S}_{t-1}\prod_{i=1}^{n}\boldsymbol{A}_{\pi(i)}+\sum_{i=1}^{n}\boldsymbol{b}_{\pi(i)}\prod_{j\gt i}\boldsymbol{A}_{\pi(j)}\tag{6}
$$

两项都得与 $\pi$ 无关。先看第一项：交换相邻两位再把两侧消去（$\boldsymbol{A}_i$ 在 $\alpha_i\ne0$、$\beta_i\ne1$ 时可逆），就得到 $\boldsymbol{A}_i\boldsymbol{A}_j=\boldsymbol{A}_j\boldsymbol{A}_i$；而相邻对换可以生成任意置换，所以第一项与顺序无关，就等价于所有 $\boldsymbol{A}_i$ 两两可交换。标量 $\alpha$ 不影响可交换性，直接展开两个更新矩阵的交换子：

$$
\big[\boldsymbol{I}-\beta_i\boldsymbol{k}_i\boldsymbol{k}_i^{\top},\ \boldsymbol{I}-\beta_j\boldsymbol{k}_j\boldsymbol{k}_j^{\top}\big]
=\beta_i\beta_j\,(\boldsymbol{k}_i^{\top}\boldsymbol{k}_j)\big(\boldsymbol{k}_i\boldsymbol{k}_j^{\top}-\boldsymbol{k}_j\boldsymbol{k}_i^{\top}\big)\tag{7}
$$

要它为零，除了 $\beta=0$（等于没写）之外，只能是两个 key 正交或者平行。平行也不行：取 $\alpha_i=\alpha_j=1$、$\boldsymbol{k}_i=\boldsymbol{k}_j=\boldsymbol{k}$，由 $\boldsymbol{k}^{\top}(\boldsymbol{I}-\beta\boldsymbol{k}\boldsymbol{k}^{\top})=(1-\beta)\boldsymbol{k}^{\top}$ 算一下，两种顺序的写入项之差是 $\beta_i\beta_j(\boldsymbol{v}_j-\boldsymbol{v}_i)\boldsymbol{k}^{\top}$，只有两个 value 恰好相同时才为零。所以要覆盖一般输入，就只剩正交，又回到了 $n\le d_k$。

第二项还给出一条独立的限制。正交时有 $\boldsymbol{k}_i^{\top}\boldsymbol{A}_j=\alpha_j\boldsymbol{k}_i^{\top}$，于是第二项化为 $\sum_i\beta_i\big(\prod_{j\ \text{排在}\ i\ \text{之后}}\alpha_j\big)\boldsymbol{v}_i\boldsymbol{k}_i^{\top}$。括号里的系数取决于哪些 token 排在 $i$ 后面；各 $\alpha_j$ 相等时它就退化成只依赖位置的幂次。可见只要帧内还留着逐 token 的遗忘门，顺序就会通过 $\alpha$ 编码进 state，key 全部正交也没用。（换个角度说，GDN 自带了一份位置信息，这也呼应了 KDA 可以使用 NoPE。）

结论是：逐 token 的方案既要求帧内 key 两两正交，又要求去掉帧内的遗忘门。前一条在视频里基本不可能，后一条则必须把门控改成帧级的。所以更自然的做法是把一帧当成一个整体来更新。

## 帧级初探

SANA-WM 里的做法是这样的：递推的单位从 token 换成帧，一帧的所有 token 一次性写进去。把它和单 token 那一行并排放，区别一眼就看得出来：

$$
\boldsymbol{S}_t=\boldsymbol{S}_{t-1}\big(\boldsymbol{I}-\beta_t\,\boldsymbol{k}_t\boldsymbol{k}_t^{\top}\big)+\beta_t\,\boldsymbol{v}_t\boldsymbol{k}_t^{\top}
\qquad\text{（单 token）}\tag{8}
$$

$$
\boldsymbol{S}_t=\boldsymbol{S}_{t-1}\,\alpha_t\big(\boldsymbol{I}-\boldsymbol{K}_t^{\top}\boldsymbol{\Gamma}\boldsymbol{K}_t\big)+\boldsymbol{V}_t^{\top}\boldsymbol{\Gamma}\boldsymbol{K}_t
\qquad\text{（整帧）}\tag{9}
$$

也就是把单个 token 的两个外积换成整帧的加权外积和：

$$
\boldsymbol{K}_t^{\top}\boldsymbol{\Gamma}\boldsymbol{K}_t=\sum_{i=1}^{n}\gamma_i\boldsymbol{k}_i\boldsymbol{k}_i^{\top}\in\mathbb{R}^{d_k\times d_k},
\qquad
\boldsymbol{V}_t^{\top}\boldsymbol{\Gamma}\boldsymbol{K}_t=\sum_{i=1}^{n}\gamma_i\boldsymbol{v}_i\boldsymbol{k}_i^{\top}\in\mathbb{R}^{d_v\times d_k}\tag{10}
$$

其中 $\boldsymbol{\Gamma}=\mathrm{diag}(\gamma_1,\cdots,\gamma_n)$ 是每个 token 的写入强度（在 SANA-WM 里就是 GDN 的 $\beta_i$），$\alpha_t$ 是帧级遗忘门，每帧一个。取 $n=1$、$\boldsymbol{\Gamma}=\beta_t$、$\alpha_t=1$，$(9)$ 就退回 $(8)$。读出时让帧内所有 query 读同一个 $\boldsymbol{S}_t$。

用 online learning 的视角看，$(9)$ 就是对整帧的总损失走一步普通的梯度下降。既然 delta rule 是拿一个 token 训一步，而帧内又没有先后，那就把这一帧的 $n$ 个 token 当成一个 batch，写出总损失（逐样本写和矩阵写是一回事）：

$$
\mathcal{L}_t(\boldsymbol{S})=\sum_{i=1}^{n}\frac{\gamma_i}{2}\big\lVert\boldsymbol{S}\boldsymbol{k}_i-\boldsymbol{v}_i\big\rVert^2
=\tfrac12\big\lVert\big(\boldsymbol{S}\boldsymbol{K}_t^{\top}-\boldsymbol{V}_t^{\top}\big)\boldsymbol{\Gamma}^{1/2}\big\rVert_F^2\tag{11}
$$

求梯度：

$$
\nabla_{\boldsymbol{S}}\mathcal{L}_t(\boldsymbol{S})
=\big(\boldsymbol{S}\boldsymbol{K}_t^{\top}-\boldsymbol{V}_t^{\top}\big)\boldsymbol{\Gamma}\boldsymbol{K}_t
=\boldsymbol{S}\,\boldsymbol{K}_t^{\top}\boldsymbol{\Gamma}\boldsymbol{K}_t-\boldsymbol{V}_t^{\top}\boldsymbol{\Gamma}\boldsymbol{K}_t\tag{12}
$$

于是 $\boldsymbol{S}_{t-1}-\nabla\mathcal{L}_t(\boldsymbol{S}_{t-1})$ 正好就是 $(9)$（不计 $\alpha_t$），即学习率为 1 的一步梯度下降；$\alpha_t$ 的作用则是先把旧 state 整体衰减一遍，再走这一步。用数值分析的话说，这是**显式**（前向 Euler）一步：梯度算在出发点上。

这样一改，两条性质就都拿到了。把帧内 token 重排相当于给 $\boldsymbol{K}_t,\boldsymbol{V}_t$ 左乘一个置换矩阵 $\boldsymbol{P}_\pi$、给 $\boldsymbol{\Gamma}$ 做相应的共轭，而 $\boldsymbol{P}_\pi^{\top}\boldsymbol{P}_\pi=\boldsymbol{I}$，所以

$$
(\boldsymbol{P}_\pi\boldsymbol{K}_t)^{\top}\big(\boldsymbol{P}_\pi\boldsymbol{\Gamma}\boldsymbol{P}_\pi^{\top}\big)(\boldsymbol{P}_\pi\boldsymbol{K}_t)=\boldsymbol{K}_t^{\top}\boldsymbol{\Gamma}\boldsymbol{K}_t\tag{13}
$$

写入那一块同理，于是 $\boldsymbol{S}_t$ 完全不变；输出侧因为帧内所有 query 读的是同一个 $\boldsymbol{S}_t$，也自动跟着 token 一起重排。说白了，置换不变就是"求和与顺序无关"这一件事。

后面要反复讨论这两块（尤其是第一块的特征值），所以给它们各起个短名字：

$$
\boldsymbol{G}_t=\boldsymbol{K}_t^{\top}\boldsymbol{\Gamma}\boldsymbol{K}_t,\qquad
\boldsymbol{C}_t=\boldsymbol{V}_t^{\top}\boldsymbol{\Gamma}\boldsymbol{K}_t\tag{14}
$$

$\boldsymbol{G}_t$ 是"擦除"，对称半正定；$\boldsymbol{C}_t$ 是"写入"。显式一步于是简写成 $\boldsymbol{S}_t=\boldsymbol{S}_{t-1}(\boldsymbol{I}-\boldsymbol{G}_t)+\boldsymbol{C}_t$。

但这么直接写是训不稳的。$\boldsymbol{G}_t$ 是帧内 $n$ 项之和，某个方向的特征值 $\lambda$ 一旦超过 2，$1-\lambda$ 的绝对值就大于 1（算上帧级门 $\alpha_t$ 是 $\lvert\alpha_t(1-\lambda)\rvert\gt1$），旧 state 会被放大；这种放大每帧都来一点，就发散了。SANA-WM 的处理是在 L2 归一化之外再给 key 除一个 $\sqrt{n}$。为了记号一致，下文 $\boldsymbol{K}_t$ 始终指单位长度的 key，$\boldsymbol{G}_t,\boldsymbol{C}_t$ 也都按它定义。这样缩放之后，擦除项变成 $\boldsymbol{G}_t/n$、写入项变成 $\boldsymbol{C}_t/\sqrt{n}$：$\boldsymbol{G}_t$ 对 key 是二次的，$\boldsymbol{C}_t$ 是一次的，两者缩的幅度差一个 $\sqrt{n}$。此时擦除项的迹是 $\frac1n\sum_i\gamma_i\le1$，而半正定矩阵的最大特征值不超过迹，$\boldsymbol{I}-\boldsymbol{G}_t/n$ 就不会放大旧 state 了。

于是这一节最终的更新式，也就是 SANA-WM 里实际跑的，是

$$
\boldsymbol{S}_t=\boldsymbol{S}_{t-1}\,\alpha_t\Big(\boldsymbol{I}-\frac{\boldsymbol{G}_t}{n}\Big)+\frac{\boldsymbol{C}_t}{\sqrt{n}}
\qquad\text{即}\qquad
\boldsymbol{S}_t=\boldsymbol{S}_{t-1}\,\alpha_t\Big(\boldsymbol{I}-\frac{\boldsymbol{K}_t^{\top}\boldsymbol{\Gamma}\boldsymbol{K}_t}{n}\Big)+\frac{\boldsymbol{V}_t^{\top}\boldsymbol{\Gamma}\boldsymbol{K}_t}{\sqrt{n}}\tag{15}
$$

## 反思一下

笔者最近重新审视这套做法，发现一个隐患。关键在于这个 $\sqrt{n}$ 到底影响了什么。决定旧 state 留下多少的是 $\boldsymbol{G}_t/n$，第 $j$ 个特征方向上的保留因子是 $1-\lambda_j/n$。而 $(15)$ 里那步缩放换来的，恰好是这些量的总和有个上限：

$$
\sum_j\frac{\lambda_j}{n}=\frac{\mathrm{tr}\,\boldsymbol{G}_t}{n}\le1\tag{16}
$$

$d_k$ 个方向分这个 1，平均每帧在每个方向上只能擦掉 $1/d_k$ 上下，$d_k=112$ 就是不到 $1\%$。这才是那个 $1/\sqrt{n}$ 真正的代价：挨刀的不是某几个 token 的写入，而是整帧的擦除总量，被压到了 1。

笔者顺手在 SANA-WM 的 checkpoint 上量了一下。擦除项 $\boldsymbol{S}_{t-1}\boldsymbol{G}_t/n$ 对 state 的相对贡献中位数只有 $0.1\%$，写入项是 $37\%$；把擦除项整个置零、退回成 gated 累加式线性注意力，解码出来的画面也只差 $35$ dB（同一条计算路径的数值噪声本身就有 $46$ dB）。也就是说，这些 GDN 层基本是在当累加式线性注意力用，delta rule 那部分只出了很少的力。

还有一个数字值得一看：$\mathrm{tr}\,\boldsymbol{G}_t/n$ 的中位数是 $0.013$，比 $(16)$ 给的上限还低两个数量级。可见真正卡住写入强度的并不是稳定性，而是 $\beta_i\le1$ 配上被缩小的 key 之后，够得着的范围本身就这么窄。实测 $\beta_i$ 的中位数已经顶到 $0.98$ 了。

## 换个思路

瓶颈的根子在于：显式这一步想稳定，就得把擦除总量压住。那有没有一种走法，稳定性完全不依赖写入强度？有，把梯度算在**落点**上就行，也就是走**隐式**（[后向 Euler](https://en.wikipedia.org/wiki/Backward_Euler_method)）的一步：

$$
\boldsymbol{S}_t=\boldsymbol{S}_{t-1}-\nabla\mathcal{L}_t(\boldsymbol{S}_t)\tag{17}
$$

这里 $\boldsymbol{S}_t$ 出现在等式两边，是个隐式方程，得解一下。好在梯度式 $(12)$ 是 $\nabla\mathcal{L}_t(\boldsymbol{S})=\boldsymbol{S}\boldsymbol{G}_t-\boldsymbol{C}_t$，它对任意 $\boldsymbol{S}$ 都成立，把还没解出来的 $\boldsymbol{S}_t$ 当作自变量代进去完全合法；而且帧级损失是二次的，梯度对 $\boldsymbol{S}$ 是仿射的，代进去得到的是关于 $\boldsymbol{S}_t$ 的**线性**方程，有闭式解。把 $\boldsymbol{S}_t$ 收到一边：

$$
\boldsymbol{S}_t\big(\boldsymbol{I}+\boldsymbol{G}_t\big)=\boldsymbol{S}_{t-1}+\boldsymbol{C}_t
\qquad\Longrightarrow\qquad
\boldsymbol{S}_t=\big(\boldsymbol{S}_{t-1}+\boldsymbol{C}_t\big)\big(\boldsymbol{I}+\boldsymbol{G}_t\big)^{-1}\tag{18}
$$

把简写还原，它就是

$$
\boldsymbol{S}_t=\big(\boldsymbol{S}_{t-1}+\boldsymbol{V}_t^{\top}\boldsymbol{\Gamma}\boldsymbol{K}_t\big)\big(\boldsymbol{I}+\boldsymbol{K}_t^{\top}\boldsymbol{\Gamma}\boldsymbol{K}_t\big)^{-1}\tag{19}
$$

和显式的 $(15)$ 比，改动只有一处：擦除从"减去 $\boldsymbol{K}_t^{\top}\boldsymbol{\Gamma}\boldsymbol{K}_t$"变成了"除以 $\boldsymbol{I}+\boldsymbol{K}_t^{\top}\boldsymbol{\Gamma}\boldsymbol{K}_t$"。

保留因子从 $1-\gamma$ 变成 $\frac{1}{1+\gamma}$：单调下降，永远落在 $(0,1]$ 内，$\gamma$ 再大也不会变负、不会放大；矩阵版同理，$\boldsymbol{G}_t$ 半正定就保证 $(\boldsymbol{I}+\boldsymbol{G}_t)^{-1}$ 的特征值都在 $(0,1]$。这就是隐式相对显式的全部好处：$\gamma_i$ 只要非负就行，不必为稳定压尺度，也不必每帧先估一个特征值上界；$\gamma$ 大的方向擦得干净，小的方向基本不动，谁也不用为谁让路。

特别地，取 $n=1$，由 [Sherman–Morrison 公式](https://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula) $(\boldsymbol{I}+\gamma\boldsymbol{k}\boldsymbol{k}^{\top})^{-1}=\boldsymbol{I}-\frac{\gamma}{1+\gamma}\boldsymbol{k}\boldsymbol{k}^{\top}$，代进去就是学习率为 $\frac{\gamma}{1+\gamma}$ 的 delta rule。换句话说，它是单 token 情形的推广，写入强度和 delta 学习率之间只差一个简单换算：

$$
\gamma_i=\frac{\beta_i}{1-\beta_i},\qquad \beta_i=\frac{\gamma_i}{1+\gamma_i}\tag{20}
$$

实现上这很省事：$\beta_i=\sigma(z_i)$ 本来就是某个投影过 sigmoid 得来的，而 $\sigma(z)/(1-\sigma(z))=e^{z}$，所以 $\gamma_i=e^{z_i}$，把 sigmoid 换成 exp 就完了。
这个更新不要求 key 正交，也不限制 $n$ 与 $d_k$ 的大小关系；它只通过 $\boldsymbol{G}_t,\boldsymbol{C}_t$ 依赖这一帧，置换不变是精确的，没有丝毫残留。那它和逐 token 的顺序连乘差多少呢？展开来看：写入很弱时 $(\boldsymbol{I}+\boldsymbol{G}_t)^{-1}=\boldsymbol{I}-\boldsymbol{G}_t+\mathcal{O}(\boldsymbol{G}_t^2)$，而 $\prod_i(\boldsymbol{I}-\beta_i\boldsymbol{k}_i\boldsymbol{k}_i^{\top})$ 的一阶项同样是 $-\boldsymbol{G}_t$，两者在一阶上一致；差别从二阶才开始，顺序连乘里每对 $(i,j)$ 谁先谁后会造成不同的二阶项，其不对称部分正是 $(7)$ 里的交换子。所以隐式步可以看成顺序 delta rule 的对称化，而帧内顺序的影响本身就是 $\mathcal{O}(\gamma^2)$ 的二阶效应。

计算上不必真的求逆。$\boldsymbol{I}+\boldsymbol{G}_t$ 对称正定、特征值全 $\ge1$，forward 就是对它做一次 [Cholesky 分解](https://en.wikipedia.org/wiki/Cholesky_decomposition)、再解一次 $\boldsymbol{S}_t(\boldsymbol{I}+\boldsymbol{G}_t)=\boldsymbol{S}_{t-1}+\boldsymbol{C}_t$，要分解的矩阵大小与 $n$ 无关。backward 也不必对 Cholesky 分解本身求导，把 forward 分解出来的那个三角矩阵留着重用就行。记 $\boldsymbol{A}=\boldsymbol{I}+\boldsymbol{G}_t$、$\boldsymbol{B}=\boldsymbol{S}_{t-1}+\boldsymbol{C}_t$，并约定加一道上划线表示损失对它的梯度（即 $\bar{\boldsymbol{X}}=\partial\mathcal{L}/\partial\boldsymbol{X}$）。对 $\boldsymbol{S}_t\boldsymbol{A}=\boldsymbol{B}$ 两边取微分得 $\mathrm{d}\boldsymbol{S}_t=(\mathrm{d}\boldsymbol{B}-\boldsymbol{S}_t\,\mathrm{d}\boldsymbol{A})\boldsymbol{A}^{-1}$，代进 $\langle\bar{\boldsymbol{S}}_t,\mathrm{d}\boldsymbol{S}_t\rangle$ 再把 $\boldsymbol{A}^{-1}$ 挪到左边，就读出

$$
\boldsymbol{Z}=\bar{\boldsymbol{S}}_t\boldsymbol{A}^{-1},\qquad
\bar{\boldsymbol{B}}=\boldsymbol{Z},\qquad
\bar{\boldsymbol{A}}=-\boldsymbol{S}_t^{\top}\boldsymbol{Z}\tag{21}
$$

也就是再解一次同一个方程、加一次矩阵乘法。注意 $\bar{\boldsymbol{A}}$ 一般不对称，而 $\boldsymbol{A}$ 只能沿对称方向变动，所以往下推之前先取 $\boldsymbol{H}=\tfrac12(\bar{\boldsymbol{A}}+\bar{\boldsymbol{A}}^{\top})$，再走一遍链式法则：

$$
\bar{\boldsymbol{V}}_t=\boldsymbol{\Gamma}\boldsymbol{K}_t\boldsymbol{Z}^{\top},\qquad
\bar{\boldsymbol{K}}_t=\boldsymbol{\Gamma}\boldsymbol{V}_t\boldsymbol{Z}+2\boldsymbol{\Gamma}\boldsymbol{K}_t\boldsymbol{H},\qquad
\bar{\gamma}_i=\boldsymbol{k}_i^{\top}\boldsymbol{H}\boldsymbol{k}_i+\boldsymbol{v}_i^{\top}\boldsymbol{Z}\boldsymbol{k}_i\tag{22}
$$

至于要往上一帧传的那一项：$\boldsymbol{B}=\boldsymbol{S}_{t-1}+\boldsymbol{C}_t$ 只是个求和，梯度原样落到两个加数上，所以 $\bar{\boldsymbol{S}}_{t-1}=\bar{\boldsymbol{C}}_t=\boldsymbol{Z}$，不需要额外计算。于是前反向合起来，是一次 Cholesky 分解、两次线性方程求解，外加若干矩阵乘法。

除了更新式本身，还有两处得跟着调整。

一是遗忘门要挪到帧边界上，$\boldsymbol{a}_t$ 每帧一组、仍然可以是通道级的。

二是写入强度还需要一个整体尺度 $s$，不过这次图的是效果，稳定性已经不用操心了。把 $\gamma_i$ 写成 $s\,e^{z_i}$、$e^{z_i}$ 量级为 $1$，则 $\mathrm{tr}\,\boldsymbol{G}_t\approx ns$，摊在最多 $\min(n,d_k)$ 个非零特征值上；$n\gt d_k$ 时平均每个方向 $ns/d_k$ 的量级，旧 state 大致只剩 $d_k/(d_k+ns)$。也就是说 $s$ 取 $\mathcal{O}(1)$ 的话，一帧就把历史抹掉了（$n\sim10^3$、$d_k\sim10^2$ 时只剩一成），$s$ 该落在 $d_k/n$ 附近甚至更小；$s=d_k/n$ 时旧 state 和这一帧大致各占一半，再往下才谈得上跨帧的长记忆。注意 $s$ 同时进 $\boldsymbol{G}_t$ 和 $\boldsymbol{C}_t$，擦除和写入同比缩放，这和显式的 $(15)$ 不同：那里给 key 乘 $1/\sqrt{n}$，擦除吃 $1/n$，写入只吃 $1/\sqrt{n}$。

合起来，完整的更新式是

$$
\begin{gathered}
\boldsymbol{S}_t=\Big(\boldsymbol{S}_{t-1}\mathrm{diag}(\boldsymbol{a}_t)+\boldsymbol{C}_t\Big)\big(\boldsymbol{I}+\boldsymbol{G}_t\big)^{-1}\\[4pt]
\boldsymbol{G}_t=\boldsymbol{K}_t^{\top}\boldsymbol{\Gamma}\boldsymbol{K}_t,\qquad
\boldsymbol{C}_t=\boldsymbol{V}_t^{\top}\boldsymbol{\Gamma}\boldsymbol{K}_t,\qquad
\boldsymbol{\Gamma}=s\,\mathrm{diag}\big(e^{z_1},\cdots,e^{z_n}\big)
\end{gathered}\tag{23}
$$

它对 $\boldsymbol{S}_{t-1}$ 仍然是仿射的，所以跨帧那层递推照样可以写成仿射对做 associative scan 并行。

## 有趣巧合

$n\le d_k$ 在视频里基本碰不上，但这个极限情形很有意思，它会把最开始那条正交化的思路重新接回来。

设 $\boldsymbol{K}_t$ 行满秩（这就要求 $n\le d_k$），把写入强度整体推到无穷，这一帧的损失被压到零，每个 $\boldsymbol{S}_t\boldsymbol{k}_i=\boldsymbol{v}_i$ 都精确成立，也就是 $\boldsymbol{S}_t\boldsymbol{K}_t^{\top}=\boldsymbol{V}_t^{\top}$。这 $n$ 个方程只约束 key 张成的那个子空间；在它的正交补上 $\boldsymbol{G}_t$ 是零，$(\boldsymbol{I}+\boldsymbol{G}_t)^{-1}$ 严格等于恒等，旧 state 原样留下。两条合起来，$\boldsymbol{S}_t$ 被唯一定出：

$$
\boldsymbol{S}_t=\boldsymbol{S}_{t-1}\big(\boldsymbol{I}-\boldsymbol{\Pi}_t\big)+\boldsymbol{V}_t^{\top}\big(\boldsymbol{K}_t\boldsymbol{K}_t^{\top}\big)^{-1}\boldsymbol{K}_t,
\qquad \boldsymbol{\Pi}_t=\boldsymbol{K}_t^{\top}\big(\boldsymbol{K}_t\boldsymbol{K}_t^{\top}\big)^{-1}\boldsymbol{K}_t\tag{24}
$$

代回去验一下：$\boldsymbol{\Pi}_t\boldsymbol{K}_t^{\top}=\boldsymbol{K}_t^{\top}$，所以第一项乘 $\boldsymbol{K}_t^{\top}$ 为零，第二项正好给出 $\boldsymbol{V}_t^{\top}$，$\boldsymbol{S}_t\boldsymbol{K}_t^{\top}=\boldsymbol{V}_t^{\top}$ 确实成立。$\boldsymbol{\Pi}_t$ 是到这些 key 张成的子空间上的正交投影，只依赖这个子空间本身，置换不变是显然的。含义也很直白：单 token 的 delta rule 擦掉一个方向再写，帧级的 delta rule 就是擦掉整个 key 子空间再写。

那如果真把帧内 key 先正交化、再照旧跑逐 token 的循环呢？注意正交化本身也得与顺序无关。[Gram–Schmidt](https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process) 不行：它保留第一个向量不动，后面的依次减去在前面所有向量上的投影，这本身就是顺序相关的，用它只是把顺序依赖从递推挪进了预处理。[Löwdin 对称正交化](https://en.wikipedia.org/wiki/Orthogonalization) $\widetilde{\boldsymbol{K}}_t=(\boldsymbol{K}_t\boldsymbol{K}_t^{\top})^{-1/2}\boldsymbol{K}_t$ 则可以：把 $\boldsymbol{K}_t$ 的行重排、即左乘 $(13)$ 里那个 $\boldsymbol{P}_\pi$，则 $\boldsymbol{K}_t\boldsymbol{K}_t^{\top}$ 变成 $\boldsymbol{P}_\pi(\boldsymbol{K}_t\boldsymbol{K}_t^{\top})\boldsymbol{P}_\pi^{\top}$，$-1/2$ 次方也随之共轭，于是 $\widetilde{\boldsymbol{K}}_t$ 跟着一起重排。

正交化之后 $\widetilde{\boldsymbol{K}}_t^{\top}\widetilde{\boldsymbol{K}}_t=\boldsymbol{\Pi}_t$；value 做同样的变换 $\widetilde{\boldsymbol{V}}_t=(\boldsymbol{K}_t\boldsymbol{K}_t^{\top})^{-1/2}\boldsymbol{V}_t$，则 $\widetilde{\boldsymbol{V}}_t^{\top}\widetilde{\boldsymbol{K}}_t=\boldsymbol{V}_t^{\top}(\boldsymbol{K}_t\boldsymbol{K}_t^{\top})^{-1}\boldsymbol{K}_t$，这和 $(24)$ 的第二项完全一样。所以"key 正交化再跑 $\beta=1$ 的顺序 delta rule"，和一次性解出帧级精确解，是同一个东西。道理也简单：key 一旦真的正交，顺序递推里那些互相修正的项全为零，先后自然不起作用。

## 其他视角

其实同一个损失还有第三种走法。显式和隐式都是离散的一步；既然帧内没有先后，也可以干脆不走离散步，让 state 顺着梯度连续地演化一段时间，也就是梯度流。

先注意一个恒等式。$\boldsymbol{k}$ 是单位向量，所以 $\boldsymbol{k}\boldsymbol{k}^{\top}$ 是个投影：$(\boldsymbol{k}\boldsymbol{k}^{\top})^2=\boldsymbol{k}(\boldsymbol{k}^{\top}\boldsymbol{k})\boldsymbol{k}^{\top}=\boldsymbol{k}\boldsymbol{k}^{\top}$，它的任意正整数次幂都等于自己。于是把矩阵指数按幂级数展开，$m\ge1$ 的每一项都带同一个 $\boldsymbol{k}\boldsymbol{k}^{\top}$，把系数加起来就是 $\sum_{m\ge1}\frac{(-\gamma)^m}{m!}=e^{-\gamma}-1$：

$$
\exp\big(-\gamma\boldsymbol{k}\boldsymbol{k}^{\top}\big)=\boldsymbol{I}+\big(e^{-\gamma}-1\big)\boldsymbol{k}\boldsymbol{k}^{\top}=\boldsymbol{I}-\beta\boldsymbol{k}\boldsymbol{k}^{\top},\qquad \beta=1-e^{-\gamma}\tag{25}
$$

也就是说，delta rule 的擦除矩阵本身就是一个矩阵指数。注意这里的 $\gamma\leftrightarrow\beta$ 换算和 $(20)$ 里那个并不一样，差别从哪来，下一节再说。

于是一帧的顺序连乘可以写成 $\prod_i e^{\boldsymbol{X}_i}$，其中 $\boldsymbol{X}_i=-\gamma_i\boldsymbol{k}_i\boldsymbol{k}_i^{\top}$。由 [BCH 公式](https://en.wikipedia.org/wiki/Baker%E2%80%93Campbell%E2%80%93Hausdorff_formula)，

$$
\prod_i e^{\boldsymbol{X}_i}=\exp\Big(\sum_i\boldsymbol{X}_i+\tfrac12\sum_{i\lt j}\big[\boldsymbol{X}_i,\boldsymbol{X}_j\big]+\cdots\Big)\tag{26}
$$

和 $\exp\big(\sum_i\boldsymbol{X}_i\big)$ 差的正是那些交换子 $[\boldsymbol{X}_i,\boldsymbol{X}_j]$，和 $(7)$ 里的是一回事。既然 $\sum_i\boldsymbol{X}_i$ 与顺序无关，那就直接把 $\exp\big(\sum_i\boldsymbol{X}_i\big)=e^{-\boldsymbol{G}_t}$ 拿来用。

把写入项也考虑进来，就得到一个微分方程：让这一帧占据一段虚拟时间 $\tau\in[0,1]$，帧内所有 token 在这段时间里同时作用，

$$
\frac{\mathrm{d}\boldsymbol{S}}{\mathrm{d}\tau}=-\boldsymbol{S}\boldsymbol{G}_t+\boldsymbol{C}_t,\qquad \boldsymbol{S}(0)=\boldsymbol{S}_{t-1}\tag{27}
$$

右端恰好是 $-\nabla\mathcal{L}_t(\boldsymbol{S})$，所以 $(27)$ 就是帧内总损失的梯度流。由于 $e^{-\boldsymbol{G}_t\tau}$ 与 $\boldsymbol{G}_t$ 可交换，解可以照标量情形写出（对 $\tau$ 求导代回即可验证）：

$$
\boldsymbol{S}_t=\boldsymbol{S}(1)=\boldsymbol{S}_{t-1}\,e^{-\boldsymbol{G}_t}+\boldsymbol{C}_t\,\varphi(\boldsymbol{G}_t),
\qquad \varphi(\boldsymbol{G}_t)=\int_0^1 e^{-\boldsymbol{G}_t u}\,\mathrm{d}u\tag{28}
$$

$\varphi$ 作为矩阵函数就是把每个特征值 $\lambda$ 换成 $(1-e^{-\lambda})/\lambda$，在 $\lambda=0$ 处取值为 1，所以 $\boldsymbol{G}_t$ 奇异时也良定义（等价的幂级数写法是 $\varphi(\boldsymbol{G}_t)=\sum_{m\ge0}\frac{(-\boldsymbol{G}_t)^m}{(m+1)!}$）。它同样只依赖 $\boldsymbol{G}_t,\boldsymbol{C}_t$，所以同样精确置换不变。

如果让这个过程一直跑下去（$\tau\to\infty$）呢？把 $\boldsymbol{G}_t$ 对角化，逐特征值看 $(28)$ 就够了。

先看哪些方向是特殊的。$n\lt d_k$ 时 $\mathrm{rank}\,\boldsymbol{G}_t\le n\lt d_k$，$\boldsymbol{G}_t$ 必有零特征值；只要所有 $\gamma_i\gt0$，由

$$
\boldsymbol{x}^{\top}\boldsymbol{G}_t\boldsymbol{x}=\boldsymbol{x}^{\top}\boldsymbol{K}_t^{\top}\boldsymbol{\Gamma}\boldsymbol{K}_t\boldsymbol{x}=\sum_{i=1}^{n}\gamma_i\big(\boldsymbol{K}_t\boldsymbol{x}\big)_i^2\tag{29}
$$

可知 $\boldsymbol{G}_t\boldsymbol{x}=\boldsymbol{0}$ 当且仅当 $\boldsymbol{K}_t\boldsymbol{x}=\boldsymbol{0}$，也就是零特征值方向恰好是与帧内所有 key 都正交的那些方向。同一个式子还告诉我们 $\boldsymbol{C}_t\boldsymbol{x}=\boldsymbol{V}_t^{\top}\boldsymbol{\Gamma}\boldsymbol{K}_t\boldsymbol{x}=\boldsymbol{0}$，即 $\boldsymbol{C}_t$ 在这些方向上也为零。

现在逐特征值取极限。$\lambda_j\gt0$ 的方向上，$e^{-\lambda_j\tau}\to0$、$\int_0^{\tau}e^{-\lambda_j u}\mathrm{d}u\to1/\lambda_j$，所以旧 state 被清空、换成 $\boldsymbol{C}_t$ 乘 $1/\lambda_j$；$\lambda_j=0$ 的方向上前者恒为 $1$、后者随 $\tau$ 线性发散，但刚才说了 $\boldsymbol{C}_t$ 在这里为零，发散项被乘掉，旧 state 原样留下。把这两类方向合起来写，就是

$$
\boldsymbol{S}_\infty=\boldsymbol{S}_{t-1}\big(\boldsymbol{I}-\boldsymbol{G}_t\boldsymbol{G}_t^{\dagger}\big)+\boldsymbol{C}_t\boldsymbol{G}_t^{\dagger}\tag{30}
$$

其中 $\boldsymbol{G}_t^{\dagger}$ 是 [Moore–Penrose 伪逆](https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse)：它在 $\lambda_j\gt0$ 的方向上取 $1/\lambda_j$、在 $\lambda_j=0$ 的方向上取 $0$，正是上面两句话的矩阵写法；$\boldsymbol{I}-\boldsymbol{G}_t\boldsymbol{G}_t^{\dagger}$ 则是到零特征值方向的投影。在 $\boldsymbol{G}_t$ 作用到的那些方向上，$\boldsymbol{S}_\infty$ 满足 $\boldsymbol{S}\boldsymbol{G}_t=\boldsymbol{C}_t$，也就是帧级损失的加权最小二乘解。若再补上上一节 $\boldsymbol{K}_t$ 行满秩的条件，它与那里的投影解完全一致。

顺带一提，把 delta rule 看成一条 ODE 的显式 Euler 离散、再去解它的精确流，这个思路和 [Exact Flow Linear Attention](https://arxiv.org/abs/2512.12602) 有点像，那篇也是靠 rank-1 结构让矩阵指数和积分项退化成简单形式。区别在于它处理的是单个 token，擦除矩阵是 rank-1 的；而这里的 $\boldsymbol{G}_t$ 是整帧汇总出来的，一般满秩，$\varphi(\boldsymbol{G}_t)$ 不会退化。

## 三者关系

把问题缩到一维就能看清三者的关系。设 $\boldsymbol{G}_t$ 在某个特征方向上的特征值是 $\lambda$，看一帧过后旧 state 在这个方向上还剩多少：显式剩 $1-\lambda$，隐式剩 $\frac{1}{1+\lambda}$，梯度流剩 $e^{-\lambda}$。

三个数摆在一起，两件事立刻清楚。第一，显式那个在 $\lambda\gt1$ 时会变负、$\lambda\gt2$ 时绝对值超过 1，另两个不管 $\lambda$ 多大都老老实实待在 $0$ 到 $1$ 之间。第二，前面 $(20)$ 和 $(25)$ 各出现过一次 $\gamma$ 与 $\beta$ 的换算，为什么对不上，来源就在这里：要让剩下的比例等于 delta rule 的 $1-\beta$，隐式解出 $\gamma=\beta/(1-\beta)$，梯度流解出 $\gamma=-\ln(1-\beta)$。同一个 $\beta$ 在两种走法里对应不同的 $\gamma$，两处换算自然不同。

有几点要交代清楚：上面比的是单个方向、单独一步。GDN 本身并不是对整帧损失走一次显式步，不计遗忘门时，它一帧下来是 $\prod_i(\boldsymbol{I}-\beta_i\boldsymbol{k}_i\boldsymbol{k}_i^{\top})$，$n$ 个擦除矩阵按顺序乘起来，这才是它认顺序的原因。另外，隐式和梯度流也不是一回事，只在写入极弱和极强两端才重合，超参不能混用。

三种走法其实是同一个式子的三个特例：

$$
\boldsymbol{S}_t=\boldsymbol{S}_{t-1}\,f(\boldsymbol{G}_t)+\boldsymbol{C}_t\,g(\boldsymbol{G}_t)\tag{31}
$$

这里 $f,g$ 的作用就是把 $\boldsymbol{G}_t$ 的每个特征值 $\lambda$ 换成 $f(\lambda)$ 和 $g(\lambda)$：显式取 $f=1-\lambda$、$g=1$；隐式取 $f=g=\frac{1}{1+\lambda}$；梯度流取 $f=e^{-\lambda}$、$g=\frac{1-e^{-\lambda}}{\lambda}$。可见置换不变完全来自 $\boldsymbol{G}_t,\boldsymbol{C}_t$ 是求和，$f,g$ 只决定怎么更新。

换一对 $f,g$ 就是一种新走法。比如把梯度流拆成 $m$ 小步，每步只走 $1/m$：重复 $m$ 次 $\boldsymbol{S}\leftarrow\boldsymbol{S}(\boldsymbol{I}-\boldsymbol{G}_t/m)+\boldsymbol{C}_t/m$，剩下的比例是 $(1-\lambda/m)^m$，$m$ 越大越接近 $e^{-\lambda}$。只要 $m$ 大于 $\lambda_{\max}/2$，每小步都不会放大旧 state，全程只有矩阵乘法，不用做分解；代价是反向要沿这 $m$ 步回放。它也置换不变，因为每步只用到 $\boldsymbol{G}_t$ 和 $\boldsymbol{C}_t$。反过来说，如果改成把帧内 token 分成几组依次乘进去，顺序依赖就又回来了。

最后说说代价。逐 token 依次擦写本身是有用的：$\beta_i=1$、无遗忘门时，$n$ 个 token 依次作用，相当于依次把 state 投影到 $\boldsymbol{S}\boldsymbol{k}_i=\boldsymbol{v}_i$ 这些约束上，也就是 [Kaczmarz 迭代](https://en.wikipedia.org/wiki/Kaczmarz_method)；后面的 token 能修正前面写坏的地方。[GLA](https://arxiv.org/abs/2312.06635)、[Mamba2](https://arxiv.org/abs/2405.21060) 虽然也是逐 token 递推，但写入不读当前 state，只是累加加整体衰减，没有这个修正机制。这种串行修正能力，可能也是 [DeltaNet 类模型](https://arxiv.org/abs/2406.06484)在一些实验里强于 GLA、Mamba2 的原因之一。帧级整体更新把帧内样本对称地合并掉，这个机制就没有了。所以这是取舍，不是改进。对视频的帧内维度，笔者认为值得换：那点串行深度是沿着一条我们自己编出来的扫描线走的，而跨帧的串行深度完整保留，那才是视频里真实存在的因果。

## 简单总结

本文先把"视频原生的 delta rule"该满足的性质写清楚：帧内换一种扫描顺序，传给下一帧的 state 和当帧的输出都不该变。然后顺着 online learning 的视角推出帧级更新的一般形状，比较了显式、隐式、梯度流三种走法。其中隐式那一步能把写入强度从稳定性的约束里解放出来，代价是每帧多做一次小矩阵分解。

最后说明一下：这些只是笔者出于 "research for fun and truth" 的理念推导的一点想法，实际效果并没有验证过。甚至"视频的 DeltaNet 到底需不需要帧内置换不变"这个问题本身都没有答案，它看着是个自然的先验，但网络自己学出别的解法也完全可能。权当一次笔者的纸上谈兵吧。

## 参考链接

- [Qwen3.5](https://qwen.ai/blog?id=qwen3.5)
- [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464)
- [Kimi K3 Technical Report](https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf)
- [Kimi Linear: An Expressive, Efficient Attention Architecture (KDA)](https://arxiv.org/abs/2510.26692)
- [Chimera: Designing and Chinchilla-Scaling Hybrid Visual Diffusion Transformers](https://papers.cool/arxiv/2607.28611)
- [SANA-WM: Efficient Minute-Scale World Modeling with Hybrid Linear Diffusion Transformer](https://arxiv.org/abs/2605.15178)
- [Exact Flow Linear Attention: Exact Solution from Continuous-Time Dynamics](https://arxiv.org/abs/2512.12602)
- [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/abs/2006.16236)
- [Linear Transformers Are Secretly Fast Weight Programmers](https://arxiv.org/abs/2102.11174)
- [Learning to (Learn at Test Time): RNNs with Expressive Hidden States (TTT)](https://arxiv.org/abs/2407.04620)
- [苏剑林：线性注意力简史——从模仿、创新到反哺](https://kexue.fm/archives/11033)
- [Parallelizing Linear Transformers with the Delta Rule over Sequence Length](https://arxiv.org/abs/2406.06484)
- [Gated Linear Attention Transformers with Hardware-Efficient Training (GLA)](https://arxiv.org/abs/2312.06635)
- [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality (Mamba2)](https://arxiv.org/abs/2405.21060)

---
title: "Diffusion 学习笔记：ODE 与 SDE 何时等价？"
title_en: "Diffusion Notes: When Are ODE and SDE Sampling Equivalent?"
date: 2026-09-09
draft: false
---

笔者最近在学习 diffusion / flow matching 时，对 ODE 和 SDE 采样之间的关系有了一些新的理解，在这里记录一下。

一个有意思的观察是，在理想条件下，同一个模型既能用 ODE 采样，也能用 SDE 采样，两者可以有相同的边缘分布。只看预训练代码里的预测目标，往往还无法判断最后会用哪一种采样器。

比如，FM 训练的是速度场，最直接的用法就是求解 ODE。那么，加入随机噪声后，为什么还能得到相同的分布？这种等价又依赖哪些条件？

## 训练目标

先看 [Flow Matching](https://arxiv.org/abs/2210.02747) 中最常见的线性插值。全文约定 $t=0$ 是数据端，$t=1$ 是噪声端；文本等条件先固定，省略在记号里。数据 $X_0$ 与独立的标准高斯噪声 $\varepsilon$ 之间有

$$
\begin{aligned}
X_t&=(1-t)X_0+t\varepsilon,\qquad
\varepsilon\sim\mathcal N(0,I),\quad \varepsilon\perp X_0,\\
v_t(x)&=\mathbb E[\varepsilon-X_0\mid X_t=x].
\end{aligned} \tag{1}
$$

训练时，用 MSE 回归 $\varepsilon-X_0$，最优预测就是上式的条件平均速度。它定义了 ODE $\mathrm{d}X_t=v_t(X_t)\,\mathrm{d}t$。在适当的正则条件下，从正确的初始分布出发并精确求解，这个 ODE 在每个时刻的分布都与 $(1)$ 中的插值分布相同，但轨迹未必是直线。

构造对应的 SDE 还需要 score。记 $p_t$ 为 $X_t$ 的密度，$s_t(x)=\nabla_x\log p_t(x)$。对一般的独立高斯加噪 $X_t=\alpha_tX_0+\sigma_t\varepsilon$，在 $\sigma_t>0$ 时有

$$
\begin{aligned}
\nabla_x\log p_t(x)
&=\mathbb E\big[\nabla_x\log p_t(x\mid X_0)\mid X_t=x\big]\\
&=-\frac{1}{\sigma_t}\mathbb E[\varepsilon\mid X_t=x].
\end{aligned} \tag{2}
$$

因此，noise-MSE 学到的条件均值乘上 $-1/\sigma_t$，就得到 score。回到 $(1)$ 的线性插值，在 $0\lt t\lt 1$ 时：

$$
\begin{aligned}
\hat x_0&=x-tv_t(x),\\
\hat\varepsilon&=x+(1-t)v_t(x),\\
s_t(x)&=-\frac{\hat\varepsilon}{t}
=-\frac{x+(1-t)v_t(x)}{t}.
\end{aligned} \tag{3}
$$

其中 $\hat x_0=\mathbb E[X_0\mid X_t=x]$、$\hat\varepsilon=\mathbb E[\varepsilon\mid X_t=x]$。也就是说，虽然 FM 的训练目标是速度，但在这个设定下，同一个输出也能给出噪声、干净样本和 score 的最优预测。

## 采样等价

有了速度和 score，就可以按照 [Score-based SDE](https://arxiv.org/abs/2011.13456) 中的对应关系构造随机采样。设 ODE 的速度为 $v_t$，它的密度 $p_t$ 满足连续性方程：

$$
\partial_t p_t=-\nabla\cdot(p_tv_t).
$$

先沿时间递增的方向考虑。任取只依赖时间的 $\lambda_t\ge0$，构造

$$
\mathrm{d}X_t=
\big[v_t(X_t)+\lambda_t\nabla\log p_t(X_t)\big]\,\mathrm{d}t
+\sqrt{2\lambda_t}\,\mathrm{d}W_t. \tag{4}
$$

其中 $W_t$ 是标准布朗运动。与原来的 ODE 相比，我们一边加入随机噪声，一边给漂移补上 $\lambda_ts_t$。这两项对分布的影响恰好可以抵消。

用 [Fokker–Planck 方程](https://math.nyu.edu/~goodman/teaching/StochCalc2022/materials/Section2.pdf#page=1)（描述 SDE 的概率密度如何随时间变化）写出来，就是

$$
\begin{aligned}
\partial_t p_t
&=-\nabla\cdot(p_tv_t)
-\lambda_t\nabla\cdot(p_t\nabla\log p_t)
+\lambda_t\Delta p_t\\
&=-\nabla\cdot(p_tv_t).
\end{aligned} \tag{5}
$$

因为 $p_t\nabla\log p_t=\nabla p_t$，新增的两项抵消，又得到了原来的连续性方程。在适当的正则条件下，从相同初始分布出发并精确求解，ODE 与这个 SDE 的边缘分布相同。

实际生成时，我们从 $t=1$ 积分到 $t=0$。此时对应的反向 SDE 为

$$
\mathrm{d}X_t=
\big[v_t(X_t)-\lambda_ts_t(X_t)\big]\,\mathrm{d}t
+\sqrt{2\lambda_t}\,\mathrm{d}\bar W_t,\qquad \mathrm{d}t<0, \tag{6}
$$

其中 $\bar W_t$ 是反向时间的布朗运动。取 $\lambda_t=0$ 就回到 ODE；取正值则会在途中加入随机性，同时调整漂移。只要速度和 score 都准确、从正确的 $p_1$ 出发并精确求解，两者就可以在每个时刻具有相同的 $p_t$。这也解释了为什么同一个预训练网络可以配合不同的采样器使用。

这里的“相同”只指每个时刻的边缘分布。给定初始状态后，ODE 的轨迹唯一确定，SDE 仍会引入新的随机性；两者并不需要沿着同一条轨迹生成样本。

## 实际误差

上面的抵消依赖真实 score。实际情况里网络只能近似它：即使 $v_t$ 精确，若预测的 score 为 $s_\theta=s_t+e_t$，$(5)$ 中仍会多出

$$
-\lambda_t\nabla\cdot(p_te_t). \tag{7}
$$

这一项通常不为零，原来的 $p_t$ 便不再同时满足两种演化方程。即使速度和 score 由同一个网络按 $(3)$ 换算得到，也不能保证它们与 ODE 实际产生的密度满足 $(5)$ 的关系。

即便模型完全精确，有限步求解仍会带来差别。举个简单例子：取 $p_t$ 为一维标准高斯、$v_t=0$，ODE 保持状态不变；固定 $\lambda>0$，$(4)$ 变成

$$
\mathrm{d}X_t=-\lambda X_t\,\mathrm{d}t+\sqrt{2\lambda}\,\mathrm{d}W_t.
$$

这个 SDE 同样保持标准高斯分布。但用 [Euler–Maruyama 方法](https://epubs.siam.org/doi/10.1137/S0036144500378302)（SDE 的显式 Euler 离散化）走一步，步长为 $h$，得到

$$
X'=(1-\lambda h)X+\sqrt{2\lambda h}\,\xi,
\qquad X,\xi\overset{\mathrm{iid}}\sim\mathcal N(0,1).
$$

一步之后的方差就变成了

$$
\operatorname{Var}(X')=(1-\lambda h)^2+2\lambda h
=1+\lambda^2h^2.
$$

当然，这个例子并不意味着随机采样总是更差。[EDM](https://arxiv.org/abs/2206.00364) 将随机采样解释为 ODE 加 Langevin 修正，用来减小此前积累的分布误差；但过量加噪、去噪也会损失细节。因此，真实模型和有限步采样的效果，需要结合具体误差来判断，不能只凭理想情况下的等价性决定。

## 高斯之外

前面还有一个条件需要单独看：$(2)(3)$ 的换算依赖独立高斯加噪。我们不妨保留独立性，只把噪声换成一维 Laplace 分布。设 $X_t=\alpha_tX_0+\sigma_t\varepsilon$，$\sigma_t>0$，且 $\varepsilon\sim\mathrm{Laplace}(0,1)$ 与 $X_0$ 独立。此时几乎处处有

$$
s_t(x)=-\frac{1}{\sigma_t}
\mathbb E[\operatorname{sign}(\varepsilon)\mid X_t=x]. \tag{8}
$$

noise-MSE 学到的仍是 $\mathbb E[\varepsilon\mid X_t=x]$。均值相同并不意味着取正值和负值的概率相同，所以这个均值一般不足以恢复 score。也就是说，换成 Laplace 噪声后，FM 仍然可以学习 ODE 的速度场，但不能再按 $(3)$ 直接换算出 SDE 所需的 score。

当然，有读者可能会说，实际应用里大家用的几乎都是高斯噪声。那我们再看一个更实际的例子：[Waver](https://arxiv.org/html/2508.15761v1#S2.SS2) 的 noise blend refiner。它将退化后的低分辨率 latent 与高斯噪声混合作为 source，再与干净 latent 插值：

$$
\begin{aligned}
Y&=(1-w_d)X_{\mathrm{lr}}+w_dN,\qquad N\sim\mathcal N(0,I),\\
X_t&=(1-t)X_0+tY.
\end{aligned} \tag{9}
$$

Waver 原文训练时在 $[0.85,0.95]$ 内随机采样 $w_d$，回归目标为 $X_0-Y$。下面分析固定 $0\lt w_d\lt 1$ 训练的同类 refiner，且 $X_{\mathrm{lr}}$ 只用于构造 source、不额外输入网络。沿用本文的时间方向，最优速度记为 $v_t(x)=\mathbb E[Y-X_0\mid X_t=x]$，与原文的预测方向相反。由于 $N$ 是与 $(X_0,X_{\mathrm{lr}})$ 独立的高斯噪声，沿用 $(2)$ 的求导方法，在 $0\lt t\lt 1$ 时有

$$
s_t(x)=-\frac{x+(1-t)v_t(x)-(1-w_d)\mathbb E[X_{\mathrm{lr}}\mid X_t=x]}{tw_d^2}. \tag{10}
$$

速度给出的是整个 source 的条件均值，score 的换算还需要其中低清分量的条件均值。固定混合系数并不能消去这一项，因此一般不能仅凭速度直接得到 score。若能另外得到真实 score，$(4)$ 中 ODE/SDE 的对应关系仍然成立。

## 蒸馏偏差

如果用这种 refiner 做 [DMD](https://arxiv.org/html/2311.18828#S3.SS2) / [DMD2](https://arxiv.org/html/2405.14867v2#S3) 蒸馏，问题就不只是采样器了：KL 梯度本身就需要 teacher 的 score。直接使用速度差，会产生什么误差？这要看 loss 怎样给学生输出重新加噪。

记 $P_t,Q_{\theta,t}$ 为目标与学生输出经过相应加噪后的分布，$J=\partial G_\theta/\partial\theta$。下面固定 $t$，假设 loss 中的高斯噪声独立重采样、ref 不随 $\theta$ 变化，只分析理想网络下的分布匹配项，略去 CFG、GAN 和额外的梯度归一化。以下公式中的 score、速度和后验均值都在学生的加噪样本处求值，期望按学生的联合采样过程计算。此时

$$
g_t=\nabla_\theta\mathrm{KL}(Q_{\theta,t}\|P_t)
=(1-t)\mathbb E\big[J^\top(s_Q-s_P)\big]. \tag{11}
$$

**第一种，loss 也沿用 noise blend。** 假设 fake 同样按这条路径训练，记 $m_P(x)=\mathbb E_P[X_{\mathrm{lr}}\mid X_t=x]$，$m_Q$ 同理。由 $(10)$ 相减，若只取速度差、补偿已知的 $w_d^2$ 缩放，所得更新 $\hat g_t$ 与真实梯度之间有

$$
\begin{aligned}
s_Q-s_P
&=-\frac{1-t}{tw_d^2}(v_Q-v_P)
+\frac{1-w_d}{tw_d^2}(m_Q-m_P),\\
\hat g_t-g_t
&=-\frac{(1-t)(1-w_d)}{tw_d^2}
\mathbb E\big[J^\top(m_Q-m_P)\big].
\end{aligned} \tag{12}
$$

遗漏的是两侧对 ref 的条件均值之差。它们足够接近时，这个近似可能有效；但复用同一个 ref，并不保证两个模型对它的后验均值相同。

**第二种，loss 改用纯高斯加噪** $X_t=(1-t)X+tN$。fake 可以通过相应的去噪训练学习 score，[DMD2 官方实现](https://github.com/tianweiy/DMD2/blob/8d8fa55633d47cfb81bbc7a892e7248f9518763f/main/sd_guidance.py#L257-L297)中的 fake model 就采用独立高斯加噪监督。但 teacher 仍然预测 noise blend 路径上的速度 $v_P$。如果直接按 $(3)$ 换算，记得到的预测为 $\tilde s_P^G$、真实的高斯加噪 score 为 $s_P^G$，则在 fake 精确时

$$
\begin{aligned}
\tilde s_P^G(x)&=-\frac{x+(1-t)v_P(x)}{t},
\qquad e_P(x)=\tilde s_P^G(x)-s_P^G(x),\\
\hat g_t-g_t&=-(1-t)\mathbb E\big[J^\top e_P\big].
\end{aligned} \tag{13}
$$

这里的误差来自把 noise blend 训练出的 teacher 当成普通高斯去噪模型：查询的加噪方式和 score 换算都没有对齐。即使 fake 学得准确，也不能自动补上 teacher 的偏差。

因此，这类用 FM 训练的 noise blend refiner，直接在 DMD 中把速度换成 score，一般会引入偏差。实践中仍可以把它当作一种近似来做蒸馏，也可能得到不错的结果；只是这种近似可能改变优化方向，最终学到的分布未必对应原本 KL 目标的最优解。

## 总结一下

本文讨论了在独立高斯加噪的设定下，同一个模型可以提供速度与 score，让 ODE/SDE 在理想条件下具有相同的边缘分布；实际效果仍受模型和求解误差影响。换成不符合这些条件的应用场景，比如用 FM 训练的 noise blend refiner，直接用速度转换 score 做 DMD 蒸馏一般会引入偏差。

## 参考链接

- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456)
- [Elucidating the Design Space of Diffusion-Based Generative Models (EDM)](https://arxiv.org/abs/2206.00364)
- [One-step Diffusion with Distribution Matching Distillation (DMD)](https://arxiv.org/abs/2311.18828)
- [Improved Distribution Matching Distillation for Fast Image Synthesis (DMD2)](https://arxiv.org/abs/2405.14867)
- [DMD2 官方代码](https://github.com/tianweiy/DMD2)
- [Waver: Wave Your Way to Lifelike Video Generation](https://arxiv.org/abs/2508.15761)

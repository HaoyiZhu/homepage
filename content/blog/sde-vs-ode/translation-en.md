I've recently been studying diffusion / flow matching and wanted to write down a few things I've learned about ODE and SDE sampling.

One interesting observation is that, under ideal conditions, the same model can be used with either ODE or SDE sampling, and the two can share the same marginal distributions. The prediction target in the pretraining code often doesn't tell you which sampler will be used.

For example, FM learns a velocity field, and the most direct way to use it is to solve the ODE. So why does adding random noise still give the same distribution? And what conditions does this equivalence depend on?

## Training Targets

Start with the most common linear interpolation in [Flow Matching](https://arxiv.org/abs/2210.02747). Throughout, $t=0$ is the data end and $t=1$ is the noise end; conditions such as text are held fixed and omitted from the notation. Linearly interpolating between the data $X_0$ and independent standard Gaussian noise $\varepsilon$ gives

$$
\begin{aligned}
X_t&=(1-t)X_0+t\varepsilon,\qquad
\varepsilon\sim\mathcal N(0,I),\quad \varepsilon\perp X_0,\\
v_t(x)&=\mathbb E[\varepsilon-X_0\mid X_t=x].
\end{aligned} \tag{1}
$$

We train with MSE using $\varepsilon-X_0$ as the target, so the optimal prediction is the conditional mean velocity above. It defines the ODE $\mathrm{d}X_t=v_t(X_t)\,\mathrm{d}t$. Under suitable regularity conditions, an exact solution starting from the correct initial distribution has the same marginal distributions as the interpolation in $(1)$, though the trajectories need not be straight lines.

Constructing the corresponding SDE also requires the score. Let $p_t$ be the density of $X_t$ and $s_t(x)=\nabla_x\log p_t(x)$. For general independent Gaussian noising $X_t=\alpha_tX_0+\sigma_t\varepsilon$, when $\sigma_t>0$,

$$
\begin{aligned}
\nabla_x\log p_t(x)
&=\mathbb E\big[\nabla_x\log p_t(x\mid X_0)\mid X_t=x\big]\\
&=-\frac{1}{\sigma_t}\mathbb E[\varepsilon\mid X_t=x].
\end{aligned} \tag{2}
$$

So multiplying the conditional mean learned by noise-MSE by $-1/\sigma_t$ gives the score. Back to the linear interpolation in $(1)$, for $0\lt t\lt 1$:

$$
\begin{aligned}
\hat x_0&=x-tv_t(x),\\
\hat\varepsilon&=x+(1-t)v_t(x),\\
s_t(x)&=-\frac{\hat\varepsilon}{t}
=-\frac{x+(1-t)v_t(x)}{t}.
\end{aligned} \tag{3}
$$

Here $\hat x_0=\mathbb E[X_0\mid X_t=x]$ and $\hat\varepsilon=\mathbb E[\varepsilon\mid X_t=x]$. In other words, although FM's training target is velocity, under this setup the same output can also give optimal predictions of the noise, the clean sample, and the score.

## Equivalent Sampling

Given the velocity and score, we can construct an SDE using the correspondence in [Score-based SDE](https://arxiv.org/abs/2011.13456). Let the ODE velocity be $v_t$; the density $p_t$ of the samples satisfies the continuity equation:

$$
\partial_t p_t=-\nabla\cdot(p_tv_t).
$$

Consider first the direction of increasing time. For arbitrary $\lambda_t\ge0$ that depends only on time, construct

$$
\mathrm{d}X_t=
\big[v_t(X_t)+\lambda_t\nabla\log p_t(X_t)\big]\,\mathrm{d}t
+\sqrt{2\lambda_t}\,\mathrm{d}W_t. \tag{4}
$$

Here $W_t$ is standard Brownian motion. Compared with the original ODE, we add random noise and adjust the drift by $\lambda_ts_t$. The effects of these two terms on the distribution cancel exactly.

Written out with the [Fokker–Planck equation](https://math.nyu.edu/~goodman/teaching/StochCalc2022/materials/Section2.pdf#page=1) (which describes how the probability density of an SDE changes over time), this becomes

$$
\begin{aligned}
\partial_t p_t
&=-\nabla\cdot(p_tv_t)
-\lambda_t\nabla\cdot(p_t\nabla\log p_t)
+\lambda_t\Delta p_t\\
&=-\nabla\cdot(p_tv_t).
\end{aligned} \tag{5}
$$

Since $p_t\nabla\log p_t=\nabla p_t$, the two added terms cancel, and we recover the original continuity equation. Under suitable regularity conditions, starting from the same initial distribution and solving exactly, the ODE and this SDE have the same marginals.

In actual generation, we integrate from $t=1$ to $t=0$. The corresponding reverse SDE is

$$
\mathrm{d}X_t=
\big[v_t(X_t)-\lambda_ts_t(X_t)\big]\,\mathrm{d}t
+\sqrt{2\lambda_t}\,\mathrm{d}\bar W_t,\qquad \mathrm{d}t<0, \tag{6}
$$

where $\bar W_t$ is a reverse-time Brownian motion. Setting $\lambda_t=0$ returns the ODE; a positive value injects randomness along the way while adjusting the drift. With accurate velocity and score, the correct initial distribution $p_1$, and exact integration, the two can have the same $p_t$ at every time. This also explains why the same pretrained network can be used with different samplers.

Here "same" refers only to the marginal distribution at each time. Given an initial state, the ODE trajectory is uniquely determined, while the SDE still introduces new randomness; the two need not generate samples along the same trajectory.

## Errors in Practice

The cancellation above relies on the true score. A real network can only approximate it: even with exact $v_t$, a predicted score of $s_\theta=s_t+e_t$ introduces an extra term in $(5)$:

$$
-\lambda_t\nabla\cdot(p_te_t). \tag{7}
$$

This term is generally nonzero, so the original $p_t$ no longer satisfies both evolution equations at once. Even if the velocity and score are converted from the same network via $(3)$, they are not guaranteed to satisfy the relation in $(5)$ with the density the ODE actually produces.

Even with an exact model, numerical solvers can produce different distributions in a finite number of steps. A simple example: take $p_t$ to be a one-dimensional standard Gaussian and $v_t=0$, so the ODE keeps the state unchanged; fix $\lambda>0$, and $(4)$ becomes

$$
\mathrm{d}X_t=-\lambda X_t\,\mathrm{d}t+\sqrt{2\lambda}\,\mathrm{d}W_t.
$$

This SDE also preserves the standard Gaussian distribution. But taking one step with the [Euler–Maruyama method](https://epubs.siam.org/doi/10.1137/S0036144500378302) (the explicit Euler discretization of an SDE), with step size $h$, gives

$$
X'=(1-\lambda h)X+\sqrt{2\lambda h}\,\xi,
\qquad X,\xi\overset{\mathrm{iid}}\sim\mathcal N(0,1).
$$

After one step, the variance becomes

$$
\operatorname{Var}(X')=(1-\lambda h)^2+2\lambda h
=1+\lambda^2h^2.
$$

Of course, this example does not mean stochastic sampling is always worse. [EDM](https://arxiv.org/abs/2206.00364) interprets stochastic sampling as an ODE plus a Langevin correction that reduces accumulated distribution error. But excessive noising and denoising can also erase detail. Sampling performance therefore depends on model and solver errors; equivalence under ideal conditions alone does not settle the question.

## Beyond Gaussian Noise

There is one more condition above that deserves a separate look: the conversion in $(2)(3)$ relies on independent Gaussian noising. Let's keep independence and replace only the noise with a one-dimensional Laplace distribution. Let $X_t=\alpha_tX_0+\sigma_t\varepsilon$, $\sigma_t>0$, with $\varepsilon\sim\mathrm{Laplace}(0,1)$ independent of $X_0$. Then almost everywhere,

$$
s_t(x)=-\frac{1}{\sigma_t}
\mathbb E[\operatorname{sign}(\varepsilon)\mid X_t=x]. \tag{8}
$$

The noise-MSE objective still learns $\mathbb E[\varepsilon\mid X_t=x]$. The same mean does not imply the same probabilities of taking positive and negative values, so this mean is generally not enough to recover the score. In other words, after switching to Laplace noise, FM can still learn the ODE velocity field, but it can no longer be converted directly via $(3)$ into the score the SDE needs.

Of course, a reader might say that in real applications almost everyone uses Gaussian noise. Then let's look at a more practical example: the noise blend refiner in [Waver](https://arxiv.org/html/2508.15761v1#S2.SS2). It mixes the degraded low-resolution latent with Gaussian noise as the source, then interpolates with the clean latent:

$$
\begin{aligned}
Y&=(1-w_d)X_{\mathrm{lr}}+w_dN,\qquad N\sim\mathcal N(0,I),\\
X_t&=(1-t)X_0+tY.
\end{aligned} \tag{9}
$$

Waver samples $w_d$ from $[0.85,0.95]$ during training, with regression target $X_0-Y$. Below I consider a refiner of the same kind trained with a fixed mixing coefficient $0\lt w_d\lt 1$, assuming that $X_{\mathrm{lr}}$ is used only to construct the source and is not fed separately to the network. With this post's time convention, the optimal velocity is $v_t(x)=\mathbb E[Y-X_0\mid X_t=x]$, opposite to the prediction direction in the original paper. Since $N$ is Gaussian noise independent of $(X_0,X_{\mathrm{lr}})$, the same derivation as in $(2)$ gives, for $0\lt t\lt 1$,

$$
s_t(x)=-\frac{x+(1-t)v_t(x)-(1-w_d)\mathbb E[X_{\mathrm{lr}}\mid X_t=x]}{tw_d^2}. \tag{10}
$$

The velocity gives the conditional mean of the entire source, while the score conversion also needs the conditional mean of the low-resolution component within it. A fixed mixing coefficient cannot eliminate this term, so the score generally cannot be obtained from the velocity alone. If the true score is available by other means, the ODE/SDE correspondence in $(4)$ still holds.

## Bias in Distillation

If this kind of refiner is used for [DMD](https://arxiv.org/html/2311.18828#S3.SS2) / [DMD2](https://arxiv.org/html/2405.14867v2#S3) distillation, the problem is no longer just the sampler: the KL gradient itself needs the teacher's score. What error does using the velocity difference directly introduce? That depends on how the loss re-noises the student output.

Let $P_t,Q_{\theta,t}$ be the distributions of the target and student outputs after the corresponding noising, and $J=\partial G_\theta/\partial\theta$. Fix $t$ and assume that the Gaussian noise in the loss is resampled independently and the reference latent does not change with $\theta$. We consider only the distribution-matching term with ideal networks, leaving out CFG, GAN, and extra gradient normalization. In the formulas below, the score, velocity, and posterior mean are all evaluated at the student's noised samples, and expectations are computed over the student's joint sampling process. Then

$$
g_t=\nabla_\theta\mathrm{KL}(Q_{\theta,t}\|P_t)
=(1-t)\mathbb E\big[J^\top(s_Q-s_P)\big]. \tag{11}
$$

**Case one: the loss also uses the noise blend.** Suppose the fake model is also trained along this path; write $m_P(x)=\mathbb E_P[X_{\mathrm{lr}}\mid X_t=x]$, and define $m_Q$ similarly. Using $(10)$, if we keep only the velocity difference and compensate for the known $w_d^2$ scaling, the resulting update $\hat g_t$ differs from the true gradient as follows:

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

The missing term is the difference in the reference latent's conditional mean under the target and student distributions. When these means are close enough, the approximation may work well. But reusing the same reference latent does not guarantee that the two models have the same posterior mean for it.

**Case two: the loss switches to pure Gaussian noising** $X_t=(1-t)X+tN$. The fake model can learn the score through the corresponding denoising training, and the fake model in the [official DMD2 implementation](https://github.com/tianweiy/DMD2/blob/8d8fa55633d47cfb81bbc7a892e7248f9518763f/main/sd_guidance.py#L257-L297) is supervised with independent Gaussian noising. But the teacher still predicts the velocity $v_P$ along the noise blend path. If we convert directly via $(3)$, denoting the resulting prediction as $\tilde s_P^G$ and the true Gaussian-noised score as $s_P^G$, then when the fake model is exact,

$$
\begin{aligned}
\tilde s_P^G(x)&=-\frac{x+(1-t)v_P(x)}{t},
\qquad e_P(x)=\tilde s_P^G(x)-s_P^G(x),\\
\hat g_t-g_t&=-(1-t)\mathbb E\big[J^\top e_P\big].
\end{aligned} \tag{13}
$$

The teacher was trained along the noise blend path, but when computing the DMD loss, we feed it student samples with pure Gaussian noise added, then convert its predicted velocity into a score using the formula for pure Gaussian noising. These two mismatches introduce the bias. Even if the fake model is accurate, it cannot automatically compensate for the teacher's bias.

So for this kind of FM-trained noise blend refiner, directly converting velocity into score in DMD generally introduces bias. In practice it can still be used as an approximation for distillation and may give good results. But the approximation may change the optimization direction, and the distribution ultimately learned need not correspond to the optimum of the original KL objective.

## Wrapping Up

This post discussed how, under independent Gaussian noising, the same model can provide velocity and score, letting ODE/SDE share the same marginals under ideal conditions; actual results are still affected by model and solver errors. In application settings that do not satisfy these conditions, such as an FM-trained noise blend refiner, converting velocity directly into score for DMD distillation generally introduces bias.

## References

- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456)
- [Elucidating the Design Space of Diffusion-Based Generative Models (EDM)](https://arxiv.org/abs/2206.00364)
- [One-step Diffusion with Distribution Matching Distillation (DMD)](https://arxiv.org/abs/2311.18828)
- [Improved Distribution Matching Distillation for Fast Image Synthesis (DMD2)](https://arxiv.org/abs/2405.14867)
- [DMD2 official code](https://github.com/tianweiy/DMD2)
- [Waver: Wave Your Way to Lifelike Video Generation](https://arxiv.org/abs/2508.15761)

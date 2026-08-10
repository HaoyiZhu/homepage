Since [Qwen3.5](https://qwen.ai/blog?id=qwen3.5) adopted [Gated DeltaNet](https://arxiv.org/abs/2412.06464), and more recently [Kimi K3](https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf) adopted [KDA](https://arxiv.org/abs/2510.26692), this family of delta-rule-based linear attention has attracted a lot of attention. But its state is passed down token by token, which is natural for language and not necessarily for video: language tokens come with a built-in order, and while video is causal in time, the $n$ tokens inside one frame have no inherent order at all — who comes first depends entirely on the scan we happen to choose: raster, snake, Hilbert curve, you name it.

Of course, flattening video tokens and feeding them straight into KDA does train — [Chimera](https://papers.cool/arxiv/2607.28611) does exactly that, and multimodal LLMs likewise encode visual inputs with a bidirectional ViT and then feed the features together with text tokens into the backbone causally. But I have always felt that a video-native operator — one whose result is completely unchanged when the intra-frame scan order changes — would be more elegant, and more interesting, even if it is not necessarily better.

My own earlier work [SANA-WM](https://arxiv.org/abs/2605.15178) took a step in this direction, changing GDN from token-by-token to frame-by-frame. Back then, though, my understanding of linear attention was still shallow, and the later experimental results were not great either. Recently I revisited this problem and gained some new insights, which I record here.

> Note: the mathematical derivations in this post were worked out partly in discussion with Claude.

## Preliminaries

Let me fix the notation first. All vectors are column vectors; the query/key dimension is $d_k$ and the value dimension is $d_v$; the state is $\boldsymbol{S}\in\mathbb{R}^{d_v\times d_k}$. When recalling the delta rule, $t$ indexes tokens; later on $t$ will index frames, the $n$ tokens within a frame are indexed by $i$, and the keys and values of frame $t$ are stacked row-wise into $\boldsymbol{K}_t\in\mathbb{R}^{n\times d_k}$ and $\boldsymbol{V}_t\in\mathbb{R}^{n\times d_v}$. By convention $\boldsymbol{k}$ is L2-normalized, i.e. $\boldsymbol{k}^{\top}\boldsymbol{k}=1$.

Linear attention compresses the history KV into a fixed-size state $\boldsymbol{S}_t$, and given a query $\boldsymbol{q}_t$ outputs $\boldsymbol{o}_t=\boldsymbol{S}_t\boldsymbol{q}_t$. [The simplest update rule](https://arxiv.org/abs/2006.16236) is accumulation:

$$
\boldsymbol{S}_t=\boldsymbol{S}_{t-1}+\boldsymbol{v}_t\boldsymbol{k}_t^{\top}\tag{1}
$$

If a batch of tokens arrives at once, accumulating the batch is $\boldsymbol{S}\leftarrow\boldsymbol{S}+\boldsymbol{V}^{\top}\boldsymbol{K}$ — a sum of outer products written as a single matrix multiplication.
Clearly, $(1)$ has no forgetting mechanism at all; once the sequence gets long, whatever is written in dilutes and interferes with each other.

The most productive angle for improving it is the [TTT](https://arxiv.org/abs/2407.04620)-style online learning view (also covered in [this blog post by Su Jianlin](https://kexue.fm/archives/11033)): treat $(\boldsymbol{k}_1,\boldsymbol{v}_1),\cdots,(\boldsymbol{k}_t,\boldsymbol{v}_t)$ as a corpus, view the state as a linear model $f(\boldsymbol{S};\boldsymbol{k})=\boldsymbol{S}\boldsymbol{k}$, and read out via $\boldsymbol{o}_t=f(\boldsymbol{S}_t;\boldsymbol{q}_t)$. "Writing" then becomes "training this model for one step on the new sample". Define the single-sample loss

$$
\mathcal{L}_t(\boldsymbol{S})=\tfrac12\lVert\boldsymbol{S}\boldsymbol{k}_t-\boldsymbol{v}_t\rVert^2,\qquad
\nabla_{\boldsymbol{S}}\mathcal{L}_t=\big(\boldsymbol{S}\boldsymbol{k}_t-\boldsymbol{v}_t\big)\boldsymbol{k}_t^{\top}\tag{2}
$$

and take one gradient step with learning rate $\beta_t$:

$$
\boldsymbol{S}_t=\boldsymbol{S}_{t-1}-\beta_t\nabla_{\boldsymbol{S}}\mathcal{L}_t(\boldsymbol{S}_{t-1})
=\boldsymbol{S}_{t-1}\big(\boldsymbol{I}-\beta_t\boldsymbol{k}_t\boldsymbol{k}_t^{\top}\big)+\beta_t\boldsymbol{v}_t\boldsymbol{k}_t^{\top}\tag{3}
$$

This is the [delta rule](https://arxiv.org/abs/2102.11174). What it does beyond accumulation is exactly the factor $\boldsymbol{I}-\beta_t\boldsymbol{k}_t\boldsymbol{k}_t^{\top}$: first erase the old content of the state along the $\boldsymbol{k}_t$ direction proportionally, then write the new one. At $\beta_t=1$, the residual at the current key goes exactly to zero — the new value completely replaces the old — whereas accumulation would just stack the two on top of each other. Adding a forget gate $\alpha_t\in(0,1)$ gives Gated DeltaNet:

$$
\boldsymbol{S}_t=\boldsymbol{S}_{t-1}\,\alpha_t\big(\boldsymbol{I}-\beta_t\boldsymbol{k}_t\boldsymbol{k}_t^{\top}\big)+\beta_t\boldsymbol{v}_t\boldsymbol{k}_t^{\top}\tag{4}
$$

[KDA](https://arxiv.org/abs/2510.26692) replaces the scalar gate with a channel-wise diagonal gate $\mathrm{diag}(\boldsymbol{a}_t)$; see the original paper for the exact form. For what follows, all that matters is that they can all be written as

$$
\boldsymbol{S}_t=\boldsymbol{S}_{t-1}\boldsymbol{A}_t+\boldsymbol{b}_t\tag{5}
$$

For GDN, $\boldsymbol{A}_t=\alpha_t(\boldsymbol{I}-\beta_t\boldsymbol{k}_t\boldsymbol{k}_t^{\top})$ and $\boldsymbol{b}_t=\beta_t\boldsymbol{v}_t\boldsymbol{k}_t^{\top}$. In other words, every token applies an affine transformation to the state, and a sequence is the ordered composition of these affine maps.

## Setting the Goal

We want the $n$ tokens within a frame to give the same result no matter how they are ordered. In mathematical terms, that is two requirements:

- **Permutation invariance**: after permuting the tokens within a frame, the state $\boldsymbol{S}_t$ passed to the next frame is completely unchanged;
- **Permutation equivariance**: each token's output follows the token itself under the permutation, with its value unchanged.

These two must be stated separately, because the former does not imply the latter: even if $\boldsymbol{S}_t$ is unchanged, as long as the $i$-th token in the frame still reads the prefix state at its own step, its output still depends on the order.

A natural idea is to make the keys of the tokens within a frame pairwise orthogonal, so that any two update matrices commute and who goes first no longer matters. But a $d_k$-dimensional space has at most $d_k$ pairwise orthogonal directions, so this route requires $n\le d_k$, while the number of tokens in a frame is usually much larger than $d_k$.

In fact, as long as we keep composing GDN-style token-by-token rank-1 affine updates in sequence, permutation invariance cannot be guaranteed structurally for general inputs. At bottom this is just the non-commutativity of matrix multiplication. Compose the $n$ affine maps of a frame in order $\pi$:

$$
\boldsymbol{S}_t=\boldsymbol{S}_{t-1}\prod_{i=1}^{n}\boldsymbol{A}_{\pi(i)}+\sum_{i=1}^{n}\boldsymbol{b}_{\pi(i)}\prod_{j\gt i}\boldsymbol{A}_{\pi(j)}\tag{6}
$$

Both terms must be independent of $\pi$. Look at the first term: swapping two adjacent factors and cancelling the two sides ($\boldsymbol{A}_i$ is invertible when $\alpha_i\ne0$ and $\beta_i\ne1$) gives $\boldsymbol{A}_i\boldsymbol{A}_j=\boldsymbol{A}_j\boldsymbol{A}_i$; and since adjacent transpositions generate all permutations, the first term being order-independent is equivalent to all $\boldsymbol{A}_i$ commuting pairwise. The scalar $\alpha$ does not affect commutativity, so directly expand the commutator of two update matrices:

$$
\big[\boldsymbol{I}-\beta_i\boldsymbol{k}_i\boldsymbol{k}_i^{\top},\ \boldsymbol{I}-\beta_j\boldsymbol{k}_j\boldsymbol{k}_j^{\top}\big]
=\beta_i\beta_j\,(\boldsymbol{k}_i^{\top}\boldsymbol{k}_j)\big(\boldsymbol{k}_i\boldsymbol{k}_j^{\top}-\boldsymbol{k}_j\boldsymbol{k}_i^{\top}\big)\tag{7}
$$

For this to vanish, apart from $\beta=0$ (which means nothing gets written), the two keys must be either orthogonal or parallel. Parallel does not work either: take $\alpha_i=\alpha_j=1$ and $\boldsymbol{k}_i=\boldsymbol{k}_j=\boldsymbol{k}$; using $\boldsymbol{k}^{\top}(\boldsymbol{I}-\beta\boldsymbol{k}\boldsymbol{k}^{\top})=(1-\beta)\boldsymbol{k}^{\top}$, a short calculation shows the difference of the write terms between the two orders is $\beta_i\beta_j(\boldsymbol{v}_j-\boldsymbol{v}_i)\boldsymbol{k}^{\top}$, which vanishes only when the two values happen to coincide. So to cover general inputs we are back to orthogonality, and hence back to $n\le d_k$.

The second term imposes an independent restriction. Under orthogonality, $\boldsymbol{k}_i^{\top}\boldsymbol{A}_j=\alpha_j\boldsymbol{k}_i^{\top}$, so the second term reduces to $\sum_i\beta_i\big(\prod_{j\ \text{placed after}\ i}\alpha_j\big)\boldsymbol{v}_i\boldsymbol{k}_i^{\top}$. The coefficient in parentheses depends on which tokens are placed after $i$; when all $\alpha_j$ are equal it degenerates into a power that depends only on the position. So as long as token-level forget gates remain within the frame, the order gets encoded into the state through $\alpha$, even if all keys are orthogonal. (Put differently, GDN comes with a built-in piece of positional information, which also echoes the fact that KDA can use NoPE.)

The conclusion: a token-by-token scheme requires both pairwise-orthogonal keys within the frame and the removal of intra-frame forget gates. The former is essentially impossible for video, and the latter forces the gating to be frame-level. So the more natural approach is to treat a whole frame as a single update.

## A First Attempt at Frame-Level Updates

The approach in SANA-WM is this: the unit of recurrence changes from token to frame, and all tokens of a frame are written in one shot. Placed side by side with the single-token line, the difference is obvious:

$$
\boldsymbol{S}_t=\boldsymbol{S}_{t-1}\big(\boldsymbol{I}-\beta_t\,\boldsymbol{k}_t\boldsymbol{k}_t^{\top}\big)+\beta_t\,\boldsymbol{v}_t\boldsymbol{k}_t^{\top}
\qquad\text{(single token)}\tag{8}
$$

$$
\boldsymbol{S}_t=\boldsymbol{S}_{t-1}\,\alpha_t\big(\boldsymbol{I}-\boldsymbol{K}_t^{\top}\boldsymbol{\Gamma}\boldsymbol{K}_t\big)+\boldsymbol{V}_t^{\top}\boldsymbol{\Gamma}\boldsymbol{K}_t
\qquad\text{(whole frame)}\tag{9}
$$

That is, the two outer products of a single token are replaced by weighted sums of outer products over the whole frame:

$$
\boldsymbol{K}_t^{\top}\boldsymbol{\Gamma}\boldsymbol{K}_t=\sum_{i=1}^{n}\gamma_i\boldsymbol{k}_i\boldsymbol{k}_i^{\top}\in\mathbb{R}^{d_k\times d_k},
\qquad
\boldsymbol{V}_t^{\top}\boldsymbol{\Gamma}\boldsymbol{K}_t=\sum_{i=1}^{n}\gamma_i\boldsymbol{v}_i\boldsymbol{k}_i^{\top}\in\mathbb{R}^{d_v\times d_k}\tag{10}
$$

where $\boldsymbol{\Gamma}=\mathrm{diag}(\gamma_1,\cdots,\gamma_n)$ holds the writing strength of each token (in SANA-WM it is just GDN's $\beta_i$), and $\alpha_t$ is the frame-level forget gate, one per frame. With $n=1$, $\boldsymbol{\Gamma}=\beta_t$ and $\alpha_t=1$, $(9)$ falls back to $(8)$. For readout, all queries in the frame read the same $\boldsymbol{S}_t$.

From the online learning viewpoint, $(9)$ is just "one ordinary gradient step on the total loss of the whole frame". Since the delta rule is "train one step on one token", and there is no order within a frame, we simply treat the frame's $n$ tokens as one batch and write down the total loss (the per-sample form and the matrix form are the same thing):

$$
\mathcal{L}_t(\boldsymbol{S})=\sum_{i=1}^{n}\frac{\gamma_i}{2}\big\lVert\boldsymbol{S}\boldsymbol{k}_i-\boldsymbol{v}_i\big\rVert^2
=\tfrac12\big\lVert\big(\boldsymbol{S}\boldsymbol{K}_t^{\top}-\boldsymbol{V}_t^{\top}\big)\boldsymbol{\Gamma}^{1/2}\big\rVert_F^2\tag{11}
$$

Taking the gradient:

$$
\nabla_{\boldsymbol{S}}\mathcal{L}_t(\boldsymbol{S})
=\big(\boldsymbol{S}\boldsymbol{K}_t^{\top}-\boldsymbol{V}_t^{\top}\big)\boldsymbol{\Gamma}\boldsymbol{K}_t
=\boldsymbol{S}\,\boldsymbol{K}_t^{\top}\boldsymbol{\Gamma}\boldsymbol{K}_t-\boldsymbol{V}_t^{\top}\boldsymbol{\Gamma}\boldsymbol{K}_t\tag{12}
$$

So $\boldsymbol{S}_{t-1}-\nabla\mathcal{L}_t(\boldsymbol{S}_{t-1})$ is exactly $(9)$ (ignoring $\alpha_t$), i.e. one gradient step with learning rate 1; the role of $\alpha_t$ is to decay the old state as a whole before taking this step. In the language of numerical analysis, this is an **explicit** (forward Euler) step: the gradient is evaluated at the starting point.

With this change, both properties are obtained. Permuting the tokens within a frame amounts to left-multiplying $\boldsymbol{K}_t,\boldsymbol{V}_t$ by a permutation matrix $\boldsymbol{P}_\pi$ and conjugating $\boldsymbol{\Gamma}$ accordingly, and since $\boldsymbol{P}_\pi^{\top}\boldsymbol{P}_\pi=\boldsymbol{I}$,

$$
(\boldsymbol{P}_\pi\boldsymbol{K}_t)^{\top}\big(\boldsymbol{P}_\pi\boldsymbol{\Gamma}\boldsymbol{P}_\pi^{\top}\big)(\boldsymbol{P}_\pi\boldsymbol{K}_t)=\boldsymbol{K}_t^{\top}\boldsymbol{\Gamma}\boldsymbol{K}_t\tag{13}
$$

The write term works the same way, so $\boldsymbol{S}_t$ is completely unchanged; on the output side, since all queries in the frame read the same $\boldsymbol{S}_t$, the outputs automatically follow the tokens under permutation. Put bluntly, permutation invariance here is just the single fact that "summation is order-independent".

We will keep coming back to these two terms (especially the eigenvalues of the first one), so let me give them short names:

$$
\boldsymbol{G}_t=\boldsymbol{K}_t^{\top}\boldsymbol{\Gamma}\boldsymbol{K}_t,\qquad
\boldsymbol{C}_t=\boldsymbol{V}_t^{\top}\boldsymbol{\Gamma}\boldsymbol{K}_t\tag{14}
$$

$\boldsymbol{G}_t$ is the "erasure", symmetric positive semidefinite; $\boldsymbol{C}_t$ is the "write". The explicit step then reads $\boldsymbol{S}_t=\boldsymbol{S}_{t-1}(\boldsymbol{I}-\boldsymbol{G}_t)+\boldsymbol{C}_t$ for short.

But written this way, training is unstable. $\boldsymbol{G}_t$ is a sum of $n$ terms within the frame; once its eigenvalue $\lambda$ in some direction exceeds 2, the magnitude of $1-\lambda$ exceeds 1 (with the frame-level gate $\alpha_t$ it is $\lvert\alpha_t(1-\lambda)\rvert\gt1$), the old state gets amplified, and if such amplification happens frame after frame, things diverge. SANA-WM's fix is to additionally divide the keys by $\sqrt{n}$ on top of L2 normalization. To keep the notation consistent, in what follows $\boldsymbol{K}_t$ always refers to unit-norm keys, and $\boldsymbol{G}_t,\boldsymbol{C}_t$ are defined from it. After this scaling, the erasure term becomes $\boldsymbol{G}_t/n$ and the write term becomes $\boldsymbol{C}_t/\sqrt{n}$: $\boldsymbol{G}_t$ is quadratic in the keys while $\boldsymbol{C}_t$ is linear, so the two are scaled by amounts differing by a factor of $\sqrt{n}$. The trace of the erasure term is then $\frac1n\sum_i\gamma_i\le1$, and since the largest eigenvalue of a PSD matrix does not exceed its trace, $\boldsymbol{I}-\boldsymbol{G}_t/n$ no longer amplifies the old state.

So the final update rule of this section, the one actually running in SANA-WM, is

$$
\boldsymbol{S}_t=\boldsymbol{S}_{t-1}\,\alpha_t\Big(\boldsymbol{I}-\frac{\boldsymbol{G}_t}{n}\Big)+\frac{\boldsymbol{C}_t}{\sqrt{n}}
\qquad\text{i.e.}\qquad
\boldsymbol{S}_t=\boldsymbol{S}_{t-1}\,\alpha_t\Big(\boldsymbol{I}-\frac{\boldsymbol{K}_t^{\top}\boldsymbol{\Gamma}\boldsymbol{K}_t}{n}\Big)+\frac{\boldsymbol{V}_t^{\top}\boldsymbol{\Gamma}\boldsymbol{K}_t}{\sqrt{n}}\tag{15}
$$

## Second Thoughts

Revisiting this approach recently, I found a hidden problem. The key is to see what the $\sqrt{n}$ actually affects. What determines how much of the old state survives is $\boldsymbol{G}_t/n$, so the retention factor along the $j$-th eigendirection is $1-\lambda_j/n$. And what the scaling in $(15)$ buys us is precisely an upper bound on the sum of these quantities:

$$
\sum_j\frac{\lambda_j}{n}=\frac{\mathrm{tr}\,\boldsymbol{G}_t}{n}\le1\tag{16}
$$

The $d_k$ directions share this budget of 1, so on average each frame can only erase about $1/d_k$ per direction — with $d_k=112$ that is less than $1\%$. This is the real cost of that $1/\sqrt{n}$: what gets cut is not the writing of a few tokens, but the total erasure of the whole frame, pressed down to 1.

I went ahead and measured this on the SANA-WM checkpoint. The median relative contribution of the erasure term $\boldsymbol{S}_{t-1}\boldsymbol{G}_t/n$ to the state is only $0.1\%$, while the write term contributes $37\%$; zeroing the erasure term entirely — falling back to gated accumulative linear attention — changes the decoded frames by only $35$ dB (the numerical noise of the same computation path is itself $46$ dB). In other words, these GDN layers are basically being used as accumulative linear attention, with the delta-rule part contributing very little.

One more number worth a look: the median of $\mathrm{tr}\,\boldsymbol{G}_t/n$ is $0.013$, two orders of magnitude below the bound given by $(16)$. So what really caps the writing strength is not stability, but the fact that $\beta_i\le1$ combined with the shrunk keys can only reach this far. The measured median of $\beta_i$ is already pinned at $0.98$.

## A Different Route

The root of the bottleneck: for the explicit step to be stable, the total erasure must be suppressed. Is there a way to walk whose stability does not depend on the writing strength at all? Yes — evaluate the gradient at the **landing point**, i.e. take an **implicit** ([backward Euler](https://en.wikipedia.org/wiki/Backward_Euler_method)) step:

$$
\boldsymbol{S}_t=\boldsymbol{S}_{t-1}-\nabla\mathcal{L}_t(\boldsymbol{S}_t)\tag{17}
$$

Here $\boldsymbol{S}_t$ appears on both sides — an implicit equation that needs solving. Fortunately the gradient formula $(12)$, $\nabla\mathcal{L}_t(\boldsymbol{S})=\boldsymbol{S}\boldsymbol{G}_t-\boldsymbol{C}_t$, holds for any $\boldsymbol{S}$, so plugging the not-yet-solved $\boldsymbol{S}_t$ in as the argument is perfectly legitimate; and since the frame-level loss is quadratic, its gradient is affine in $\boldsymbol{S}$, so after substitution we get a **linear** equation in $\boldsymbol{S}_t$ with a closed-form solution. Collecting $\boldsymbol{S}_t$ on one side:

$$
\boldsymbol{S}_t\big(\boldsymbol{I}+\boldsymbol{G}_t\big)=\boldsymbol{S}_{t-1}+\boldsymbol{C}_t
\qquad\Longrightarrow\qquad
\boldsymbol{S}_t=\big(\boldsymbol{S}_{t-1}+\boldsymbol{C}_t\big)\big(\boldsymbol{I}+\boldsymbol{G}_t\big)^{-1}\tag{18}
$$

Restoring the shorthands, it is

$$
\boldsymbol{S}_t=\big(\boldsymbol{S}_{t-1}+\boldsymbol{V}_t^{\top}\boldsymbol{\Gamma}\boldsymbol{K}_t\big)\big(\boldsymbol{I}+\boldsymbol{K}_t^{\top}\boldsymbol{\Gamma}\boldsymbol{K}_t\big)^{-1}\tag{19}
$$

Compared with the explicit $(15)$, only one thing changes: erasure goes from "subtract $\boldsymbol{K}_t^{\top}\boldsymbol{\Gamma}\boldsymbol{K}_t$" to "divide by $\boldsymbol{I}+\boldsymbol{K}_t^{\top}\boldsymbol{\Gamma}\boldsymbol{K}_t$".

The retention factor changes from $1-\gamma$ to $\frac{1}{1+\gamma}$: monotonically decreasing, always inside $(0,1]$, never negative and never amplifying no matter how large $\gamma$ gets; the matrix version is analogous — $\boldsymbol{G}_t$ being PSD guarantees the eigenvalues of $(\boldsymbol{I}+\boldsymbol{G}_t)^{-1}$ all lie in $(0,1]$. This is the entire advantage of implicit over explicit: $\gamma_i$ only needs to be non-negative; there is no need to shrink scales for stability, nor to estimate an eigenvalue bound per frame; directions with large $\gamma$ get erased cleanly, directions with small $\gamma$ are mostly left alone, and no direction has to make way for the largest one.

In particular, with $n=1$, the [Sherman–Morrison formula](https://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula) gives $(\boldsymbol{I}+\gamma\boldsymbol{k}\boldsymbol{k}^{\top})^{-1}=\boldsymbol{I}-\frac{\gamma}{1+\gamma}\boldsymbol{k}\boldsymbol{k}^{\top}$, and substituting back yields exactly the delta rule with learning rate $\frac{\gamma}{1+\gamma}$. In other words, this update generalizes the single-token case, and the writing strength and the delta learning rate differ only by a simple conversion:

$$
\gamma_i=\frac{\beta_i}{1-\beta_i},\qquad \beta_i=\frac{\gamma_i}{1+\gamma_i}\tag{20}
$$

This is very convenient in implementation: $\beta_i=\sigma(z_i)$ is obtained by passing some projection through a sigmoid, and $\sigma(z)/(1-\sigma(z))=e^{z}$, so $\gamma_i=e^{z_i}$ — just replace the sigmoid with an exp.
This update requires neither orthogonal keys nor any size relation between $n$ and $d_k$; it depends on the frame only through $\boldsymbol{G}_t,\boldsymbol{C}_t$, so its permutation invariance is exact, with nothing left over. And how different is it from the token-by-token sequential product? Expand: for weak writing, $(\boldsymbol{I}+\boldsymbol{G}_t)^{-1}=\boldsymbol{I}-\boldsymbol{G}_t+\mathcal{O}(\boldsymbol{G}_t^2)$, and the first-order term of $\prod_i(\boldsymbol{I}-\beta_i\boldsymbol{k}_i\boldsymbol{k}_i^{\top})$ is likewise $-\boldsymbol{G}_t$ — the two agree to first order; the discrepancy starts at second order, where which of each pair $(i,j)$ goes first in the sequential product produces different terms, the antisymmetric part of which is exactly the commutator in $(7)$. So the implicit step can be seen as a symmetrization of the sequential delta rule, and the influence of intra-frame order is itself an $\mathcal{O}(\gamma^2)$, second-order effect.

Computationally there is no need to actually invert anything. $\boldsymbol{I}+\boldsymbol{G}_t$ is symmetric positive definite with all eigenvalues $\ge1$; the forward pass is one [Cholesky factorization](https://en.wikipedia.org/wiki/Cholesky_decomposition) followed by one solve of $\boldsymbol{S}_t(\boldsymbol{I}+\boldsymbol{G}_t)=\boldsymbol{S}_{t-1}+\boldsymbol{C}_t$, and the size of the factorized matrix does not depend on $n$. The backward pass does not need to differentiate through the Cholesky factorization either — just keep the triangular factor from the forward pass and reuse it. Write $\boldsymbol{A}=\boldsymbol{I}+\boldsymbol{G}_t$ and $\boldsymbol{B}=\boldsymbol{S}_{t-1}+\boldsymbol{C}_t$, and let an overline denote the gradient of the loss with respect to that quantity (i.e. $\bar{\boldsymbol{X}}=\partial\mathcal{L}/\partial\boldsymbol{X}$). Taking differentials of $\boldsymbol{S}_t\boldsymbol{A}=\boldsymbol{B}$ on both sides gives $\mathrm{d}\boldsymbol{S}_t=(\mathrm{d}\boldsymbol{B}-\boldsymbol{S}_t\,\mathrm{d}\boldsymbol{A})\boldsymbol{A}^{-1}$; substituting into $\langle\bar{\boldsymbol{S}}_t,\mathrm{d}\boldsymbol{S}_t\rangle$ and moving $\boldsymbol{A}^{-1}$ to the left, one reads off

$$
\boldsymbol{Z}=\bar{\boldsymbol{S}}_t\boldsymbol{A}^{-1},\qquad
\bar{\boldsymbol{B}}=\boldsymbol{Z},\qquad
\bar{\boldsymbol{A}}=-\boldsymbol{S}_t^{\top}\boldsymbol{Z}\tag{21}
$$

which is one more solve of the same system plus one matrix multiplication. Note that $\bar{\boldsymbol{A}}$ is generally not symmetric, while $\boldsymbol{A}$ can only vary along symmetric directions, so before pushing further down we first take $\boldsymbol{H}=\tfrac12(\bar{\boldsymbol{A}}+\bar{\boldsymbol{A}}^{\top})$, then run the chain rule once more:

$$
\bar{\boldsymbol{V}}_t=\boldsymbol{\Gamma}\boldsymbol{K}_t\boldsymbol{Z}^{\top},\qquad
\bar{\boldsymbol{K}}_t=\boldsymbol{\Gamma}\boldsymbol{V}_t\boldsymbol{Z}+2\boldsymbol{\Gamma}\boldsymbol{K}_t\boldsymbol{H},\qquad
\bar{\gamma}_i=\boldsymbol{k}_i^{\top}\boldsymbol{H}\boldsymbol{k}_i+\boldsymbol{v}_i^{\top}\boldsymbol{Z}\boldsymbol{k}_i\tag{22}
$$

As for the term to be passed back to the previous frame: $\boldsymbol{B}=\boldsymbol{S}_{t-1}+\boldsymbol{C}_t$ is just a sum, so the gradient lands on both addends as is, giving $\bar{\boldsymbol{S}}_{t-1}=\bar{\boldsymbol{C}}_t=\boldsymbol{Z}$ with no extra computation. Altogether, forward and backward combined cost one Cholesky factorization, two linear solves, and a few matrix multiplications.

Besides the update rule itself, two more things need to follow suit.

First, the forget gate moves to frame boundaries: $\boldsymbol{a}_t$ is one group per frame, and can still be channel-wise.

Second, the writing strength needs an overall scale $s$ — but this time for quality, not for stability; stability is no longer a concern. Writing $\gamma_i = s\,e^{z_i}$ with $e^{z_i}$ of order $1$, we have $\mathrm{tr}\,\boldsymbol{G}_t\approx ns$, spread over at most $\min(n,d_k)$ nonzero eigenvalues; when $n\gt d_k$, the average per direction is of order $ns/d_k$, and the old state roughly retains $d_k/(d_k+ns)$. So if $s$ is $\mathcal{O}(1)$, one frame wipes out the history (with $n\sim10^3$ and $d_k\sim10^2$ only about a tenth of the old state remains); $s$ should sit around $d_k/n$ or even smaller. At $s=d_k/n$ the old state and the current frame each take roughly half; only below that can we talk about long memory across frames. Note that $s$ enters both $\boldsymbol{G}_t$ and $\boldsymbol{C}_t$, so erasure and writing are scaled proportionally — unlike the explicit $(15)$, where multiplying the keys by $1/\sqrt{n}$ costs the erasure $1/n$ but the write only $1/\sqrt{n}$.

Putting everything together, the full update rule is

$$
\begin{gathered}
\boldsymbol{S}_t=\Big(\boldsymbol{S}_{t-1}\mathrm{diag}(\boldsymbol{a}_t)+\boldsymbol{C}_t\Big)\big(\boldsymbol{I}+\boldsymbol{G}_t\big)^{-1}\\[4pt]
\boldsymbol{G}_t=\boldsymbol{K}_t^{\top}\boldsymbol{\Gamma}\boldsymbol{K}_t,\qquad
\boldsymbol{C}_t=\boldsymbol{V}_t^{\top}\boldsymbol{\Gamma}\boldsymbol{K}_t,\qquad
\boldsymbol{\Gamma}=s\,\mathrm{diag}\big(e^{z_1},\cdots,e^{z_n}\big)
\end{gathered}\tag{23}
$$

It is still affine in $\boldsymbol{S}_{t-1}$, so the cross-frame recurrence can still be written as affine pairs and parallelized with an associative scan.

## A Nice Coincidence

$n\le d_k$ essentially never happens in video, but this limiting case is interesting: it connects back to the orthogonalization idea we started with.

Suppose $\boldsymbol{K}_t$ has full row rank (which requires $n\le d_k$), and push the overall writing strength to infinity. Then the loss of this frame is pressed to zero, every $\boldsymbol{S}_t\boldsymbol{k}_i=\boldsymbol{v}_i$ holds exactly, i.e. $\boldsymbol{S}_t\boldsymbol{K}_t^{\top}=\boldsymbol{V}_t^{\top}$. These $n$ equations only constrain the subspace spanned by the keys; on its orthogonal complement $\boldsymbol{G}_t$ is zero, so $(\boldsymbol{I}+\boldsymbol{G}_t)^{-1}$ is strictly the identity there and the old state is left untouched. Together, $\boldsymbol{S}_t$ is uniquely determined:

$$
\boldsymbol{S}_t=\boldsymbol{S}_{t-1}\big(\boldsymbol{I}-\boldsymbol{\Pi}_t\big)+\boldsymbol{V}_t^{\top}\big(\boldsymbol{K}_t\boldsymbol{K}_t^{\top}\big)^{-1}\boldsymbol{K}_t,
\qquad \boldsymbol{\Pi}_t=\boldsymbol{K}_t^{\top}\big(\boldsymbol{K}_t\boldsymbol{K}_t^{\top}\big)^{-1}\boldsymbol{K}_t\tag{24}
$$

A quick check: $\boldsymbol{\Pi}_t\boldsymbol{K}_t^{\top}=\boldsymbol{K}_t^{\top}$, so the first term times $\boldsymbol{K}_t^{\top}$ is zero and the second term gives exactly $\boldsymbol{V}_t^{\top}$; $\boldsymbol{S}_t\boldsymbol{K}_t^{\top}=\boldsymbol{V}_t^{\top}$ indeed holds. $\boldsymbol{\Pi}_t$ is the orthogonal projection onto the subspace spanned by these keys, which depends only on the subspace itself, so permutation invariance is obvious. The meaning is also plain: the single-token delta rule erases one direction and writes; the frame-level delta rule erases the entire key subspace and writes.

And what if we actually orthogonalize the intra-frame keys first, then run the token-by-token recurrence as before? Note that the orthogonalization itself must also be order-independent. [Gram–Schmidt](https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process) does not qualify: it keeps the first vector fixed and subtracts from each subsequent vector its projections onto all previous ones — order-dependent by construction; using it merely moves the order-dependence from the recurrence into the preprocessing. [Löwdin symmetric orthogonalization](https://en.wikipedia.org/wiki/Orthogonalization) $\widetilde{\boldsymbol{K}}_t=(\boldsymbol{K}_t\boldsymbol{K}_t^{\top})^{-1/2}\boldsymbol{K}_t$ does qualify: permuting the rows of $\boldsymbol{K}_t$, i.e. left-multiplying by the $\boldsymbol{P}_\pi$ from $(13)$, turns $\boldsymbol{K}_t\boldsymbol{K}_t^{\top}$ into $\boldsymbol{P}_\pi(\boldsymbol{K}_t\boldsymbol{K}_t^{\top})\boldsymbol{P}_\pi^{\top}$, whose $-1/2$ power is conjugated accordingly, so $\widetilde{\boldsymbol{K}}_t$ is permuted along with it.

After orthogonalization, $\widetilde{\boldsymbol{K}}_t^{\top}\widetilde{\boldsymbol{K}}_t=\boldsymbol{\Pi}_t$; applying the same transform to the values, $\widetilde{\boldsymbol{V}}_t=(\boldsymbol{K}_t\boldsymbol{K}_t^{\top})^{-1/2}\boldsymbol{V}_t$, gives $\widetilde{\boldsymbol{V}}_t^{\top}\widetilde{\boldsymbol{K}}_t=\boldsymbol{V}_t^{\top}(\boldsymbol{K}_t\boldsymbol{K}_t^{\top})^{-1}\boldsymbol{K}_t$, identical to the second term of $(24)$. So "orthogonalize the keys, then run the sequential delta rule with $\beta=1$" and "solve the frame-level exact solution in one shot" are the same thing. The reason is simple: once the keys really are orthogonal, all the mutually correcting terms in the sequential recurrence vanish, and the order naturally stops mattering.

## Another Viewpoint

There is actually a third way to walk the same loss. Both explicit and implicit are discrete single steps; since there is no order within a frame, we can also skip discrete steps altogether and let the state evolve continuously along the gradient for a while — a gradient flow.

First an identity. $\boldsymbol{k}$ is a unit vector, so $\boldsymbol{k}\boldsymbol{k}^{\top}$ is a projection: $(\boldsymbol{k}\boldsymbol{k}^{\top})^2=\boldsymbol{k}(\boldsymbol{k}^{\top}\boldsymbol{k})\boldsymbol{k}^{\top}=\boldsymbol{k}\boldsymbol{k}^{\top}$, and all its positive integer powers equal itself. Expanding the matrix exponential as a power series, every term with $m\ge1$ carries the same $\boldsymbol{k}\boldsymbol{k}^{\top}$, and the coefficients add up to $\sum_{m\ge1}\frac{(-\gamma)^m}{m!}=e^{-\gamma}-1$:

$$
\exp\big(-\gamma\boldsymbol{k}\boldsymbol{k}^{\top}\big)=\boldsymbol{I}+\big(e^{-\gamma}-1\big)\boldsymbol{k}\boldsymbol{k}^{\top}=\boldsymbol{I}-\beta\boldsymbol{k}\boldsymbol{k}^{\top},\qquad \beta=1-e^{-\gamma}\tag{25}
$$

That is to say, the erasure matrix of the delta rule is itself a matrix exponential. Note that the $\gamma\leftrightarrow\beta$ conversion here is not the same as the one in $(20)$; where the difference comes from will be explained in the next section.

So the sequential product over a frame can be written as $\prod_i e^{\boldsymbol{X}_i}$ with $\boldsymbol{X}_i=-\gamma_i\boldsymbol{k}_i\boldsymbol{k}_i^{\top}$. By the [BCH formula](https://en.wikipedia.org/wiki/Baker%E2%80%93Campbell%E2%80%93Hausdorff_formula),

$$
\prod_i e^{\boldsymbol{X}_i}=\exp\Big(\sum_i\boldsymbol{X}_i+\tfrac12\sum_{i\lt j}\big[\boldsymbol{X}_i,\boldsymbol{X}_j\big]+\cdots\Big)\tag{26}
$$

What separates it from $\exp\big(\sum_i\boldsymbol{X}_i\big)$ is precisely those commutators $[\boldsymbol{X}_i,\boldsymbol{X}_j]$, the same objects as in $(7)$. Since $\sum_i\boldsymbol{X}_i$ is order-independent, we simply take $\exp\big(\sum_i\boldsymbol{X}_i\big)=e^{-\boldsymbol{G}_t}$.

Including the write term gives a differential equation: let the frame occupy a virtual time interval $\tau\in[0,1]$, during which all tokens of the frame act simultaneously,

$$
\frac{\mathrm{d}\boldsymbol{S}}{\mathrm{d}\tau}=-\boldsymbol{S}\boldsymbol{G}_t+\boldsymbol{C}_t,\qquad \boldsymbol{S}(0)=\boldsymbol{S}_{t-1}\tag{27}
$$

The right-hand side is exactly $-\nabla\mathcal{L}_t(\boldsymbol{S})$, so $(27)$ is the gradient flow of the frame-level total loss. Since $e^{-\boldsymbol{G}_t\tau}$ commutes with $\boldsymbol{G}_t$, the solution can be written exactly as in the scalar case (verify by differentiating with respect to $\tau$ and substituting back):

$$
\boldsymbol{S}_t=\boldsymbol{S}(1)=\boldsymbol{S}_{t-1}\,e^{-\boldsymbol{G}_t}+\boldsymbol{C}_t\,\varphi(\boldsymbol{G}_t),
\qquad \varphi(\boldsymbol{G}_t)=\int_0^1 e^{-\boldsymbol{G}_t u}\,\mathrm{d}u\tag{28}
$$

As a matrix function, $\varphi$ replaces each eigenvalue $\lambda$ by $(1-e^{-\lambda})/\lambda$, taking the value 1 at $\lambda=0$, so it is well-defined even when $\boldsymbol{G}_t$ is singular (the equivalent power series is $\varphi(\boldsymbol{G}_t)=\sum_{m\ge0}\frac{(-\boldsymbol{G}_t)^m}{(m+1)!}$). It likewise depends only on $\boldsymbol{G}_t,\boldsymbol{C}_t$, so it is likewise exactly permutation invariant.

What if we let the process run forever ($\tau\to\infty$)? Diagonalize $\boldsymbol{G}_t$ and look at $(28)$ eigenvalue by eigenvalue.

First, which directions are special. When $n\lt d_k$, $\mathrm{rank}\,\boldsymbol{G}_t\le n\lt d_k$, so $\boldsymbol{G}_t$ must have zero eigenvalues; and as long as all $\gamma_i\gt0$, from

$$
\boldsymbol{x}^{\top}\boldsymbol{G}_t\boldsymbol{x}=\boldsymbol{x}^{\top}\boldsymbol{K}_t^{\top}\boldsymbol{\Gamma}\boldsymbol{K}_t\boldsymbol{x}=\sum_{i=1}^{n}\gamma_i\big(\boldsymbol{K}_t\boldsymbol{x}\big)_i^2\tag{29}
$$

we know $\boldsymbol{G}_t\boldsymbol{x}=\boldsymbol{0}$ if and only if $\boldsymbol{K}_t\boldsymbol{x}=\boldsymbol{0}$, i.e. the zero-eigenvalue directions are exactly those orthogonal to all keys of the frame. The same identity also tells us $\boldsymbol{C}_t\boldsymbol{x}=\boldsymbol{V}_t^{\top}\boldsymbol{\Gamma}\boldsymbol{K}_t\boldsymbol{x}=\boldsymbol{0}$ — $\boldsymbol{C}_t$ vanishes on these directions as well.

Now take the limit eigenvalue by eigenvalue. Along directions with $\lambda_j\gt0$, $e^{-\lambda_j\tau}\to0$ and $\int_0^{\tau}e^{-\lambda_j u}\mathrm{d}u\to1/\lambda_j$, so the old state is cleared and replaced by $\boldsymbol{C}_t$ times $1/\lambda_j$; along directions with $\lambda_j=0$, the former is identically $1$ and the latter diverges linearly in $\tau$, but as just said $\boldsymbol{C}_t$ vanishes there, so the diverging term is multiplied away and the old state is left intact. Writing the two kinds of directions together:

$$
\boldsymbol{S}_\infty=\boldsymbol{S}_{t-1}\big(\boldsymbol{I}-\boldsymbol{G}_t\boldsymbol{G}_t^{\dagger}\big)+\boldsymbol{C}_t\boldsymbol{G}_t^{\dagger}\tag{30}
$$

where $\boldsymbol{G}_t^{\dagger}$ is the [Moore–Penrose pseudoinverse](https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse): it takes $1/\lambda_j$ along directions with $\lambda_j\gt0$ and $0$ along directions with $\lambda_j=0$ — precisely the matrix form of the two sentences above; and $\boldsymbol{I}-\boldsymbol{G}_t\boldsymbol{G}_t^{\dagger}$ is the projection onto the zero-eigenvalue directions. On the directions that $\boldsymbol{G}_t$ acts on, $\boldsymbol{S}_\infty$ satisfies $\boldsymbol{S}\boldsymbol{G}_t=\boldsymbol{C}_t$, the weighted least-squares solution of the frame-level loss. If we additionally impose the full-row-rank condition on $\boldsymbol{K}_t$ from the previous section, it coincides exactly with the projection solution there.

Incidentally, viewing the delta rule as an explicit-Euler discretization of an ODE and then solving its exact flow is somewhat similar to [Exact Flow Linear Attention](https://arxiv.org/abs/2512.12602) — that paper likewise relies on the rank-1 structure to degenerate the matrix exponential and the integral term into simple forms. The difference is that it deals with a single token, where the erasure matrix is rank-1; here $\boldsymbol{G}_t$ aggregates a whole frame, is generally full rank, and $\varphi(\boldsymbol{G}_t)$ does not degenerate.

## How the Three Relate

Reducing the problem to one dimension makes the relationship among the three clear. Suppose the eigenvalue of $\boldsymbol{G}_t$ along some eigendirection is $\lambda$, and ask how much of the old state survives one frame along this direction: explicit keeps $1-\lambda$, implicit keeps $\frac{1}{1+\lambda}$, and gradient flow keeps $e^{-\lambda}$.

With the three numbers side by side, two things become clear at once. First, the explicit one goes negative for $\lambda\gt1$ and exceeds 1 in magnitude for $\lambda\gt2$, while the other two honestly stay between $0$ and $1$ no matter how large $\lambda$ gets. Second, the two $\gamma$-versus-$\beta$ conversions in $(20)$ and $(25)$ — why they disagree — originate right here: to match the surviving fraction to the delta rule's $1-\beta$, the implicit side solves to $\gamma=\beta/(1-\beta)$, while the gradient-flow side solves to $\gamma=-\ln(1-\beta)$. The same $\beta$ corresponds to different $\gamma$ in the two walks, so the two conversions naturally differ.

A few things should be made clear: what we compared above is a single direction and a single step. GDN itself does not take one explicit step on the whole-frame loss — ignoring the forget gate, what it does over a frame is $\prod_i(\boldsymbol{I}-\beta_i\boldsymbol{k}_i\boldsymbol{k}_i^{\top})$, the $n$ erasure matrices multiplied in order, which is exactly why it cares about the order. Also, implicit and gradient flow are not the same thing either; they coincide only in the two extremes of extremely weak and extremely strong writing, and their hyperparameters must not be mixed.

The three walks are in fact three special cases of one formula:

$$
\boldsymbol{S}_t=\boldsymbol{S}_{t-1}\,f(\boldsymbol{G}_t)+\boldsymbol{C}_t\,g(\boldsymbol{G}_t)\tag{31}
$$

Here $f,g$ act by replacing each eigenvalue $\lambda$ of $\boldsymbol{G}_t$ with $f(\lambda)$ and $g(\lambda)$: explicit takes $f=1-\lambda$, $g=1$; implicit takes $f=g=\frac{1}{1+\lambda}$; gradient flow takes $f=e^{-\lambda}$, $g=\frac{1-e^{-\lambda}}{\lambda}$. So permutation invariance comes entirely from $\boldsymbol{G}_t,\boldsymbol{C}_t$ being sums; $f,g$ only decide how to update.

Each choice of $f,g$ is a new walk. For example, split the gradient flow into $m$ substeps of size $1/m$: repeat $m$ times $\boldsymbol{S}\leftarrow\boldsymbol{S}(\boldsymbol{I}-\boldsymbol{G}_t/m)+\boldsymbol{C}_t/m$, and the surviving fraction is $(1-\lambda/m)^m$, approaching $e^{-\lambda}$ as $m$ grows. As long as $m$ is larger than $\lambda_{\max}/2$, no substep amplifies the old state, and the whole pass is pure matrix multiplication with no factorization; the cost is that the backward pass has to replay these $m$ steps. It is also permutation invariant, because each step only uses $\boldsymbol{G}_t$ and $\boldsymbol{C}_t$. Conversely, if instead we split the frame's tokens into several groups and multiply them in one group after another, the order-dependence comes right back.

Finally, the price. Token-by-token erase-then-write is itself useful: with $\beta_i=1$ and no forget gate, $n$ tokens acting in sequence amount to projecting the state successively onto the constraints $\boldsymbol{S}\boldsymbol{k}_i=\boldsymbol{v}_i$ — that is, [Kaczmarz iteration](https://en.wikipedia.org/wiki/Kaczmarz_method) — so later tokens can repair what earlier tokens wrote badly. [GLA](https://arxiv.org/abs/2312.06635) and [Mamba2](https://arxiv.org/abs/2405.21060), although also recurrent token by token, write without reading the current state — mere accumulation plus global decay — and lack this correction mechanism. This serial correction ability may also be one reason why [DeltaNet-style models](https://arxiv.org/abs/2406.06484) beat GLA and Mamba2 in some experiments. Frame-level whole updates merge the intra-frame samples symmetrically, and this mechanism is gone. So this is a trade-off, not an improvement. For the intra-frame dimension of video, I think the trade is worth making: that bit of serial depth runs along a scan line we made up ourselves, while the serial depth across frames is fully preserved — and that is the causality that really exists in video.

## Summary

This post first pins down the properties a "video-native delta rule" should satisfy: change the intra-frame scan order, and neither the state passed to the next frame nor the current frame's outputs should change. Following the online learning viewpoint, we then derived the general shape of frame-level updates and compared three walks: explicit, implicit, and gradient flow. The implicit step frees the writing strength from the stability constraint, at the cost of one extra small matrix factorization per frame.

One final disclaimer: these are just some ideas I derived out of the spirit of "research for fun and truth", and their actual effect has never been verified. Even the question "does a video DeltaNet really need intra-frame permutation invariance" has no answer — it looks like a natural prior, but the network may well learn its own different solution. Take this as no more than pen-and-paper reasoning on my part.

## References

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
- [Su Jianlin: A Brief History of Linear Attention — From Imitation and Innovation to Giving Back](https://kexue.fm/archives/11033)
- [Parallelizing Linear Transformers with the Delta Rule over Sequence Length](https://arxiv.org/abs/2406.06484)
- [Gated Linear Attention Transformers with Hardware-Efficient Training (GLA)](https://arxiv.org/abs/2312.06635)
- [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality (Mamba2)](https://arxiv.org/abs/2405.21060)

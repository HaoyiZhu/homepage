After finishing the previous blog post, I had an in-depth discussion with my colleagues ([Haoge Deng](https://bitterdhg.github.io/), [Yuhang Yang](https://yyvhang.github.io/), and [Wen Wang](https://encounter1997.github.io/)), and some new findings and reflections came out of it. I record and share them here.

## Picking Up Where We Left Off

In the [previous post](https://haoyizhu.site/blog/sparse-linear-attention/), we introduced schemes that mix sparse attention and linear attention along the sequence dimension. Take my second-order-corrected scheme [PWT](https://github.com/HaoyiZhu/Piecewise-Taylor-Attention) as an example: for each query $\boldsymbol{q}_t$, we first score blocks and select the top-K blocks $\mathcal{S}_t$ for exact softmax attention; the remaining blocks $\mathcal{U}_t$ are not discarded — instead, they are Taylor-expanded around the block centroid $\bar{\boldsymbol{k}}_j$ and compressed into "virtual tokens" with a second-order correction as compensation. Writing the output in numerator/denominator form $\boldsymbol{o}_t=\mathcal{N}_t/\mathcal{D}_t$:

$$
\mathcal{D}_t=\underbrace{\sum_{j\in\mathcal{S}_t}\sum_{i\in\mathcal{B}_j} e^{\boldsymbol{q}_t\cdot\boldsymbol{k}_i}}_{\text{exact branch}}+\underbrace{\sum_{j\in\mathcal{U}_t} B\,\exp\big(\boldsymbol{q}_t\cdot\bar{\boldsymbol{k}}_j+w_{t,j}\big)}_{\text{Taylor-approximated branch}}
$$

$$
\mathcal{N}_t=\sum_{j\in\mathcal{S}_t}\sum_{i\in\mathcal{B}_j} e^{\boldsymbol{q}_t\cdot\boldsymbol{k}_i}\,\boldsymbol{v}_i+\sum_{j\in\mathcal{U}_t} \exp\big(\boldsymbol{q}_t\cdot\bar{\boldsymbol{k}}_j+w_{t,j}\big)\Big(\sum_{i\in\mathcal{B}_j}\boldsymbol{v}_i+\boldsymbol{H}_j\,\boldsymbol{q}_t\Big)
$$

where $w_{t,j}=\frac{1}{2}\boldsymbol{q}_t^{\top}\boldsymbol{C}_j\boldsymbol{q}_t$ is the second-order correction to the block mass ($\boldsymbol{C}_j$ being the within-block covariance of the keys), $\boldsymbol{H}_j=\sum_{i\in\mathcal{B}_j}\boldsymbol{v}_i(\boldsymbol{k}_i-\bar{\boldsymbol{k}}_j)^{\top}$ is the first-order key–value cross statistic within a block, and the two branches share the same denominator, so normalization is consistent by construction.

One can see that this family of schemes combining sparse and linear attention (PWT / SLA / PISA, etc.) consists of two parts: **top-K selection + long-tail compensation**. The former decides which positions get exact attention; the latter decides how to approximate the remaining region, e.g., with a Taylor-expanded linear attention as compensation. The previous post focused on the compensation part — how to upgrade "discard" into "approximate"; in this post we turn to the other half: how exactly should the top-K be selected?

## Selection Error

The mainstream sparse attention schemes in video models today (e.g., [VSA](https://arxiv.org/abs/2505.13389), [PISA](https://arxiv.org/abs/2602.01077), [PWT](https://github.com/HaoyiZhu/Piecewise-Taylor-Attention)) all use a [MoBA](https://arxiv.org/abs/2502.13189)-style selection: partition into blocks, average within each block, and score with the centroid to pick the top-K blocks. This family is simple and parameter-free, but the loss in precision is evident. The first source is averaging itself: within-block averaging flattens local peaks, so the larger the block size, the more likely a true top-K token gets "averaged out" and missed. On top of that, there is a less frequently discussed source of error: positional encoding. RoPE applies a position-dependent rotation $\boldsymbol{R}_i$ to the key at position $i$, and satisfies the relative-position property

$$
(\boldsymbol{R}_m\boldsymbol{q})\cdot(\boldsymbol{R}_n\boldsymbol{k})=\boldsymbol{q}^{\top}\boldsymbol{R}_{n-m}\,\boldsymbol{k}
$$

i.e., the inner product depends only on the relative distance $n-m$. But if we apply RoPE first and then do block-wise averaging, the resulting "centroid" is

$$
\widetilde{\boldsymbol{k}}_j^{\mathrm{rope}}=\frac{1}{B}\sum_{i\in\mathcal{B}_j}\boldsymbol{R}_i\,\boldsymbol{k}_i
$$

which is a mixture of $B$ different rotations — it cannot be written in the form $\boldsymbol{R}_{c_j}\bar{\boldsymbol{k}}_j$, i.e., "average first, then rotate at a single position" (note that $\boldsymbol{R}_{c_j}\bar{\boldsymbol{k}}_j=\frac{1}{B}\sum_{i\in\mathcal{B}_j}\boldsymbol{R}_{c_j}\boldsymbol{k}_i$ differs from the above exactly by replacing the per-token rotation $\boldsymbol{R}_i$ with $\boldsymbol{R}_{c_j}$). The block score then becomes

$$
(\boldsymbol{R}_t\boldsymbol{q}_t)\cdot\widetilde{\boldsymbol{k}}_j^{\mathrm{rope}}=\frac{1}{B}\sum_{i\in\mathcal{B}_j}\boldsymbol{q}_t^{\top}\boldsymbol{R}_{i-t}\,\boldsymbol{k}_i
$$

a mixture of inner products at many different relative offsets, so RoPE's relative-position structure is broken — and the larger the block, the worse the distortion.

For the RoPE issue, one mitigation is to average *before* RoPE, and then apply RoPE with the block center's index $c_j$, i.e., take the block score as

$$
(\boldsymbol{R}_t\boldsymbol{q}_t)\cdot\big(\boldsymbol{R}_{c_j}\bar{\boldsymbol{k}}_j\big)=\boldsymbol{q}_t^{\top}\boldsymbol{R}_{c_j-t}\,\bar{\boldsymbol{k}}_j
$$

so the block score once again becomes a well-defined inner product that depends only on the relative position $c_j-t$. This may help a little in the training-free setting; but if what we want is a *trainable* selection scheme, this patch is somewhat inessential — it is still patching up "averaging" rather than answering head-on "which ones should be selected".

In fact, in the LLM domain, the most famous large-scale application of top-K selection for sparse attention is DeepSeek-V3.2's [DSA](https://arxiv.org/abs/2512.02556) (in contrast, Kimi's MoBA was never deployed at scale in Kimi's production models, which may also indirectly suggest that the performance of block-mean selection is limited). Recalling DSA's scheme: it trains an additional lightweight lightning indexer that scores every (query, key) pair, and then selects the top-K tokens by score (with K = 2048) for exact attention:

$$
I_{t,s}=\sum_{j=1}^{H^I} w^{I}_{t,j}\cdot\mathrm{ReLU}\big(\boldsymbol{q}^{I}_{t,j}\cdot\boldsymbol{k}^{I}_{s}\big)
$$

where $w^{I}_{t,j}$ is a head weight computed from the query. The indexer is learnable, but its training is decoupled from the main model: first, in a dense warm-up stage, the main model is frozen and a KL divergence aligns the indexer's score distribution with the main model's attention distribution; then comes the joint sparse-training stage, where the indexer receives only its own alignment loss and the main model receives only the LM loss — the two are mutually detached in the computation graph.

As we can see, this indexer has two core advantages: first, it is **learnable** — the scoring criterion is trained from data rather than relying on the "the mean represents" heuristic; second, it is **token-wise precise** — top-K is selected directly at token granularity, so there is no mis-selection caused by block averaging. Together, its selection quality is clearly better than MoBA-style block averaging. On the other hand, the indexer's scoring itself is performed over the full sequence: every query computes $I_{t,s}$ against every key (in the causal setting, every preceding token), so the complexity is still $\mathcal{O}(L^2)$ — only with a much smaller constant than the main attention (fewer heads, lower head dimension, FP8, and possibly tricks like sharing scores across layers). When the token count becomes very large — say 10M or more — this quadratic term is still unaffordable.

## From Coarse to Fine

Let us review the two families of top-K selection: MoBA-style block averaging is coarse, but parameter-free and cheap; DSA's indexer is precise and learnable, but expensive on long sequences. A very natural idea is therefore to combine them into a coarse-to-fine scheme that takes the best of both — let me tentatively call it HSA (Hierarchical Sparse Attention). Concretely, it proceeds in two steps:

1. **Coarse selection**: do block-wise averaging first, and score with centroids to select the top-M blocks. Here M is deliberately set large — we'd rather over-select than miss;
2. **Fine selection**: only inside the selected M blocks, run a DSA-style learnable indexer for token-wise precise top-K selection.

The insight here is: although block averaging has error, the magnitude of that error is bounded — as long as M is set large enough, the coarsely selected blocks cover the true top-K tokens with overwhelming probability. For example, we may coarsely select the top 10% of blocks and then select the top 1% of tokens within them: block averaging may shift the ranking of the blocks containing some true top-1% tokens away from the top 1%, but this deviation can hardly push them out of the top 10%. Of course, the actual hyperparameters need to be tuned according to training, but the core idea remains: **first coarsely select, at low cost, a candidate subset that contains the final top-K with high probability, and then perform precise learnable selection within the subset**.

The cost picture is also intuitive: in the coarse stage, each query only takes inner products with $L/B$ centroids — $B$ times cheaper than token-level scoring, and parameter-free to boot; in the fine stage, the indexer scores only $MB$ candidate tokens. The overall complexity drops from DSA's full-sequence $\mathcal{O}(L^2)$ (small constant, but still quadratic) to a two-level "coarse + fine" cost of

$$
\mathcal{O}\Big(\frac{L^2}{B}+L\cdot MB\Big)
$$

and the fine stage can fully inherit DSA's engineering tricks (few heads, low dimension, FP8). When sequences reach the 10M-token regime, this constant-factor difference is often exactly the difference between "runnable" and "not runnable". Furthermore, if one level of coarse selection is not enough, this hierarchical structure can naturally recurse into more levels — we will not expand on that here.

## Wrapping Up

In this post we discussed the easily overlooked other half of sparse attention — how to select the top-K: block-averaging schemes are cheap but suffer from averaging- and positional-encoding-induced errors; learnable indexers are precise but their full-sequence scoring is too expensive at ultra-long contexts. A natural way out is to combine the two with a coarse-to-fine hierarchical selection (HSA): use cheap coarse selection to guarantee recall, and learnable fine selection to guarantee precision. Note also that top-K selection and the linear-approximation compensation from the previous post are **orthogonal**: HSA decides "where to compute exactly", while PWT-style compensation decides "how to approximate the rest", and the two can absolutely be combined. In fact, as token counts keep growing, I believe that even for LLMs, DSA's current "select and discard" practice will hit a bottleneck — if we scale to 100M tokens in the future and still only attend exactly to the top 2048, the amount of discarded information is intuitively just too much — so bringing linear-approximation compensation into the ultra-long-sequence regime would be a very valuable direction.

## References

- [Previous post: Sparse Linear Attention — When Sparsity Meets Linear Attention](https://haoyizhu.site/blog/sparse-linear-attention/)
- [MoBA: Mixture of Block Attention for Long-Context LLMs](https://arxiv.org/abs/2502.13189)
- [DeepSeek-V3.2 (DSA)](https://arxiv.org/abs/2512.02556)
- [VSA: Faster Video Diffusion with Trainable Sparse Attention](https://arxiv.org/abs/2505.13389)
- [SLA: Beyond Sparsity in Diffusion Transformers via Fine-Tunable Sparse–Linear Attention](https://arxiv.org/abs/2509.24006)
- [PISA: Piecewise Sparse Attention Is Wiser for Efficient Diffusion Transformers](https://arxiv.org/abs/2602.01077)
- [PWT code: HaoyiZhu/Piecewise-Taylor-Attention](https://github.com/HaoyiZhu/Piecewise-Taylor-Attention)

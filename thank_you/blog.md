# Saying Thank You to a LLM Isn't Free — Measuring the Energy Cost of Politeness

*What if a simple “thank you” to your favorite chatbot came with an energy price tag?*

## Is it worth saying thank you to a language model?

In everyday conversations, it’s common — almost instinctive — to end with a polite “thank you.” But when the conversation is with a large language model, that small gesture isn’t entirely free. Behind the scenes, even a simple “thank you!” triggers a full inference pass through billions of parameters, consuming real computational resources and energy.

Unlike humans, LLMs don’t remember conversations once they end — unless developers explicitly store information in external systems. So from the model’s point of view, saying “thank you” serves no functional purpose.

**We asked ourselves: What is the energy cost of being polite to an LLM?**

To find out, we created a custom dataset with thousands of conversations, each ending in a final “thank you,” and measured the energy it takes for the model to respond.

![Thank You Message on ChatGPT](benchlab/thank_you/static/thank_you_chat_gpt.png) 

## So, how much does it cost to say "thank you"?

We measured the average energy consumption of a single "thank you!" response on **LLaMA 3–8B** across **10,000 conversations**. On an H100 GPU, this polite gesture consumes on average:

* **0.202 ± 0.096 Wh** on the **GPU**
* **0.024 ± 0.014 Wh** on the **CPU**
* **0.019 ± 0.010 Wh** from **RAM**
  **→ Total: \~0.245 Wh**

That’s roughly equivalent to:

* Charging a smartphone by **2.45%** (assuming a 10 Wh battery), or
* Powering a **5W LED bulb for \~3 minutes** (0.245 Wh ÷ 5 W × 60 = \~2.94 minutes)

The **bar plot** below shows GPU usage dominates by far, with a high variance — suggesting sensitivity to runtime conditions.

The **second figure** plots the distribution of GPU energy per generation. It exhibits a **right-skewed Gaussian** shape, with a long tail toward higher values — indicating that some “thank you” generations consume **much more energy** than the average.

---

## What causes these fluctuations?

To answer that, we need to understand what happens during inference.

Each transformer block in GPT has three main components:

1. **Dense projections** (Q, K, V)
2. **Self-attention mechanism**
3. **Feed-forward network (MLP)**

These blocks apply **large matrix multiplications**, which dominate both compute time and energy.

### Inference Phases

During inference, the model goes through two key phases:

* **Prefill**: the entire prompt (i.e. the full conversation history) is passed through all transformer layers. Each token is processed in parallel through the feed-forward components, while attention layers compute interactions between tokens. During this phase, the model constructs and stores the **key-value (KV) cache** — a set of intermediate attention matrices that capture contextual information for each layer. This cache will be reused in the next phase to avoid recomputing past context.
* **Decode**: with the KV cache ready, the model generates new tokens **one at a time**, using only the **most recent token** and the stored cache. This step is sequential and **autoregressive**: each generated token becomes the input for the next step, progressively building the response.

> *Prefill: parallel processing of prompt tokens; Decode: sequential output token generation.*
> *(Credits: [https://huggingface.co/blog/tngtech/llm-performance-prefill-decode-concurrent-requests](https://huggingface.co/blog/tngtech/llm-performance-prefill-decode-concurrent-requests))*

---

## Component-wise Energy Consumption

We measured average energy usage across GPU, CPU, and RAM for prefill and decode phases across 10,000 generations.

* **Bar plots** (left):
  Decode dominates across **all components**.
  GPU usage: **>8× CPU**, **>10× RAM**

* **Histogram** (right):
  Decode accounts for most of the **long tail**.
  Prefill is more **variable**, but contributes less overall.

---

## 🧪 Experimental Setup

To measure the energy cost of polite messages, we built a reproducible pipeline using Hugging Face tools and hardware monitoring.

**Hardware**:

* NVIDIA H100 (80GB SXM)

**Software**:

* `transformers==4.52.3`, `torch==2.7.0`
* `codecarbon==3.0.2` (GPU via NVML, CPU via pyRAPL, RAM estimated)

**Dataset**:

* 10,000 samples from `ultrachat_200k`, ending in “thank you”

**Procedure**:

* 5 warmup runs
* 10 runs per sample → averaged
* Measured: GPU/CPU/RAM energy (mJ), prompt/output length, latency
* Prefill/decode phases separated when possible

**Models tested**:

* LLaMA 3.1–8B-Instruct
* Qwen 2.5 family (0.5B → 14B)
* Mistral-7B-Instruct-v0.3

---

## A Deeper Dive into Compute Costs: Prefill vs Decode

| Step                    | Formula (Prefill / Decode)                                                       | Total Cost (Prefill)     | Total Cost (Decode)      | Notes                                                             |
|-------------------------|----------------------------------------------------------------------------------|---------------------------|---------------------------|--------------------------------------------------------------------|
| **Embedding lookup**    | $x = E[t]$                                                                       | $\mathcal{O}(n \cdot d)$  | –                         | Simple table lookup, negligible compute cost                      |
|                         |                                                                                  |                           |                           |                                                                    |
| **(repeated $N$ times)**|                                                                                  |                           |                           | **Transformer block components repeated $N$ times**                |
| LayerNorm ×2            | –                                                                                | $\mathcal{O}(n \cdot d)$  | $\mathcal{O}(m \cdot d)$  | Negligible                                                        |
| Q, K, V projections     | $Q = XW^Q$, $K = XW^K$, $V = XW^V$                                               | $\mathcal{O}(n \cdot d^2)$| $\mathcal{O}(m \cdot d^2)$| Dense linear layers                                               |
| Attention scores        | Prefill: $QK^\top \in \mathbb{R}^{n \times n}$ <br> Decode: $qK^\top \in \mathbb{R}^{1 \times n}$ | $\mathcal{O}(n^2 \cdot d)$| $\mathcal{O}(m \cdot n \cdot d)$| Uses cached $K$ in decode                                |
| Weighted sum (AV)       | Prefill: $AV \in \mathbb{R}^{n \times d}$ <br> Decode: $aV \in \mathbb{R}^{1 \times d}$          | $\mathcal{O}(n^2 \cdot d)$| $\mathcal{O}(m \cdot n \cdot d)$| Uses attention weights                                  |
| MLP (2 dense layers)    | $xW_1 \rightarrow \text{act} \rightarrow W_2$                                   | $\mathcal{O}(n \cdot d \cdot d_{\text{ff}})$| $\mathcal{O}(m \cdot d \cdot d_{\text{ff}})$ | Usually the most expensive sublayer                   |
| Output projection       | $xW^O \in \mathbb{R}^{d}$                                                       | $\mathcal{O}(n \cdot d^2)$| $\mathcal{O}(m \cdot d^2)$| Final linear projection in each block                             |
|                         |                                                                                  |                           |                           |                                                                    |
| **LM Head**             | $xE^\top \in \mathbb{R}^V$                                                      | –                         | $\mathcal{O}(m \cdot d \cdot V)$| Costly if vocabulary size $V$ is large                     |

### Legend:

* **n** = input sequence length
* **m** = number of generated tokens
* **d** = model hidden size
* **d\_ff ≈ 4d** = FFN hidden size
* **V** = vocabulary size (e.g., 128k)

## Parameter Breakdown: LLaMA 3 8B vs 70B

| Component                  | Formula / Description                              | LLaMA 3 8B       | LLaMA 3 70B      |
|---------------------------|----------------------------------------------------|------------------|------------------|
| **# Transformer Layers**  | –                                                  | 32               | 80               |
| **Hidden size (d)**       | Model dimension                                    | 4096             | 8192             |
| **FFN hidden size (d_ff)**| Usually 4×d                                        | 11008            | 28672            |
| **# Attention Heads**     | –                                                  | 32               | 64               |
| **Head dim (d_k)**        | $d / \text{heads}$                                 | 128              | 128              |
| **Embedding params**      | $V \times d$ (Vocab ≈128k)                         | ~0.5B            | ~1.0B            |
| **Attention params**      | 4 × $(d × d)$ per layer                            | ~2.1B            | ~13.4B           |
| **FFN params**            | 2 × $(d × d_{ff})$ per layer                       | ~4.6B            | ~50.0B           |
| **LayerNorm + others**    | Small (bias, norm, etc.)                           | ~0.1B            | ~0.6B            |
| **LM Head**               | Shared with embeddings (tied weights)              | –                | –                |
| **Total**                 | Sum                                                | **~8B**          | **~70B**         |


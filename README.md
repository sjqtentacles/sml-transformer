# sml-transformer

[![CI](https://github.com/sjqtentacles/sml-transformer/actions/workflows/ci.yml/badge.svg)](https://github.com/sjqtentacles/sml-transformer/actions/workflows/ci.yml)

Transformer building blocks in pure Standard ML, layered on
[`sml-tensor`](https://github.com/sjqtentacles/sml-tensor): the pieces a
decoder-only (GPT-style) model needs that a general tensor library doesn't
provide — embedding lookup, row slicing, a numerically-stable row-wise softmax,
LayerNorm, GELU, an affine linear layer, multi-head **causal** self-attention,
and a full pre-LayerNorm transformer block.

Reusable for any transformer (GPT-2/3, BERT, …). Everything is pure and
deterministic: reals are compared with `Tensor.approxEq` (never `Real.toString`),
and the only transcendentals are `Math.tanh`/`Math.exp`/`Math.sqrt` (both MLton
and Poly/ML defer to the same `libm`), so results — including a whole attention
block — are **byte-identical across the two compilers**.

## Usage

```sml
open Transformer   (* type t = Tensor.t *)

val h   = embed wte tokenIds                       (* token embeddings [n, d]   *)
val y   = layerNorm {eps=1e~5, g=g, b=b} h
val z   = linear {w=w, b=b} y                      (* y = x@W + b (Conv1D)      *)
val a   = attention {nHeads=12, wQkv=wq, bQkv=bq, wProj=wp, bProj=bp} y
val out = block blockWeights h                     (* one full GPT-2 block      *)
```

## API (`signature TRANSFORMER`)

```sml
type t = Tensor.t
val embed       : t -> int list -> t                  (* gather rows by token id *)
val sliceRows   : int * int -> t -> t
val softmaxRows : t -> t                              (* stable, per row          *)
val gelu        : t -> t                              (* GPT-2 gelu_new (tanh)    *)
val layerNorm   : {eps:real, g:t, b:t} -> t -> t      (* biased variance          *)
val linear      : {w:t, b:t} -> t -> t                (* y = x@w + b              *)
val attention   : {nHeads:int, wQkv:t, bQkv:t, wProj:t, bProj:t} -> t -> t
type blockWeights = { eps:real, nHeads:int,
                      ln1g:t, ln1b:t, wQkv:t, bQkv:t, wProj:t, bProj:t,
                      ln2g:t, ln2b:t, wFc:t, bFc:t, wFcProj:t, bFcProj:t }
val block : blockWeights -> t -> t
```

Conventions match GPT-2 exactly: HF `Conv1D` linear layout (`y = x@W + b`), heads
split by columns, scaled dot-product attention with a causal mask, `gelu_new`
tanh approximation, and biased LayerNorm variance.

## Testing

```
make test       # MLton
make test-poly  # Poly/ML
```

9 assertions, green on MLton and Poly/ML with byte-identical output. The
reference (golden) values are **NumPy-derived** — see `tools/gen_reference.py`,
which computes each op (softmax, gelu, layernorm, linear, attention, block) on
fixed tiny inputs with GPT-2-exact math; the SML must reproduce them within
epsilon. (CI needs no Python; only the SML and the committed constants ship.)

## License

MIT

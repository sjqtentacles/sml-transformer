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

## Installation

```
smlpkg add github.com/sjqtentacles/sml-transformer
smlpkg sync
```

Depends on [`sml-tensor`](https://github.com/sjqtentacles/sml-tensor) (fetched by `smlpkg sync`).

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

## Example

`make example` builds and runs [`examples/demo.sml`](examples/demo.sml), which
runs a tiny embedding table through softmax, GELU, LayerNorm, a linear
layer, causal self-attention, and a full pre-LayerNorm transformer block,
printing every tensor through `Tensor.toString` (output is byte-identical
under MLton and Poly/ML):

```
=== sml-transformer demo ===

-- embed [4,1,3] over the vocab table, shape [3,4] --
tensor(shape=[3,4])
[1.7000, 1.8000, 1.9000, 2.0000, 0.5000, 0.6000, 0.7000, 0.8000, 1.3000, 1.4000, 1.5000, 1.6000]
-- sliceRows (0,2): first two rows --
tensor(shape=[2,4])
[1.7000, 1.8000, 1.9000, 2.0000, 0.5000, 0.6000, 0.7000, 0.8000]
-- softmaxRows on a small [2,3] tensor --
tensor(shape=[2,3])
[0.0900, 0.2447, 0.6652, 0.3333, 0.3333, 0.3333]
-- gelu on [-2,-1,0,1,2] --
tensor(shape=[5])
[-0.0454, -0.1588, 0.0000, 0.8412, 1.9546]
-- layerNorm(eps=1e-5, g=ones, b=zeros) on x --
tensor(shape=[3,4])
[-1.3411, -0.4470, 0.4470, 1.3411, -1.3411, -0.4470, 0.4470, 1.3411, -1.3411, -0.4470, 0.4470, 1.3411]
-- linear layer [4,2] + bias on x --
tensor(shape=[3,2])
[3.7000, 3.6000, 1.3000, 1.2000, 2.9000, 2.8000]
-- causal self-attention (nHeads=2) on x --
tensor(shape=[3,4])
[-0.0087, 0.0168, 0.0424, 0.0679, -0.0083, 0.0106, 0.0294, 0.0482, -0.0084, 0.0112, 0.0307, 0.0503]
-- full pre-LayerNorm transformer block on x --
tensor(shape=[3,4])
[1.5752, 1.7483, 1.9215, 2.0946, 0.3752, 0.5483, 0.7215, 0.8946, 1.1752, 1.3483, 1.5215, 1.6946]
```

## License

MIT

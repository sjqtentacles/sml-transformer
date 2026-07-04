(* transformer.sig -- transformer building blocks in pure Standard ML,
   layered on sml-tensor. The pieces a decoder-only (GPT-style) model needs
   that a general tensor library does not provide: embedding lookup, row
   slicing, a numerically-stable row-wise softmax, LayerNorm, GELU, an affine
   linear layer (HF Conv1D layout `y = x@W + b`), multi-head causal
   self-attention, and a full pre-LayerNorm transformer block.

   Reusable for any transformer (GPT-2/3, BERT, ...). All ops are pure and
   deterministic (compare reals with `Tensor.approxEq`, never `Real.toString`),
   so results are byte-identical under MLton and Poly/ML. *)

signature TRANSFORMER =
sig
  type t = Tensor.t

  (* embed table ids : gather rows of `table` (shape [vocab, d]) by the integer
     token ids, giving [length ids, d]. *)
  val embed : t -> int list -> t

  (* Rows [lo, hi) of a rank-2 tensor (axis 0). *)
  val sliceRows : int * int -> t -> t

  (* Numerically-stable softmax over the last axis of a rank-2 tensor (per row:
     subtract the row max before exponentiating). *)
  val softmaxRows : t -> t

  (* GELU, GPT-2's `gelu_new` tanh approximation, elementwise. *)
  val gelu : t -> t

  (* LayerNorm over the last axis (per row), with GPT-2's biased variance:
     `(x - mean)/sqrt(var + eps) * g + b`, where g, b are rank-1 [d]. *)
  val layerNorm : {eps : real, g : t, b : t} -> t -> t

  (* Affine linear layer, HF `Conv1D` convention: `y = x@w + b`, where x is
     [n, in], w is [in, out], b is rank-1 [out]. *)
  val linear : {w : t, b : t} -> t -> t

  (* Multi-head causal self-attention on x [seq, d]. Weights follow GPT-2:
     `wQkv` [d, 3d] / `bQkv` [3d] project to concatenated (q,k,v); each is split
     into `nHeads` heads of size d/nHeads; scaled dot-product with a causal mask
     and row softmax; heads are concatenated and projected by `wProj` [d,d] /
     `bProj` [d]. Returns [seq, d]. *)
  val attention :
      {nHeads : int, wQkv : t, bQkv : t, wProj : t, bProj : t} -> t -> t

  (* A full pre-LayerNorm transformer block:
       a = x + attention(layerNorm_1 x)
       y = a + linear_2(gelu(linear_1(layerNorm_2 a)))    (the MLP). *)
  type blockWeights =
    { eps : real, nHeads : int,
      ln1g : t, ln1b : t, wQkv : t, bQkv : t, wProj : t, bProj : t,
      ln2g : t, ln2b : t, wFc : t, bFc : t, wFcProj : t, bFcProj : t }
  val block : blockWeights -> t -> t
end

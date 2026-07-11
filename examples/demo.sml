(* demo.sml - run a tiny embedding table through softmax, GELU, LayerNorm,
   a linear layer, causal self-attention, and a full pre-LayerNorm
   transformer block. All weights are small literal/generated tensors (no
   pretrained data). Reals are printed via Tensor.toString/fmtReal (forced
   decimal, `-` not `~`), never Real.toString, so output is byte-identical
   under MLton and Poly/ML. *)

structure T = Tensor
structure Tr = Transformer

fun gen n f = List.tabulate (n, fn i => f (real i))

val () = print "=== sml-transformer demo ===\n\n"

(* A 5-token vocabulary, embedding dim d = 4. *)
val embedTable = T.fromRows [
  [0.1, 0.2, 0.3, 0.4],
  [0.5, 0.6, 0.7, 0.8],
  [0.9, 1.0, 1.1, 1.2],
  [1.3, 1.4, 1.5, 1.6],
  [1.7, 1.8, 1.9, 2.0]
]
val x = Tr.embed embedTable [4, 1, 3]   (* a 3-token sequence *)
val () = print "-- embed [4,1,3] over the vocab table, shape [3,4] --\n"
val () = print (T.toString 4 x)

val () = print "\n-- sliceRows (0,2): first two rows --\n"
val () = print (T.toString 4 (Tr.sliceRows (0, 2) x))

val () = print "\n-- softmaxRows on a small [2,3] tensor --\n"
val scores = T.fromRows [[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]
val () = print (T.toString 4 (Tr.softmaxRows scores))

val () = print "\n-- gelu on [-2,-1,0,1,2] --\n"
val () = print (T.toString 4 (Tr.gelu (T.fromList [5] [~2.0, ~1.0, 0.0, 1.0, 2.0])))

val () = print "\n-- layerNorm(eps=1e-5, g=ones, b=zeros) on x --\n"
val ln = Tr.layerNorm {eps = 1e~5, g = T.ones [4], b = T.zeros [4]} x
val () = print (T.toString 4 ln)

val () = print "\n-- linear layer [4,2] + bias on x --\n"
val w1 = T.fromRows [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]]
val b1 = T.fromList [2] [0.1, ~0.1]
val () = print (T.toString 4 (Tr.linear {w = w1, b = b1} x))

val () = print "\n-- causal self-attention (nHeads=2) on x --\n"
val attnOut = Tr.attention
  { nHeads = 2,
    wQkv = T.fromList [4, 12] (gen 48 (fn i => 0.01 * i - 0.24)),
    bQkv = T.zeros [12],
    wProj = T.fromList [4, 4] (gen 16 (fn i => 0.02 * i - 0.15)),
    bProj = T.zeros [4] } x
val () = print (T.toString 4 attnOut)

val () = print "\n-- full pre-LayerNorm transformer block on x --\n"
val blockOut = Tr.block
  { eps = 1e~5, nHeads = 2,
    ln1g = T.ones [4], ln1b = T.zeros [4],
    wQkv = T.fromList [4, 12] (gen 48 (fn i => 0.01 * i - 0.24)),
    bQkv = T.zeros [12],
    wProj = T.fromList [4, 4] (gen 16 (fn i => 0.02 * i - 0.15)),
    bProj = T.zeros [4],
    ln2g = T.ones [4], ln2b = T.zeros [4],
    wFc = T.fromList [4, 8] (gen 32 (fn i => 0.015 * i - 0.24)),
    bFc = T.zeros [8],
    wFcProj = T.fromList [8, 4] (gen 32 (fn i => 0.01 * i - 0.16)),
    bFcProj = T.zeros [4] } x
val () = print (T.toString 4 blockOut)

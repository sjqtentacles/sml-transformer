(* transformer.sml -- transformer building blocks on sml-tensor.

   Pure `+ - * /` plus Math.tanh/exp/sqrt (both compilers' libm), so results are
   byte-identical across MLton and Poly/ML. Attention/block follow GPT-2: HF
   Conv1D linear (y = x@W + b), split heads by columns, scaled dot-product with a
   causal mask, row softmax, concat heads, pre-LayerNorm residual, gelu_new MLP. *)

structure Transformer :> TRANSFORMER =
struct
  type t = Tensor.t

  fun mapi f xs =
      let fun go (_, []) = [] | go (i, x :: r) = f (i, x) :: go (i + 1, r)
      in go (0, xs) end

  fun embed table ids =
      let val rows = Tensor.toRows table
      in Tensor.fromRows (List.map (fn i => List.nth (rows, i)) ids) end

  fun sliceRows (lo, hi) x =
      Tensor.fromRows (List.take (List.drop (Tensor.toRows x, lo), hi - lo))

  (* numerically-stable softmax over each row *)
  fun softmaxRow xs =
      let val m = List.foldl Real.max (hd xs) (tl xs)
          val es = List.map (fn v => Math.exp (v - m)) xs
          val s = List.foldl op+ 0.0 es
      in List.map (fn e => e / s) es end
  fun softmaxRows x = Tensor.fromRows (List.map softmaxRow (Tensor.toRows x))

  (* GELU: GPT-2 gelu_new tanh approximation *)
  val geluC = Math.sqrt (2.0 / Math.pi)
  fun geluScalar v = 0.5 * v * (1.0 + Math.tanh (geluC * (v + 0.044715 * v * v * v)))
  fun gelu x = Tensor.map geluScalar x

  (* LayerNorm over the last axis, per row, biased variance *)
  fun zip3 (v :: vs, g :: gs, b :: bs) f = f (v, g, b) :: zip3 (vs, gs, bs) f
    | zip3 _ _ = []
  fun layerNorm {eps, g, b} x =
      let val gr = Tensor.toList g
          val br = Tensor.toList b
          fun normRow row =
              let val d  = real (length row)
                  val mu = (List.foldl op+ 0.0 row) / d
                  val vr = (List.foldl (fn (v, s) => s + (v - mu) * (v - mu)) 0.0 row) / d
                  val inv = 1.0 / Math.sqrt (vr + eps)
              in zip3 (row, gr, br) (fn (v, gi, bi) => (v - mu) * inv * gi + bi) end
      in Tensor.fromRows (List.map normRow (Tensor.toRows x)) end

  (* affine linear, HF Conv1D layout: y = x@w + b (b broadcasts over rows) *)
  fun linear {w, b} x = Tensor.add (Tensor.matmul x w) b

  (* column slice / horizontal concat on rank-2 tensors *)
  fun colSlice (lo, hi) x =
      Tensor.fromRows
        (List.map (fn row => List.take (List.drop (row, lo), hi - lo)) (Tensor.toRows x))
  fun hconcat ts =
      let val rls = List.map Tensor.toRows ts
          fun go rls = if List.all List.null rls then []
                       else List.concat (List.map hd rls) :: go (List.map tl rls)
      in Tensor.fromRows (go rls) end

  fun causalMask s =
      Tensor.fromRows
        (mapi (fn (i, row) => mapi (fn (j, v) => if j > i then ~1E10 else v) row)
              (Tensor.toRows s))

  fun attention {nHeads, wQkv, bQkv, wProj, bProj} x =
      let val C    = List.nth (Tensor.shape x, 1)
          val hdim = C div nHeads
          val qkv  = linear {w = wQkv, b = bQkv} x
          val q = colSlice (0, C) qkv
          val k = colSlice (C, 2 * C) qkv
          val v = colSlice (2 * C, 3 * C) qkv
          val sc = 1.0 / Math.sqrt (real hdim)
          fun head h =
              let val lo = h * hdim
                  val qh = colSlice (lo, lo + hdim) q
                  val kh = colSlice (lo, lo + hdim) k
                  val vh = colSlice (lo, lo + hdim) v
                  val scores = Tensor.scale sc (Tensor.matmul qh (Tensor.transpose kh))
                  val attn   = softmaxRows (causalMask scores)
              in Tensor.matmul attn vh end
      in linear {w = wProj, b = bProj} (hconcat (List.tabulate (nHeads, head))) end

  type blockWeights =
    { eps : real, nHeads : int,
      ln1g : t, ln1b : t, wQkv : t, bQkv : t, wProj : t, bProj : t,
      ln2g : t, ln2b : t, wFc : t, bFc : t, wFcProj : t, bFcProj : t }

  fun block {eps, nHeads, ln1g, ln1b, wQkv, bQkv, wProj, bProj,
             ln2g, ln2b, wFc, bFc, wFcProj, bFcProj} x =
      let val a = Tensor.add x
                    (attention {nHeads = nHeads, wQkv = wQkv, bQkv = bQkv,
                                wProj = wProj, bProj = bProj}
                               (layerNorm {eps = eps, g = ln1g, b = ln1b} x))
          val m = linear {w = wFcProj, b = bFcProj}
                    (gelu (linear {w = wFc, b = bFc}
                                  (layerNorm {eps = eps, g = ln2g, b = ln2b} a)))
      in Tensor.add a m end
end

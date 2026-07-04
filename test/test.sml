(* Tests for sml-transformer. Reference (golden) values are NumPy-derived
   (tools/gen_reference.py) and compared within an epsilon via Tensor.approxEq
   (never Real.toString), so the suite output is byte-identical across MLton and
   Poly/ML. Inputs are built with `grid` = NumPy's arange(n)*scale+off, so the
   SML and NumPy inputs are bit-for-bit the same and only the outputs are pasted. *)

structure Tests =
struct
  open Harness
  structure T = Tensor
  structure X = Transformer

  fun grid n scale off = List.tabulate (n, fn i => real i * scale + off)
  fun ck name (expected, actual) = check name (T.approxEq 1E~6 expected actual)

  fun run () =
    let
      val () = section "softmax (row-wise, stable)"
      val () = ck "softmax [[1,2,3]]"
                  (T.fromList [1,3] [0.09003057317, 0.2447284711, 0.6652409558],
                   X.softmaxRows (T.fromList [1,3] [1.0, 2.0, 3.0]))
      val () = ck "softmax [[1,2,3],[0,0,0]]"
                  (T.fromList [2,3] [0.09003057317, 0.2447284711, 0.6652409558,
                                     0.3333333333, 0.3333333333, 0.3333333333],
                   X.softmaxRows (T.fromList [2,3] [1.0,2.0,3.0, 0.0,0.0,0.0]))

      val () = section "gelu (gelu_new tanh approximation)"
      val () = ck "gelu [-2,-1,0,1,2]"
                  (T.fromList [5] [~0.04540230591, ~0.1588080094, 0.0, 0.8411919906, 1.954597694],
                   X.gelu (T.fromList [5] [~2.0, ~1.0, 0.0, 1.0, 2.0]))

      val () = section "layerNorm (biased variance)"
      val () = ck "layerNorm [[1,2,3,4]]"
                  (T.fromList [1,4] [~1.34163542, ~0.4472118067, 0.4472118067, 1.34163542],
                   X.layerNorm {eps=1E~5, g=T.ones [4], b=T.zeros [4]}
                               (T.fromList [1,4] [1.0, 2.0, 3.0, 4.0]))

      val () = section "linear (y = xW + b, Conv1D layout)"
      val () = ck "linear"
                  (T.fromList [1,3] [1.5, 2.5, 3.5],
                   X.linear {w = T.fromList [2,3] [1.0,0.0,1.0, 0.0,1.0,1.0],
                             b = T.fromList [3] [0.5, 0.5, 0.5]}
                            (T.fromList [1,2] [1.0, 2.0]))

      val () = section "embed + sliceRows"
      val tbl = T.fromRows [[1.0,2.0],[3.0,4.0],[5.0,6.0]]
      val () = ck "embed [2,0,2]"
                  (T.fromRows [[5.0,6.0],[1.0,2.0],[5.0,6.0]], X.embed tbl [2,0,2])
      val () = ck "sliceRows (1,3)"
                  (T.fromRows [[3.0,4.0],[5.0,6.0]], X.sliceRows (1,3) tbl)

      (* ---- attention & block: T=2, C=4, H=2; weights = grid (matches NumPy) ---- *)
      val x  = T.fromList [2,4]  (grid 8  0.1  ~0.1)
      val wq = T.fromList [4,12] (grid 48 0.01 ~0.05)
      val bq = T.fromList [12]   (grid 12 0.01 0.0)
      val wp = T.fromList [4,4]  (grid 16 0.02 ~0.03)
      val bp = T.fromList [4]    (grid 4  0.005 0.0)

      val () = section "multi-head causal self-attention"
      val () = ck "attention output [2,4]"
                  (T.fromList [2,4] [0.0768, 0.0978, 0.1188, 0.1398,
                                     0.1501820691, 0.1866048295, 0.2230275899, 0.2594503503],
                   X.attention {nHeads=2, wQkv=wq, bQkv=bq, wProj=wp, bProj=bp} x)

      val () = section "full transformer block (pre-LN residual)"
      val wfc = T.fromList [4,16]  (grid 64 0.01 ~0.1)
      val wfp = T.fromList [16,4]  (grid 64 0.01 ~0.2)
      val bw : X.blockWeights =
          { eps=1E~5, nHeads=2,
            ln1g=T.ones [4], ln1b=T.zeros [4], wQkv=wq, bQkv=bq, wProj=wp, bProj=bp,
            ln2g=T.ones [4], ln2b=T.zeros [4], wFc=wfc, bFc=T.zeros [16],
            wFcProj=wfp, bFcProj=T.zeros [4] }
      val () = ck "block output [2,4]"
                  (T.fromList [2,4] [1.004414169, 1.247239025, 1.49006388, 1.732888735,
                                     1.404414169, 1.647239025, 1.89006388, 2.132888735],
                   X.block bw x)
    in
      Harness.run ()
    end
end

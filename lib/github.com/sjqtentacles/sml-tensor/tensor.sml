(* tensor.sml

   N-dimensional dense arrays of reals. Storage is an immutable, contiguous,
   row-major `real vector` plus its integer shape. See tensor.sig for the
   contract. Everything is pure +,-,*,/ over the data, so results are
   byte-identical across MLton and Poly/ML (the only transcendental, the
   `Math.sqrt` inside `normFro`, defers to the same libm on both). *)

structure Tensor :> TENSOR =
struct
  exception Shape of string

  type t = { shape : int list, data : real vector }

  (* product of a dimension list (empty product = 1, so a scalar has numel 1) *)
  fun prod xs = List.foldl (op * ) 1 xs

  (* row-major strides: stride_j = product of the dimensions after axis j *)
  fun stridesOf s =
    let fun go [] = []
          | go (_ :: rest) = prod rest :: go rest
    in go s end

  (* coordinates of a flat offset, given the strides of its shape *)
  fun unravel (strides, flat) =
    let fun go ([], _) = []
          | go (st :: sts, f) = (f div st) :: go (sts, f mod st)
    in go (strides, flat) end

  fun insertAt (xs, 0, v) = v :: xs
    | insertAt (x :: xs, n, v) = x :: insertAt (xs, n - 1, v)
    | insertAt ([], n, v) = if n = 0 then [v] else raise Shape "insertAt: out of range"

  fun nub xs =
    let fun go ([], _) = []
          | go (x :: rest, seen) =
              if List.exists (fn y => y = x) seen then go (rest, seen)
              else x :: go (rest, x :: seen)
    in go (xs, []) end

  (* dot of strides with a coordinate list -> flat offset *)
  fun flatOf (strides, coords) =
    ListPair.foldlEq (fn (st, c, acc) => acc + st * c) 0 (strides, coords)

  (* ---- construction ---- *)

  fun build (s, v) =
    if prod s = Vector.length v then { shape = s, data = v }
    else raise Shape "data length does not match shape"

  fun fromList s xs = build (s, Vector.fromList xs)
  fun fromArray s a = build (s, Array.vector a)
  fun scalar x = { shape = [], data = Vector.fromList [x] }
  fun fill s x = { shape = s, data = Vector.tabulate (prod s, fn _ => x) }
  fun zeros s = fill s 0.0
  fun ones s = fill s 1.0
  fun fromRows rows =
    let
      val r = length rows
      val c = case rows of [] => 0 | (x :: _) => length x
      val () = if List.all (fn row => length row = c) rows then ()
               else raise Shape "fromRows: ragged rows"
    in fromList [r, c] (List.concat rows) end

  (* ---- introspection ---- *)

  fun shape ({ shape = s, ... } : t) = s
  fun rank t = length (shape t)
  fun numel ({ data, ... } : t) = Vector.length data
  fun toList ({ data, ... } : t) = Vector.foldr (op ::) [] data
  fun toArray ({ data, ... } : t) =
    Array.tabulate (Vector.length data, fn i => Vector.sub (data, i))

  fun toRows (t : t) =
    case shape t of
      [r, c] => List.tabulate (r, fn i =>
                  List.tabulate (c, fn j => Vector.sub (#data t, i * c + j)))
    | _ => raise Shape "toRows: not rank-2"

  fun sub (t : t) idx =
    let
      val s = shape t
      val () = if length idx = length s then ()
               else raise Shape "sub: index rank mismatch"
      val () = ListPair.appEq
                 (fn (i, d) => if i >= 0 andalso i < d then ()
                               else raise Shape "sub: index out of range")
                 (idx, s)
    in Vector.sub (#data t, flatOf (stridesOf s, idx)) end

  (* ---- shape transforms ---- *)

  fun reshape newshape (t : t) =
    if prod newshape = numel t then { shape = newshape, data = #data t }
    else raise Shape "reshape: size mismatch"

  fun flatten (t : t) = { shape = [numel t], data = #data t }

  fun permute perm (t : t) =
    let
      val s = shape t
      val n = length s
      val () = if length perm = n then () else raise Shape "permute: bad length"
      val () = if List.all (fn p => p >= 0 andalso p < n) perm
                  andalso length (nub perm) = n
               then () else raise Shape "permute: not a permutation"
      val newshape = List.map (fn p => List.nth (s, p)) perm
      val oldStrides = stridesOf s
      val newStrides = stridesOf newshape
      val total = prod s
      val data = #data t
      val out = Array.tabulate (total, fn oi =>
        let
          val newCoords = unravel (newStrides, oi)
          val oldArr = Array.array (n, 0)
          val () = ListPair.appEq
                     (fn (p, c) => Array.update (oldArr, p, c))
                     (perm, newCoords)
          val oldCoords = List.tabulate (n, fn i => Array.sub (oldArr, i))
        in Vector.sub (data, flatOf (oldStrides, oldCoords)) end)
    in { shape = newshape, data = Array.vector out } end

  fun transpose (t : t) =
    let val n = rank t
    in permute (List.tabulate (n, fn i => n - 1 - i)) t end

  fun broadcastTo target (t : t) =
    let
      val s = shape t
      val ns = length s and nt = length target
      val () = if nt >= ns then ()
               else raise Shape "broadcastTo: target rank too small"
      val padded = List.tabulate (nt - ns, fn _ => 1) @ s
      val () = ListPair.appEq
                 (fn (a, b) => if a = b orelse a = 1 then ()
                               else raise Shape "broadcastTo: incompatible shape")
                 (padded, target)
      (* a stretched (size-1) axis contributes stride 0 *)
      val effStrides =
        ListPair.mapEq (fn (st, d) => if d = 1 then 0 else st)
                       (stridesOf padded, padded)
      val tStrides = stridesOf target
      val total = prod target
      val data = #data t
      val out = Array.tabulate (total, fn oi =>
        let val coords = unravel (tStrides, oi)
        in Vector.sub (data, flatOf (effStrides, coords)) end)
    in { shape = target, data = Array.vector out } end

  (* ---- elementwise ---- *)

  fun map f (t : t) = { shape = #shape t, data = Vector.map f (#data t) }

  fun broadcastShape (a, b) =
    let
      val na = length a and nb = length b
      val n = Int.max (na, nb)
      val pa = List.tabulate (n - na, fn _ => 1) @ a
      val pb = List.tabulate (n - nb, fn _ => 1) @ b
    in
      ListPair.mapEq
        (fn (x, y) =>
            if x = y then x
            else if x = 1 then y
            else if y = 1 then x
            else raise Shape "broadcast: incompatible shapes")
        (pa, pb)
    end

  fun map2 f a b =
    let
      val bs = broadcastShape (shape a, shape b)
      val a' = broadcastTo bs a
      val b' = broadcastTo bs b
      val d = Vector.tabulate (prod bs, fn i =>
                f (Vector.sub (#data a', i), Vector.sub (#data b', i)))
    in { shape = bs, data = d } end

  fun add a b = map2 (op +) a b
  fun sub' a b = map2 (op -) a b
  fun mul a b = map2 (op * ) a b
  fun divide a b = map2 (op /) a b
  fun neg t = map (fn x => ~ x) t
  fun scale c t = map (fn x => c * x) t
  fun addScalar c t = map (fn x => x + c) t

  (* ---- reductions over all elements ---- *)

  fun sumAll ({ data, ... } : t) = Vector.foldl (op +) 0.0 data
  fun prodAll ({ data, ... } : t) = Vector.foldl (op * ) 1.0 data
  fun meanAll t = sumAll t / Real.fromInt (numel t)

  fun maxAll ({ data, ... } : t) =
    if Vector.length data = 0 then raise Shape "maxAll: empty tensor"
    else Vector.foldl Real.max (Vector.sub (data, 0)) data
  fun minAll ({ data, ... } : t) =
    if Vector.length data = 0 then raise Shape "minAll: empty tensor"
    else Vector.foldl Real.min (Vector.sub (data, 0)) data

  fun argmaxAll ({ data, ... } : t) =
    let
      val n = Vector.length data
      val () = if n = 0 then raise Shape "argmaxAll: empty tensor" else ()
      fun go (i, bi, bv) =
        if i = n then bi
        else let val v = Vector.sub (data, i)
             in if v > bv then go (i + 1, i, v) else go (i + 1, bi, bv) end
    in go (1, 0, Vector.sub (data, 0)) end

  (* ---- reductions along one axis ---- *)

  fun reduceAxisWith red axis (t : t) =
    let
      val s = shape t
      val n = length s
      val () = if axis >= 0 andalso axis < n then ()
               else raise Shape "axis out of range"
      val dim = List.nth (s, axis)
      val newshape = List.take (s, axis) @ List.drop (s, axis + 1)
      val strides = stridesOf s
      val newStrides = stridesOf newshape
      val total = prod newshape
      val data = #data t
      val out = Array.tabulate (total, fn oi =>
        let
          val coords = unravel (newStrides, oi)
          val vals = List.tabulate (dim, fn j =>
                       Vector.sub (data, flatOf (strides, insertAt (coords, axis, j))))
        in red vals end)
    in { shape = newshape, data = Array.vector out } end

  fun sum axis t = reduceAxisWith (List.foldl (op +) 0.0) axis t
  fun mean axis t =
    reduceAxisWith
      (fn vs => List.foldl (op +) 0.0 vs / Real.fromInt (length vs)) axis t
  fun maxAxis axis t =
    reduceAxisWith
      (fn [] => raise Shape "maxAxis: empty axis"
        | (v :: vs) => List.foldl Real.max v vs) axis t

  (* ---- linear algebra ---- *)

  fun matmul a b =
    case (shape a, shape b) of
      ([m, k], [k2, n]) =>
        if k <> k2 then raise Shape "matmul: inner dimension mismatch"
        else
          let
            val da = #data a and db = #data b
            val out = Array.array (m * n, 0.0)
            fun loopI i =
              if i = m then () else
              let
                fun loopJ j =
                  if j = n then () else
                  let
                    fun loopL (l, acc) =
                      if l = k then acc
                      else loopL (l + 1,
                            acc + Vector.sub (da, i * k + l) * Vector.sub (db, l * n + j))
                  in (Array.update (out, i * n + j, loopL (0, 0.0)); loopJ (j + 1)) end
              in (loopJ 0; loopI (i + 1)) end
            val () = loopI 0
          in { shape = [m, n], data = Array.vector out } end
    | ([k], [k2, n]) =>
        reshape [n] (matmul (reshape [1, k] a) b)
    | ([m, k], [k2]) =>
        reshape [m] (matmul a (reshape [k2, 1] b))
    | ([k], [k2]) =>
        if k <> k2 then raise Shape "matmul: vector length mismatch"
        else scalar (let
                       val da = #data a and db = #data b
                       fun go (i, acc) = if i = k then acc
                                         else go (i + 1, acc + Vector.sub (da, i) * Vector.sub (db, i))
                     in go (0, 0.0) end)
    | _ => raise Shape "matmul: unsupported ranks"

  fun dot a b =
    case (shape a, shape b) of
      ([n], [m]) =>
        if n <> m then raise Shape "dot: length mismatch"
        else sumAll (mul a b)
    | _ => raise Shape "dot: operands must be rank-1"

  fun normFro t = Math.sqrt (sumAll (mul t t))

  (* ---- einsum-lite ---- *)

  fun splitArrow s =
    let
      val n = String.size s
      fun find i =
        if i + 1 >= n then raise Shape "einsum: missing '->'"
        else if String.sub (s, i) = #"-" andalso String.sub (s, i + 1) = #">"
             then i else find (i + 1)
      val i = find 0
    in (String.substring (s, 0, i), String.extract (s, i + 2, NONE)) end

  fun einsum spec tensors =
    let
      val s = String.translate (fn c => if c = #" " then "" else str c) spec
      val (lhsStr, rhsStr) = splitArrow s
      val inSpecs = String.fields (fn c => c = #",") lhsStr
      val () = if length inSpecs = length tensors then ()
               else raise Shape "einsum: operand count mismatch"
      val inLabels = List.map explode inSpecs
      val outLabels = explode rhsStr
      val pairs = ListPair.zip (inLabels, tensors)
      val () = List.app
                 (fn (lbls, t) =>
                     if length lbls = rank t then ()
                     else raise Shape "einsum: operand rank mismatch") pairs

      (* label -> size, checking consistency across all occurrences *)
      val sizeTab = ref ([] : (char * int) list)
      fun setSize (c, sz) =
        case List.find (fn (c', _) => c' = c) (!sizeTab) of
          SOME (_, sz') => if sz = sz' then ()
                           else raise Shape "einsum: inconsistent index size"
        | NONE => sizeTab := (c, sz) :: (!sizeTab)
      val () = List.app
                 (fn (lbls, t) =>
                     ListPair.appEq (fn (c, d) => setSize (c, d)) (lbls, shape t))
                 pairs
      fun sizeOf c =
        case List.find (fn (c', _) => c' = c) (!sizeTab) of
          SOME (_, sz) => sz
        | NONE => raise Shape "einsum: output index absent from inputs"

      val lhsAll = nub (List.concat inLabels)
      val sumLabels =
        List.filter (fn c => not (List.exists (fn ol => ol = c) outLabels)) lhsAll
      val allLabels = outLabels @ sumLabels
      val nLabels = length allLabels
      fun labelIndex c =
        let fun go (_, []) = raise Shape "einsum: unknown label"
              | go (i, x :: xs) = if x = c then i else go (i + 1, xs)
        in go (0, allLabels) end
      val sizes = Vector.fromList (List.map sizeOf allLabels)

      val outShape = List.map sizeOf outLabels
      val outStrides = stridesOf outShape
      val outInfo = ListPair.zip (List.map labelIndex outLabels, outStrides)
      val inInfo =
        List.map
          (fn (lbls, t) =>
              (#data t,
               ListPair.zip (List.map labelIndex lbls, stridesOf (shape t))))
          pairs

      val out = Array.array (prod outShape, 0.0)
      val assign = Array.array (Int.max (nLabels, 1), 0)

      fun term () =
        List.foldl
          (fn ((data, info), acc) =>
              acc * Vector.sub (data,
                List.foldl (fn ((li, st), a) => a + st * Array.sub (assign, li)) 0 info))
          1.0 inInfo
      fun outPos () =
        List.foldl (fn ((li, st), a) => a + st * Array.sub (assign, li)) 0 outInfo
      fun loop k =
        if k = nLabels then
          let val p = outPos () in Array.update (out, p, Array.sub (out, p) + term ()) end
        else
          let
            val sz = Vector.sub (sizes, k)
            fun iter j =
              if j = sz then ()
              else (Array.update (assign, k, j); loop (k + 1); iter (j + 1))
          in iter 0 end
      val () = loop 0
    in { shape = outShape, data = Array.vector out } end

  (* ---- comparison & formatting ---- *)

  fun approxEq eps a b =
    shape a = shape b
    andalso
    let
      val da = #data a and db = #data b
      val n = Vector.length da
      fun go i =
        if i = n then true
        else if Real.abs (Vector.sub (da, i) - Vector.sub (db, i)) <= eps
        then go (i + 1) else false
    in go 0 end

  fun fmtReal n r =
    let val s = Real.fmt (StringCvt.FIX (SOME n)) r
    in if String.isPrefix "~" s then "-" ^ String.extract (s, 1, NONE) else s end

  fun toString n (t : t) =
    let
      val s = shape t
      val hdr = "tensor(shape=[" ^ String.concatWith "," (List.map Int.toString s) ^ "])\n"
      val body = "[" ^ String.concatWith ", " (List.map (fmtReal n) (toList t)) ^ "]"
    in hdr ^ body end
end

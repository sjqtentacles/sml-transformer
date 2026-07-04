(* tensor.sig

   N-dimensional dense arrays of reals in pure Standard ML (Basis only).

   A value of type `t` is an immutable, contiguous, row-major tensor: a
   `real vector` of data together with a `shape` (a list of non-negative
   dimension sizes). The rank is the length of the shape; a rank-0 tensor
   (shape `[]`) is a scalar holding a single element. The number of elements
   `numel` is the product of the shape (the empty product, 1, for a scalar).

   Indexing is zero-based and row-major: for shape `[d0, d1, ..., d{k-1}]` the
   element at multi-index `[i0, ..., i{k-1}]` lives at flat offset
   `sum_j i_j * stride_j`, where `stride_j` is the product of the dimensions
   after axis `j`.

   Conventions:
   - All operations are pure; tensors never mutate after construction.
   - Shape errors (size mismatches, bad axes, non-broadcastable operands,
     ragged rows, malformed `einsum` specs) raise `Shape` with a message.
   - Elementwise binary operations broadcast their operands using NumPy rules:
     shapes are right-aligned and, per axis, the sizes must be equal or one of
     them must be 1 (which is then stretched).
   - Reals are inexact: compare with `approxEq eps` and print through the
     forced-decimal `fmtReal` (always a decimal point, leading '-' not '~') so
     output is byte-identical across MLton and Poly/ML. *)

signature TENSOR =
sig
  type t

  exception Shape of string

  (* ---- construction ---- *)

  (* fromList shape data : build from a row-major element list. Raises Shape if
     `length data <> product shape`. *)
  val fromList  : int list -> real list -> t
  val fromArray : int list -> real array -> t
  (* A rank-0 scalar holding x. *)
  val scalar    : real -> t
  (* fill shape x : every element equal to x. *)
  val fill      : int list -> real -> t
  val zeros     : int list -> t
  val ones      : int list -> t
  (* 2-D convenience: build from equal-length rows (raises Shape on ragged). *)
  val fromRows  : real list list -> t

  (* ---- introspection ---- *)

  val shape   : t -> int list
  val rank    : t -> int
  val numel   : t -> int
  (* Flattened row-major contents. *)
  val toList  : t -> real list
  val toArray : t -> real array
  (* Rows of a rank-2 tensor; raises Shape otherwise. *)
  val toRows  : t -> real list list
  (* Element at a multi-index; raises Shape on rank/range errors. *)
  val sub     : t -> int list -> real

  (* ---- shape transformations ---- *)

  (* Reinterpret the same data under a new shape of equal numel. *)
  val reshape   : int list -> t -> t
  (* Collapse to a rank-1 tensor. *)
  val flatten   : t -> t
  (* Reverse all axes (generalised matrix transpose). *)
  val transpose : t -> t
  (* General axis permutation; `perm` must be a permutation of 0..rank-1. *)
  val permute   : int list -> t -> t
  (* Broadcast (read-only stretch) to a compatible larger shape. *)
  val broadcastTo : int list -> t -> t

  (* ---- elementwise ---- *)

  val map    : (real -> real) -> t -> t
  (* Broadcasting elementwise combine. *)
  val map2   : (real * real -> real) -> t -> t -> t
  val add    : t -> t -> t
  val sub'   : t -> t -> t
  val mul    : t -> t -> t
  val divide : t -> t -> t
  val neg    : t -> t
  val scale     : real -> t -> t
  val addScalar : real -> t -> t

  (* ---- reductions over all elements ---- *)

  val sumAll  : t -> real
  val prodAll : t -> real
  val maxAll  : t -> real
  val minAll  : t -> real
  val meanAll : t -> real
  (* Flat index of the (first) maximum element. *)
  val argmaxAll : t -> int

  (* ---- reductions along one axis (the axis is removed) ---- *)

  val sum     : int -> t -> t
  val mean    : int -> t -> t
  val maxAxis : int -> t -> t

  (* ---- linear algebra ---- *)

  (* Matrix product. Accepts 2-D*2-D, 1-D*2-D, 2-D*1-D and 1-D*1-D (the last
     gives a scalar). Raises Shape on incompatible inner dimensions. *)
  val matmul : t -> t -> t
  (* Dot product of two equal-length rank-1 tensors. *)
  val dot    : t -> t -> real
  (* Frobenius norm: sqrt of the sum of squares. *)
  val normFro : t -> real

  (* ---- einsum-lite ----

     A general Einstein-summation over any number of operands, written
     "<in0>,<in1>,...-><out>" with single-letter indices, e.g.
     "ij,jk->ik" (matmul), "ij->ji" (transpose), "ii->" (trace),
     "i,i->" (dot), "ij->" (sum all). Repeated indices not present in the
     output are summed. Spaces are ignored. *)
  val einsum : string -> t list -> t

  (* ---- comparison & formatting ---- *)

  (* Same shape and every element within `eps` (absolute). *)
  val approxEq : real -> t -> t -> bool
  (* Forced-decimal formatting: `fmtReal d x` always shows a decimal point with
     `d` fractional digits and a leading '-' (never '~') for negatives. *)
  val fmtReal  : int -> real -> string
  (* Human-readable multi-line rendering with `d` fractional digits. *)
  val toString : int -> t -> string
end

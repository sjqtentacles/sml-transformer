#!/usr/bin/env python3
# Reference values (the oracle) for sml-transformer's golden tests.
# Run:  python3 tools/gen_reference.py
# Uses only NumPy; prints SML-ready `real list` constants. The SML library must
# reproduce these within epsilon. Math matches GPT-2 (biased LayerNorm variance,
# gelu_new tanh approximation, HF Conv1D layout y = x@W + b, scaled causal attn).
import numpy as np
np.set_printoptions(precision=12, suppress=False)

def sml(a):
    a = np.asarray(a, dtype=np.float64).ravel()
    return "[" + ", ".join(("~%.10g" % -v) if v < 0 else ("%.10g" % v) for v in a) + "]"

def softmax(x, axis=-1):
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)

def gelu_new(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

def layernorm(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu)**2).mean(-1, keepdims=True)   # biased (GPT-2)
    return (x - mu) / np.sqrt(var + eps) * g + b

def linear(x, W, b):          # Conv1D: y = x @ W + b, W is [in,out]
    return x @ W + b

def attention(x, Wqkv, bqkv, Wproj, bproj, n_heads):
    T, C = x.shape
    hd = C // n_heads
    qkv = x @ Wqkv + bqkv
    q, k, v = np.split(qkv, 3, axis=-1)
    def heads(z): return z.reshape(T, n_heads, hd).transpose(1, 0, 2)  # [h,T,hd]
    q, k, v = heads(q), heads(k), heads(v)
    scores = q @ k.transpose(0, 2, 1) / np.sqrt(hd)                    # [h,T,T]
    mask = np.triu(np.ones((T, T)), 1).astype(bool)
    scores = np.where(mask[None], -1e10, scores)
    a = softmax(scores, -1) @ v                                       # [h,T,hd]
    a = a.transpose(1, 0, 2).reshape(T, C)
    return a @ Wproj + bproj

# ---- fixed tiny inputs (deterministic) ----
def grid(shape, scale=0.1, off=0.0):
    n = int(np.prod(shape))
    return (np.arange(n, dtype=np.float64) * scale + off).reshape(shape)

print("(* softmax *)")
print("softmax [1,2,3]      =", sml(softmax(np.array([1.,2.,3.]))))
print("softmax [[1,2,3],[0,0,0]] =", sml(softmax(np.array([[1.,2.,3.],[0.,0.,0.]]))))
print("(* gelu_new on [-2,-1,0,1,2] *)")
print(sml(gelu_new(np.array([-2.,-1.,0.,1.,2.]))))
print("(* layernorm x=[[1,2,3,4]] g=1 b=0 eps=1e-5 *)")
x = np.array([[1.,2.,3.,4.]]); g = np.ones(4); b = np.zeros(4)
print(sml(layernorm(x, g, b)))
print("(* linear x=[[1,2]] W=[[1,0,1],[0,1,1]] b=[.5,.5,.5] *)")
print(sml(linear(np.array([[1.,2.]]), np.array([[1.,0.,1.],[0.,1.,1.]]), np.array([.5,.5,.5]))))

# attention + block: T=2, C=4, heads=2 (hd=2)
T, C, H = 2, 4, 2
x  = grid((T, C), 0.1, -0.1)
Wq = grid((C, 3*C), 0.01, -0.05); bq = grid((3*C,), 0.01, 0.0)
Wp = grid((C, C), 0.02, -0.03);   bp = grid((C,), 0.005, 0.0)
print("(* attention T=2 C=4 H=2 -- inputs: x,Wqkv,bqkv,Wproj,bproj below *)")
print("x     =", sml(x))
print("Wqkv  =", sml(Wq)); print("bqkv  =", sml(bq))
print("Wproj =", sml(Wp)); print("bproj =", sml(bp))
print("ATTN_OUT =", sml(attention(x, Wq, bq, Wp, bp, H)))

# full block
ln1g=grid((C,),0.0,1.0); ln1b=grid((C,),0.0,0.0)
ln2g=grid((C,),0.0,1.0); ln2b=grid((C,),0.0,0.0)
Wfc=grid((C,4*C),0.01,-0.1); bfc=grid((4*C,),0.0,0.0)
Wfp=grid((4*C,C),0.01,-0.2); bfp=grid((C,),0.0,0.0)
def block(x):
    a = x + attention(layernorm(x, ln1g, ln1b), Wq, bq, Wp, bp, H)
    m = linear(gelu_new(linear(layernorm(a, ln2g, ln2b), Wfc, bfc)), Wfp, bfp)
    return a + m
print("(* block adds mlp weights Wfc,bfc,Wfcproj,bfcproj + ln1/ln2 gamma=1 beta=0 *)")
print("Wfc     =", sml(Wfc)); print("Wfcproj =", sml(Wfp))
print("BLOCK_OUT =", sml(block(x)))

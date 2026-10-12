"""Inference-only kernels. Preserve model-specific rounding and calibration."""

import torch
import triton
import triton.language as tl


@triton.jit
def _rms(
    X,
    W,
    Y,
    N: tl.constexpr,
    EPS: tl.constexpr,
    OFFSET: tl.constexpr,
    ROUND: tl.constexpr,
    B: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, B)
    x = tl.load(X + row * N + cols, cols < N, 0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 0) / N + EPS)
    y = x * inv
    if ROUND:
        y = y.to(X.dtype.element_ty).to(tl.float32)
    w = tl.load(W + cols, cols < N, 0).to(tl.float32) + OFFSET
    tl.store(Y + row * N + cols, y * w, cols < N)


def rms_norm(x, weight, eps, offset=0.0, round_before_weight=False):
    if x.stride(-1) != 1 or not x.is_contiguous():
        x = x.contiguous()
    out = torch.empty_like(x)
    n = x.shape[-1]
    _rms[(x.numel() // n,)](
        x,
        weight,
        out,
        n,
        eps,
        offset,
        round_before_weight,
        triton.next_power_of_2(n),
        enable_fp_fusion=False,
    )
    return out


@triton.jit
def _project(
    X,
    W,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr = 16,
    BN: tl.constexpr = 32,
    BK: tl.constexpr = 64,
):
    r = tl.program_id(0) * BM + tl.arange(0, BM)
    c = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for block in range(tl.cdiv(K, BK)):
        kk = block * BK + k
        a = tl.load(
            X + r[:, None] * K + kk[None, :], (r[:, None] < M) & (kk[None, :] < K), 0
        )
        b = tl.load(
            W + c[None, :] * K + kk[:, None], (c[None, :] < N) & (kk[:, None] < K), 0
        )
        acc += tl.dot(a, b)
    tl.store(Y + r[:, None] * N + c[None, :], acc, (r[:, None] < M) & (c[None, :] < N))


def candidate_projection(x, weight):
    shape = x.shape[:-1]
    n, k = weight.shape
    x = x.contiguous().view(-1, k)
    out = torch.empty((x.shape[0], n), device=x.device, dtype=x.dtype)
    _project[(triton.cdiv(x.shape[0], 16), triton.cdiv(n, 32))](
        x, weight, out, x.shape[0], n, k
    )
    return out.view(*shape, n)

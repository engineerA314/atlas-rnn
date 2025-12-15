from __future__ import annotations

import torch
from torch import Tensor


def _assoc_compose_affine(
    left: tuple[Tensor, Tensor],
    right: tuple[Tensor, Tensor],
) -> tuple[Tensor, Tensor]:
    """
    Compose two right-affine transforms over a row-state H:
      T(H) = H @ A + B

    If we have:
      T1(H) = H @ A1 + B1
      T2(H) = H @ A2 + B2

    Then:
      (T2 ∘ T1)(H) = (H @ A1 + B1) @ A2 + B2
                  = H @ (A1 @ A2) + (B1 @ A2 + B2)

    Returns (A12, B12) for the composed transform.

    Shapes:
      A*: [..., D, D]
      B*: [..., R, D]   (R is "row dimension" (e.g. dv))
    """
    A1, B1 = left
    A2, B2 = right
    A12 = A1 @ A2
    B12 = B1 @ A2 + B2
    return A12, B12


def affine_associative_scan(
    A: Tensor,
    B: Tensor,
    *,
    prev: Tensor | None = None,
    remove_prev: bool = False,
) -> Tensor:
    """
    Torch reference implementation of an associative scan for right-affine transforms.

    Computes prefix states for recurrence:
      H_t = H_{t-1} @ A_t + B_t

    by scanning *transforms* (A_t, B_t) using the associative composition above.

    Args:
      A: [N, T, D, D]
      B: [N, T, R, D]
      prev: optional initial state H_0 of shape [N, R, D]
      remove_prev: if True and prev is provided, drop the first output (aligned to original T)

    Returns:
      H: [N, T, R, D] (or [N, T+1, R, D] if prev is provided and remove_prev=False)

    Notes:
      - This is designed for correctness + CPU portability.
      - It is NOT the final accelerated CUDA implementation.
    """
    assert A.ndim == 4 and B.ndim == 4, "Expected A=[N,T,D,D], B=[N,T,R,D]"
    N, T, D, D2 = A.shape
    assert D == D2
    nB, tB, R, dB = B.shape
    assert (nB, tB, dB) == (N, T, D)

    if prev is not None:
        assert prev.shape == (N, R, D)

    # build list of transforms
    # transforms are tuples (A, B) each with batch N and a single time index
    A_cur = A
    B_cur = B

    # If prev is provided, we conceptually prepend an identity transform with B=0 and
    # later apply it to prev. We do it by carrying prev separately and concatenating
    # output states after scan.
    #
    # We'll compute prefix transforms for the T steps: (A_prefix[t], B_prefix[t])
    # such that:
    #   H_t = H_0 @ A_prefix[t] + B_prefix[t]
    # where A_prefix[t],B_prefix[t] correspond to composition T1..Tt.

    # Reduce/upsweep style associative scan for transforms.
    # We'll compute all prefix transforms by recursive pair reduction.
    # Implementation adapted from the structure of assoc_scan.associative_scan.

    def _interleave(a: Tensor, b: Tensor) -> Tensor:
        # a: [N, Ta, ...], b: [N, Tb, ...]
        Ta, Tb = a.shape[1], b.shape[1]
        output_axis_len = Ta + Tb
        if Ta == Tb + 1:
            pad = torch.zeros_like(b[:, :1])
            b = torch.cat([b, pad], dim=1)
        stacked = torch.stack([a, b], dim=2)  # [N, Ta, 2, ...]
        out = stacked.flatten(1, 2)
        return out[:, :output_axis_len]

    def _scan(Ae: Tensor, Be: Tensor) -> tuple[Tensor, Tensor]:
        # Ae: [N, T, D, D], Be: [N, T, R, D]
        t = Ae.shape[1]
        if t < 2:
            return Ae, Be

        # combine adjacent pairs: (0,1), (2,3), ...
        A_even = Ae[:, :-1:2]
        B_even = Be[:, :-1:2]
        A_odd = Ae[:, 1::2]
        B_odd = Be[:, 1::2]

        A_red, B_red = _assoc_compose_affine((A_even, B_even), (A_odd, B_odd))
        A_odd_pref, B_odd_pref = _scan(A_red, B_red)

        if t % 2 == 0:
            # even positions use odd prefix excluding last, composed with original even (skip first)
            A_left = A_odd_pref[:, :-1]
            B_left = B_odd_pref[:, :-1]
            A_right = Ae[:, 2::2]
            B_right = Be[:, 2::2]
            A_even_pref, B_even_pref = _assoc_compose_affine((A_left, B_left), (A_right, B_right))
        else:
            A_left = A_odd_pref
            B_left = B_odd_pref
            A_right = Ae[:, 2::2]
            B_right = Be[:, 2::2]
            A_even_pref, B_even_pref = _assoc_compose_affine((A_left, B_left), (A_right, B_right))

        # first element is identity composition -> equals first transform
        A_even_pref = torch.cat([Ae[:, :1], A_even_pref], dim=1)
        B_even_pref = torch.cat([Be[:, :1], B_even_pref], dim=1)

        A_out = _interleave(A_even_pref, A_odd_pref)
        B_out = _interleave(B_even_pref, B_odd_pref)
        return A_out, B_out

    A_pref, B_pref = _scan(A_cur, B_cur)

    # apply to prev to obtain states
    if prev is None:
        H0 = torch.zeros(N, R, D, device=A.device, dtype=A.dtype)
    else:
        H0 = prev

    # H_t = H0 @ A_pref[t] + B_pref[t]
    H = torch.einsum("nrd,ntde->nrte", H0, A_pref)  # [N, R, T, D]
    H = H.permute(0, 2, 1, 3).contiguous()          # [N, T, R, D]
    H = H + B_pref

    if prev is not None and not remove_prev:
        H = torch.cat([prev[:, None], H], dim=1)
    if prev is not None and remove_prev:
        # align to original T
        H = H[:, :T]

    return H



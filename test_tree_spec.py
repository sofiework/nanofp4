"""Unit tests for tree speculative decoding (CPU: tree structure + accept walk; GPU: attention oracle).
Run: python test_tree_spec.py
"""
import torch
from nanovllm.utils.spec_decoding.tree import build_spec_tree, walk_accept


def test_tree_structure():
    t = build_spec_tree("[(0,),(1,),(0,0),(0,1),(1,0),(1,1)]", 0)
    assert t.num_nodes == 7
    assert t.parent == [-1, 0, 0, 1, 1, 2, 2]
    assert t.depth == [0, 1, 1, 2, 2, 2, 2]
    assert t.levels == [[0], [1, 2], [3, 4, 5, 6]]
    assert [j for j in range(7) if t.mask[3, j]] == [0, 1, 3]   # C attends root, A, self
    assert [j for j in range(7) if t.mask[6, j]] == [0, 2, 6]   # F attends root, B, self
    # chain == causal
    c = build_spec_tree("", 3)
    assert c.num_nodes == 4 and c.parent == [-1, 0, 1, 2]
    assert torch.equal(c.mask, torch.tril(torch.ones(4, 4, dtype=torch.bool)))
    print("ok: tree structure")


def test_accept_walk():
    #        root0
    #       /     \
    #     A1        B2
    #    /  \      /  \
    #  C3    D4  E5    F6
    t = build_spec_tree("[(0,),(1,),(0,0),(0,1),(1,0),(1,1)]", 0)
    # draft tokens per node (node 0 = last_token anchor, ignored by walk)
    drafts = [999, 11, 22, 33, 44, 55, 66]

    # (a) full-depth accept down the left branch: target picks A1 then C3, bonus after C3
    argmax = [11, 33, 0, 77, 0, 0, 0]   # root->11(=A1), A1->33(=C3), C3->77(bonus)
    path, committed = walk_accept(t, argmax, drafts)
    assert path == [0, 1, 3] and committed == [11, 33, 77], (path, committed)

    # (b) mismatch at root: target's argmax matches no child -> only bonus token
    argmax = [12, 0, 0, 0, 0, 0, 0]
    path, committed = walk_accept(t, argmax, drafts)
    assert path == [0] and committed == [12], (path, committed)

    # (c) right branch, accept B2 then F6
    argmax = [22, 0, 66, 0, 0, 0, 88]   # root->22(=B2), B2->66(=F6), F6->88(bonus)
    path, committed = walk_accept(t, argmax, drafts)
    assert path == [0, 2, 6] and committed == [22, 66, 88], (path, committed)

    # (d) accept one level then mismatch (target picks A1, but A1's argmax matches no child)
    argmax = [11, 99, 0, 0, 0, 0, 0]
    path, committed = walk_accept(t, argmax, drafts)
    assert path == [0, 1] and committed == [11, 99], (path, committed)

    # chain reduces to longest-prefix + bonus
    ch = build_spec_tree("", 3)                 # nodes 0..3, chain
    cd = [999, 1, 2, 3]
    p, c = walk_accept(ch, [1, 2, 9, 0], cd)    # accept d1,d2; mismatch at d3 -> bonus 9
    assert p == [0, 1, 2] and c == [1, 2, 9], (p, c)
    print("ok: accept walk")


def test_tree_attention_oracle():
    if not torch.cuda.is_available():
        print("skip: tree attention oracle (no cuda)")
        return
    from flash_attn import flash_attn_with_kvcache
    torch.manual_seed(0)
    dev, dt = "cuda", torch.bfloat16
    S, N, H, Hkv, D, bs = 3, 7, 16, 8, 128, 256
    scale = D ** -0.5
    t = build_spec_tree("[(0,),(1,),(0,0),(0,1),(1,0),(1,1)]", 0)
    bias = t.additive_bias(dev)
    ctx_lens = torch.tensor([300, 130, 500], dtype=torch.int32, device=dev)

    q = torch.randn(S, N, H, D, device=dev, dtype=dt)
    k_tree = torch.randn(S, N, Hkv, D, device=dev, dtype=dt)
    v_tree = torch.randn(S, N, Hkv, D, device=dev, dtype=dt)
    max_blocks = (int(ctx_lens.max()) + bs - 1) // bs
    k_cache = torch.randn(S * max_blocks + 1, bs, Hkv, D, device=dev, dtype=dt)
    v_cache = torch.randn(S * max_blocks + 1, bs, Hkv, D, device=dev, dtype=dt)
    block_table = (torch.arange(1, S * max_blocks + 1, device=dev, dtype=torch.int32).reshape(S, max_blocks))

    exp = lambda x: x.repeat_interleave(H // Hkv, dim=-2)
    # reference: full fp32 dense over context ++ tree
    ref = torch.zeros(S, N, H, D, device=dev)
    for s in range(S):
        c = int(ctx_lens[s])
        kc = torch.stack([k_cache[int(block_table[s, p // bs]), p % bs] for p in range(c)])
        vc = torch.stack([v_cache[int(block_table[s, p // bs]), p % bs] for p in range(c)])
        kk = exp(torch.cat([kc, k_tree[s]], 0)).float()
        vv = exp(torch.cat([vc, v_tree[s]], 0)).float()
        sc = torch.einsum("nhd,khd->hnk", q[s].float(), kk) * scale
        b = torch.zeros(N, c + N, device=dev); b[:, c:] = bias
        ref[s] = torch.einsum("hnk,khd->nhd", (sc + b[None]).softmax(-1), vv)

    # method under test == Attention.tree_forward's math
    pre_out, pre_lse = flash_attn_with_kvcache(q, k_cache, v_cache, cache_seqlens=ctx_lens,
                                               block_table=block_table, softmax_scale=scale,
                                               causal=False, return_softmax_lse=True)
    kt, vt = exp(k_tree).float(), exp(v_tree).float()
    sc = torch.einsum("snhd,smhd->shnm", q.float(), kt) * scale + bias[None, None]
    suf_lse = torch.logsumexp(sc, -1)
    suf_out = torch.einsum("shnm,smhd->snhd", sc.softmax(-1), vt)
    pl, sl = pre_lse.transpose(1, 2).float(), suf_lse.transpose(1, 2).float()
    m = torch.maximum(pl, sl); p = torch.exp(pl - m); s = torch.exp(sl - m)
    out = (pre_out.float() * p[..., None] + suf_out * s[..., None]) / (p + s)[..., None]
    err = (out - ref).abs().max().item()
    assert err < 1e-2, err
    print(f"ok: tree attention oracle (max abs err {err:.2e} vs fp32 dense)")


if __name__ == "__main__":
    test_tree_structure()
    test_accept_walk()
    test_tree_attention_oracle()
    print("ALL PASS")

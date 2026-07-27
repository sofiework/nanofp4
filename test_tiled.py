"""Validate the query-tiled paged-attn kernel vs (A) the per-row reference kernel [fp4+fp8 scale] and
(B) torch SDPA ground truth [bf16]. Run on the 4090."""
import torch, torch.nn.functional as F
import nanovllm.layers.paged_attn as pa
from nanovllm.layers.paged_attn import paged_attn, store_kv_quant
torch.manual_seed(0)
dev = "cuda"
D, KVH, H, BLOCK = 128, 8, 64, 256
G = H // KVH
CTX = 300                      # one seq, 300 tokens, full prefill (Nq=300 query rows)
nb = (CTX + BLOCK - 1) // BLOCK
sm = 1.0 / (D ** 0.5)

k = (torch.randn(CTX, KVH, D, device=dev, dtype=torch.bfloat16) * 0.5)
v = (torch.randn(CTX, KVH, D, device=dev, dtype=torch.bfloat16) * 0.5)
q = (torch.randn(CTX, H, D, device=dev, dtype=torch.bfloat16) * 0.5)
block_tables = torch.arange(nb, dtype=torch.int32, device=dev).view(1, nb)
q_seq = torch.zeros(CTX, dtype=torch.int32, device=dev)
q_ctx = torch.arange(1, CTX + 1, dtype=torch.int32, device=dev)   # causal length per row


def ref_sdpa():
    # ground truth: expand GQA, causal SDPA in fp32 on the ORIGINAL bf16 k,v
    kf = k.float().repeat_interleave(G, dim=1).permute(1, 0, 2)   # [H, CTX, D]
    vf = v.float().repeat_interleave(G, dim=1).permute(1, 0, 2)
    qf = q.float().permute(1, 0, 2)                               # [H, CTX, D]
    o = F.scaled_dot_product_attention(qf, kf, vf, is_causal=True, scale=sm)
    return o.permute(1, 0, 2)                                     # [CTX, H, D]


def run(fmt, scale_fp8):
    if fmt == "fp4":
        cache = torch.zeros(2, 1, nb, BLOCK, KVH, D // 2, dtype=torch.uint8, device=dev)
        sdt = torch.float8_e4m3fn if scale_fp8 else torch.float32
        scale = torch.zeros(2, 1, nb, BLOCK, KVH, D // 16, dtype=sdt, device=dev)
        kc, vc, ks, vs = cache[0, 0], cache[1, 0], scale[0, 0], scale[1, 0]
        slot = torch.arange(CTX, dtype=torch.int32, device=dev)
        store_kv_quant(k, v, kc.view(-1, KVH, D // 2), vc.view(-1, KVH, D // 2),
                       ks.view(-1, KVH, D // 16), vs.view(-1, KVH, D // 16), slot, "fp4")
    else:
        kc = torch.zeros(nb, BLOCK, KVH, D, dtype=torch.bfloat16, device=dev)
        vc = torch.zeros(nb, BLOCK, KVH, D, dtype=torch.bfloat16, device=dev)
        kc.view(-1, KVH, D)[:CTX] = k
        vc.view(-1, KVH, D)[:CTX] = v
        ks = vs = None
    row = paged_attn(q, kc, vc, block_tables, q_seq, q_ctx, sm, fmt, ks, vs, prefill=False)
    tiled = paged_attn(q, kc, vc, block_tables, q_seq, q_ctx, sm, fmt, ks, vs, prefill=True)
    return row.float(), tiled.float()


def rel(a, b):
    return ((a - b).abs().max().item(), (a - b).abs().mean().item())

gt = ref_sdpa()
# B: bf16 tiled vs SDPA + row vs SDPA
row0, tiled0 = run("", False)
print(f"[bf16]  tiled-vs-SDPA  max={rel(tiled0, gt)[0]:.4e} mean={rel(tiled0, gt)[1]:.4e}")
print(f"[bf16]  row-vs-SDPA    max={rel(row0, gt)[0]:.4e} mean={rel(row0, gt)[1]:.4e}")
print(f"[bf16]  tiled-vs-row   max={rel(tiled0, row0)[0]:.4e}")
# A: fp4 fp8-scale tiled vs row (both read the same quantized cache)
row4, tiled4 = run("fp4", True)
print(f"[fp4/fp8scale] tiled-vs-row max={rel(tiled4, row4)[0]:.4e} mean={rel(tiled4, row4)[1]:.4e}")
row4b, tiled4b = run("fp4", False)
print(f"[fp4/fp32scale] tiled-vs-row max={rel(tiled4b, row4b)[0]:.4e} mean={rel(tiled4b, row4b)[1]:.4e}")


# ---- decode split-KV validation: 1 new query attends to all CTX cached tokens (non-causal) ----
def decode_ref(q_dec):
    kf = k.float().repeat_interleave(G, dim=1).permute(1, 0, 2)
    vf = v.float().repeat_interleave(G, dim=1).permute(1, 0, 2)
    qf = q_dec.float().permute(1, 0, 2)
    return F.scaled_dot_product_attention(qf, kf, vf, is_causal=False, scale=sm).permute(1, 0, 2)


def decode_run(fmt, scale_fp8, split):
    if fmt == "fp4":
        cache = torch.zeros(2, 1, nb, BLOCK, KVH, D // 2, dtype=torch.uint8, device=dev)
        sdt = torch.float8_e4m3fn if scale_fp8 else torch.float32
        sc = torch.zeros(2, 1, nb, BLOCK, KVH, D // 16, dtype=sdt, device=dev)
        kc, vc, ks, vs = cache[0, 0], cache[1, 0], sc[0, 0], sc[1, 0]
        slot = torch.arange(CTX, dtype=torch.int32, device=dev)
        store_kv_quant(k, v, kc.view(-1, KVH, D // 2), vc.view(-1, KVH, D // 2),
                       ks.view(-1, KVH, D // 16), vs.view(-1, KVH, D // 16), slot, "fp4")
    else:
        kc = torch.zeros(nb, BLOCK, KVH, D, dtype=torch.bfloat16, device=dev)
        vc = torch.zeros(nb, BLOCK, KVH, D, dtype=torch.bfloat16, device=dev)
        kc.view(-1, KVH, D)[:CTX] = k; vc.view(-1, KVH, D)[:CTX] = v
        ks = vs = None
    qd = torch.randn(1, H, D, device=dev, dtype=torch.bfloat16, generator=None) * 0.5
    qseq = torch.zeros(1, dtype=torch.int32, device=dev)
    qctx = torch.tensor([CTX], dtype=torch.int32, device=dev)
    pa._DECODE_SPLIT = split
    o = paged_attn(qd, kc, vc, block_tables, qseq, qctx, sm, fmt, ks, vs, prefill=False)
    return qd, o.float()

torch.manual_seed(1)
qd, o_row = decode_run("", False, 1)      # per-row reference (bf16)
_,  o_spl = decode_run("", False, 8)      # split-KV (bf16) — same seed -> same qd? re-seed:
torch.manual_seed(1); qd_b, o_row_b = decode_run("", False, 1)
torch.manual_seed(1); qd_s, o_spl_b = decode_run("", False, 8)
gt_d = decode_ref(qd_b)
print(f"[decode bf16] split-vs-row max={rel(o_spl_b, o_row_b)[0]:.4e}  split-vs-SDPA max={rel(o_spl_b, gt_d)[0]:.4e}")
torch.manual_seed(2); _, o_row4 = decode_run("fp4", True, 1)
torch.manual_seed(2); _, o_spl4 = decode_run("fp4", True, 8)
print(f"[decode fp4/fp8scale] split-vs-row max={rel(o_spl4, o_row4)[0]:.4e} mean={rel(o_spl4, o_row4)[1]:.4e}")

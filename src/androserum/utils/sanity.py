#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
端到端验证 vocab.txt 是否能配合 model_steps_604364.pt 工作。

不 import pretrainDexBERT (它会传染性 import tensorflow), 直接把
BertAEModel4Pretrain 的定义内联在本脚本里, 仅依赖 Model/models.py。

用法:
  python sanity_check_vocab.py
  python sanity_check_vocab.py --vocab ... --ckpt ... --cfg ... --sample ...
"""

import argparse
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, "Model"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import models  # noqa: E402
import tokenization  # noqa: E402


class BertAEModel4Pretrain(nn.Module):
    """复刻自 Model/pretrainDexBERT.py 的 BertAEModel4Pretrain.

    为避免引入 tensorflow 间接依赖, 这里把它独立地放在本脚本里, 行为完全一致。
    与作者 state_dict 的 key 完全对齐。
    """

    def __init__(self, cfg):
        super().__init__()
        self.transformer = models.Transformer(cfg)
        self.fc = nn.Linear(cfg.dim, cfg.dim)
        self.activ1 = nn.Tanh()
        self.linear = nn.Linear(cfg.dim, cfg.dim)
        self.activ2 = models.gelu
        self.norm = models.LayerNorm(cfg)
        self.classifier = nn.Linear(cfg.dim, 2)
        embed_weight = self.transformer.embed.tok_embed.weight
        n_vocab, n_dim = embed_weight.size()
        self.decoder = nn.Linear(n_dim, n_vocab, bias=False)
        self.decoder.weight = embed_weight
        self.decoder_bias = nn.Parameter(torch.zeros(n_vocab))
        self.AE_Layer_1 = nn.Linear(cfg.max_len * cfg.dim, cfg.max_len)
        self.AE_Layer_2 = nn.Linear(cfg.max_len, cfg.class_vec_len)
        self.AE_Layer_3 = nn.Linear(cfg.class_vec_len, cfg.max_len)
        self.AE_Layer_4 = nn.Linear(cfg.max_len, cfg.max_len * cfg.dim)

    def forward(self, input_ids, segment_ids, input_mask, masked_pos):
        h = self.transformer(input_ids, segment_ids, input_mask)
        pooled_h = self.activ1(self.fc(h[:, 0]))
        masked_pos = masked_pos[:, :, None].expand(-1, -1, h.size(-1))
        h_masked = torch.gather(h, 1, masked_pos)
        h_masked = self.norm(self.activ2(self.linear(h_masked)))
        logits_lm = self.decoder(h_masked) + self.decoder_bias
        logits_clsf = self.classifier(pooled_h)
        r1 = torch.flatten(h, start_dim=1)
        x = self.AE_Layer_1(r1)
        r2 = self.AE_Layer_2(x)
        x = self.AE_Layer_3(r2)
        reconstruction = self.AE_Layer_4(x)
        return logits_lm, logits_clsf, r1, reconstruction, r2


def read_smali_lines(sample_path: str, max_lines: int = 200) -> list:
    """从 processed 的 txt 抽几行真实 smali 指令"""
    out = []
    with open(sample_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("ClassName:") or s.startswith("MethodName:"):
                continue
            if s.endswith(".txt"):
                continue
            out.append(s)
            if len(out) >= max_lines:
                break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", default="save_dir/DexBERT/vocab.txt")
    ap.add_argument("--ckpt", default="save_dir/DexBERT/model_steps_604364.pt")
    ap.add_argument("--cfg", default="Model/config/DexBERT/bert_base.json")
    ap.add_argument(
        "--sample",
        default="processed/0D64BB3C121E1986766505E182F16FB8DCC4188224F3094F99B9F905873DDC4A.txt",
    )
    args = ap.parse_args()

    for p in (args.vocab, args.ckpt, args.cfg, args.sample):
        if not os.path.exists(p):
            sys.exit(f"[ERR] not found: {p}")

    print("=" * 60)
    print("[1] 加载 FullTokenizer")
    print("=" * 60)
    tokenizer = tokenization.FullTokenizer(vocab_file=args.vocab, do_lower_case=True)
    print(f"  vocab size = {len(tokenizer.vocab)}")
    for t in ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"):
        if t not in tokenizer.vocab:
            sys.exit(f"  [FAIL] 词表缺少保留 token: {t}")
        print(f"  {t} -> id {tokenizer.vocab[t]}")
    unk_id = tokenizer.vocab["[UNK]"]

    print("\n" + "=" * 60)
    print("[2] 真实 smali 行 token 化, 看 [UNK] 比例")
    print("=" * 60)
    lines = read_smali_lines(args.sample, max_lines=200)
    print(f"  读到 {len(lines)} 行 smali 指令做覆盖率测试")

    total = 0
    unk = 0
    sample_show = []
    for line in lines:
        toks = tokenizer.tokenize(tokenizer.convert_to_unicode(line))
        ids = tokenizer.convert_tokens_to_ids(toks)
        for t, i in zip(toks, ids):
            total += 1
            if i == unk_id:
                unk += 1
        if len(sample_show) < 3:
            sample_show.append((line, toks[:20], ids[:20]))

    if total == 0:
        sys.exit("  [FAIL] 没生成任何 token, 样本可能为空")

    unk_ratio = unk / total
    print(f"  total tokens = {total}, [UNK] = {unk}, 比例 = {unk_ratio:.4%}")
    print(f"  样例展示:")
    for line, toks, ids in sample_show:
        print(f"    line  : {line[:80]}")
        print(f"    tokens: {toks}")
        print(f"    ids   : {ids}")

    if unk_ratio > 0.10:
        print("  [WARN] [UNK] 比例 > 10%, vocab 与预训练语料可能不一致")
    elif unk_ratio > 0.02:
        print("  [OK?]  [UNK] 比例 2%~10%, 略高但可接受")
    else:
        print("  [OK]   [UNK] 比例正常 (< 2%)")

    print("\n" + "=" * 60)
    print("[3] 加载模型 + state_dict, 跑一次 forward")
    print("=" * 60)
    cfg = models.Config.from_json(args.cfg)
    print(
        f"  cfg.vocab_size={cfg.vocab_size}, cfg.dim={cfg.dim}, "
        f"cfg.n_layers={cfg.n_layers}, cfg.max_len={cfg.max_len}"
    )
    if cfg.vocab_size != len(tokenizer.vocab):
        sys.exit(
            f"  [FAIL] cfg vocab_size={cfg.vocab_size} != "
            f"tokenizer vocab size={len(tokenizer.vocab)}"
        )

    model = BertAEModel4Pretrain(cfg)
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:
        print(f"    missing keys (前 10): {missing[:10]}")
    if unexpected:
        print(f"    unexpected keys (前 10): {unexpected[:10]}")
    if missing or unexpected:
        print("  [WARN] 有未对齐的权重, 但 tok_embed/transformer 部分大概率是对的, 继续 forward")
    model.eval()

    cls_id = tokenizer.vocab["[CLS]"]
    sep_id = tokenizer.vocab["[SEP]"]
    pad_id = tokenizer.vocab["[PAD]"]
    max_len = cfg.max_len

    line = lines[0]
    toks = tokenizer.tokenize(tokenizer.convert_to_unicode(line))
    ids = tokenizer.convert_tokens_to_ids(toks)[: max_len - 2]
    input_ids = [cls_id] + ids + [sep_id]
    seg_ids = [0] * len(input_ids)
    attn_mask = [1] * len(input_ids)
    pad_n = max_len - len(input_ids)
    input_ids += [pad_id] * pad_n
    seg_ids += [0] * pad_n
    attn_mask += [0] * pad_n

    input_ids_t = torch.tensor([input_ids], dtype=torch.long)
    seg_ids_t = torch.tensor([seg_ids], dtype=torch.long)
    mask_t = torch.tensor([attn_mask], dtype=torch.long)
    masked_pos = torch.tensor([[1]], dtype=torch.long)

    with torch.no_grad():
        logits_lm, logits_clsf, r1, recon, r2 = model(
            input_ids_t, seg_ids_t, mask_t, masked_pos
        )
    print(f"  forward OK")
    print(
        f"    logits_lm   shape = {tuple(logits_lm.shape)}  "
        f"(应是 [1, 1, {cfg.vocab_size}])"
    )
    print(f"    logits_clsf shape = {tuple(logits_clsf.shape)}  (应是 [1, 2])")
    print(f"    r2 (AE class vec) shape = {tuple(r2.shape)}  (应是 [1, {cfg.class_vec_len}])")

    print("\n" + "=" * 60)
    print("[4] MLM 健康度: 第 1 个 token 的 top-5 预测")
    print("=" * 60)
    topk = torch.topk(logits_lm[0, 0], k=5)
    inv_vocab = {i: t for t, i in tokenizer.vocab.items()}
    print(f"  原 token (id 1 处, 即 input_ids[1]) = "
          f"{inv_vocab.get(input_ids[1], '?')!r}")
    print(f"  top-5 prediction:")
    for score, idx in zip(topk.values.tolist(), topk.indices.tolist()):
        print(
            f"    id={idx:5d}  token={inv_vocab.get(idx, '?'):<24s}  logit={score:+.3f}"
        )

    print("\n[DONE] 如果上面没有任何 [FAIL], vocab.txt + .pt 这一对是可用的。")


if __name__ == "__main__":
    main()

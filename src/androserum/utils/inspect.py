#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查 DexBERT 预训练 checkpoint 的内容:
  1) 顶层是否是 state_dict
  2) token embedding 形状是否与 bert_base.json (vocab_size=9537, dim=768) 匹配
  3) 是否额外包含 vocab / tokenizer 信息

用法:
  python inspect_checkpoint.py save_dir/DexBERT/model_steps_604364.pt
"""

import argparse
import os
import sys
from typing import Any

import torch


def is_tensor_dict(obj: Any) -> bool:
    return isinstance(obj, dict) and all(
        isinstance(v, torch.Tensor) for v in obj.values()
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", help=".pt 路径")
    args = ap.parse_args()

    if not os.path.exists(args.ckpt):
        sys.exit(f"[ERR] 找不到文件: {args.ckpt}")

    print(f"[INFO] Loading {args.ckpt} ...")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    print(f"[INFO] top-level type = {type(ckpt).__name__}")

    state_dict = None
    other_payload = {}

    if isinstance(ckpt, dict):
        if is_tensor_dict(ckpt):
            state_dict = ckpt
            print("[INFO] 顶层就是纯 tensor 字典 (state_dict)")
        else:
            print("[INFO] 顶层是 dict, 但不全是 tensor. 顶层 keys:")
            for k, v in ckpt.items():
                vt = type(v).__name__
                size_hint = ""
                if isinstance(v, torch.Tensor):
                    size_hint = f", shape={tuple(v.shape)}"
                elif hasattr(v, "__len__"):
                    try:
                        size_hint = f", len={len(v)}"
                    except Exception:
                        pass
                print(f"  - {k!r}: {vt}{size_hint}")
                if isinstance(v, dict) and is_tensor_dict(v):
                    state_dict = v
                    print(f"    -> 这看起来是 state_dict, 用它")
                else:
                    other_payload[k] = v
    else:
        sys.exit(f"[ERR] 顶层既不是 state_dict 也不是 dict: {type(ckpt)}")

    if state_dict is None:
        sys.exit("[ERR] 没找到 state_dict")

    print(f"\n[INFO] state_dict 共 {len(state_dict)} 个张量, 前 30 个 key:")
    for k in list(state_dict.keys())[:30]:
        print(f"  {k:<60s} {tuple(state_dict[k].shape)}")

    embed_keys = [k for k in state_dict if "tok_embed" in k or "embeddings.word" in k]
    if embed_keys:
        print(f"\n[INFO] 找到 token embedding 权重:")
        for k in embed_keys:
            shape = tuple(state_dict[k].shape)
            print(f"  {k}: {shape}")
            if len(shape) == 2:
                print(
                    f"  -> 实际词表大小 (vocab_size) = {shape[0]}, 隐层维度 = {shape[1]}"
                )
                if shape[0] == 9537:
                    print("  [OK] 与 bert_base.json (vocab_size=9537) 一致")
                else:
                    print(
                        f"  [WARN] 与 bert_base.json (9537) 不一致, "
                        f"实际是 {shape[0]}"
                    )
    else:
        print("\n[WARN] 没找到 token embedding 权重 key. 可能 key 名不一样, "
              "请人工查看上面 30 个 key.")

    print("\n[INFO] 检查 checkpoint 是否夹带 vocab / tokenizer 信息 ...")
    vocab_hit = False
    for k, v in other_payload.items():
        kl = k.lower()
        if any(x in kl for x in ("vocab", "tokenizer", "token_map", "ids_to_tokens")):
            vocab_hit = True
            print(f"  [HIT] payload key = {k!r}, type = {type(v).__name__}")
            if isinstance(v, dict):
                items = list(v.items())[:10]
                print(f"        前 10 个 items: {items}")
            elif isinstance(v, (list, tuple)):
                print(f"        len={len(v)}, 前 10 项: {list(v)[:10]}")
            else:
                print(f"        value = {v!r}")
    if not vocab_hit:
        print("  [MISS] 未发现任何 vocab/tokenizer 字段. "
              "结合作者的 pretrainDexBERT.py (只 save state_dict), "
              "这个 .pt 99% 只含权重, 必须额外拿到 vocab.txt 才能用。")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AndroZoo 下载脚本 - 直接通过 sha256 列表下载 APK。

三种用法（任选其一）：

1) 直接修改下方 SHA256_LIST 里的 hash, 然后运行:
       python androzoo_download_by_sha.py

2) 命令行传入 sha (可多个):
       python androzoo_download_by_sha.py --sha ABC...123 DEF...456

3) 从文本文件读入 (一行一个 sha, '#' 开头视为注释):
       python androzoo_download_by_sha.py --sha_file my_shas.txt

API Key:
    优先级 --apikey > 环境变量 ANDROZOO_APIKEY。 不要把 key 写进源码。
    申请地址: https://androzoo.uni.lu/access
"""

import argparse
import hashlib
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

import requests
from tqdm import tqdm


SHA256_LIST: List[str] = [
    "0d64bb3c121e1986766505e182f16fb8dcc4188224f3094f99b9f905873ddc4a"
]


ANDROZOO_URL = "https://androzoo.uni.lu/api/download"


def load_sha_list(args: argparse.Namespace) -> List[str]:
    """合并 SHA256_LIST / --sha / --sha_file 三个来源, 去重保持顺序。"""
    raw: List[str] = []

    raw.extend(SHA256_LIST)

    if args.sha:
        raw.extend(args.sha)

    if args.sha_file:
        if not os.path.exists(args.sha_file):
            raise FileNotFoundError(f"sha_file 不存在: {args.sha_file}")
        with open(args.sha_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                raw.append(line.split(",")[0].strip())

    seen = set()
    out: List[str] = []
    for s in raw:
        s = s.strip().upper()
        if len(s) != 64 or not all(c in "0123456789ABCDEF" for c in s):
            print(f"[WARN] 跳过非法 sha256: {s!r}", file=sys.stderr)
            continue
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def sha256_of_file(path: str, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest().upper()


def download_one(
    sha_up: str,
    out_dir: str,
    apikey: str,
    timeout: int,
    retries: int,
    verify_hash: bool,
) -> str:
    """下载单个 apk。返回结果字符串, 以 [OK]/[SKIP]/[FAIL] 开头。"""
    out_path = os.path.join(out_dir, f"{sha_up}.apk")
    tmp_path = out_path + ".part"

    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        if not verify_hash:
            return f"[SKIP] {sha_up}"
        got = sha256_of_file(out_path)
        if got == sha_up:
            return f"[SKIP] {sha_up}"
        os.remove(out_path)

    last_err: Optional[str] = None
    for attempt in range(1, retries + 1):
        try:
            params = {"apikey": apikey, "sha256": sha_up}
            with requests.get(
                ANDROZOO_URL,
                params=params,
                stream=True,
                timeout=timeout,
                headers={"User-Agent": "androzoo-downloader/1.0"},
            ) as resp:
                if resp.status_code == 404:
                    return f"[FAIL] {sha_up} 404 not_in_androzoo"
                if resp.status_code == 401 or resp.status_code == 403:
                    return f"[FAIL] {sha_up} {resp.status_code} apikey_invalid_or_forbidden"
                resp.raise_for_status()

                with open(tmp_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 64):
                        if chunk:
                            f.write(chunk)

            if os.path.getsize(tmp_path) == 0:
                raise IOError("downloaded file is empty")

            if verify_hash:
                got = sha256_of_file(tmp_path)
                if got != sha_up:
                    os.remove(tmp_path)
                    raise IOError(f"sha256 mismatch: got {got}")

            os.replace(tmp_path, out_path)
            return f"[OK] {sha_up}"

        except Exception as e:
            last_err = str(e)
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            if attempt < retries:
                time.sleep(min(2 ** attempt, 15))
                continue

    return f"[FAIL] {sha_up} after {retries} retries: {last_err}"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="按 sha256 直接从 AndroZoo 下载 APK",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--sha", nargs="*", default=None, help="一个或多个 sha256 (空格分隔)")
    ap.add_argument("--sha_file", default=None, help="纯文本文件, 一行一个 sha256")
    ap.add_argument("--out_dir", default="downloaded_samples", help="输出目录")
    ap.add_argument("--apikey",  default="9ba98b64abb3ef608ebfce6a383e5ea8c2b8165b83ea2aae827de2cd07393d67", help="AndroZoo apikey, 优先级高于环境变量")
    ap.add_argument("--workers", type=int, default=4, help="并发线程数, 不建议过高")
    ap.add_argument("--timeout", type=int, default=120, help="单次请求超时秒")
    ap.add_argument("--retries", type=int, default=3, help="单个样本重试次数")
    ap.add_argument(
        "--no_verify",
        action="store_true",
        help="不在下载后校验 sha256 (默认会校验)",
    )
    args = ap.parse_args()

    apikey = args.apikey or os.environ.get("ANDROZOO_APIKEY", "")
    if not apikey:
        sys.exit(
            "[ERR] 缺少 apikey. 用 --apikey 传入, 或 export ANDROZOO_APIKEY=..."
        )

    sha_list = load_sha_list(args)
    if not sha_list:
        sys.exit(
            "[ERR] 没有任何 sha256 可下载. 请在 SHA256_LIST 里填, "
            "或使用 --sha / --sha_file."
        )

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[INFO] 待下载: {len(sha_list)} 个样本 -> {args.out_dir}")

    results: List[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(
                download_one,
                sha,
                args.out_dir,
                apikey,
                args.timeout,
                args.retries,
                not args.no_verify,
            ): sha
            for sha in sha_list
        }
        for fut in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Downloading",
            unit="apk",
        ):
            results.append(fut.result())

    ok = sum(1 for r in results if r.startswith("[OK]"))
    skip = sum(1 for r in results if r.startswith("[SKIP]"))
    fail = sum(1 for r in results if r.startswith("[FAIL]"))
    print(f"\nDone. OK={ok}, SKIP={skip}, FAIL={fail}")

    if fail:
        log_path = os.path.join(args.out_dir, "download_failures.log")
        with open(log_path, "w", encoding="utf-8") as f:
            for r in results:
                if r.startswith("[FAIL]"):
                    f.write(r + "\n")
        print(f"失败记录已写入: {log_path}")


if __name__ == "__main__":
    main()

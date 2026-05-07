#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单 APK 处理脚本: APK -> smali -> 指令 .txt
产物格式与 Data/data4pretraining.py 完全一致, 可直接喂给后续的
DexBERT 词表生成 / 推理流程。

用法:
    python process_apk.py downloaded_samples/XXXX.apk
    python process_apk.py downloaded_samples/             # 处理目录下全部 .apk
    python process_apk.py XXXX.apk --out_dir processed/   # 自定义输出
    python process_apk.py XXXX.apk --keep_smali           # 保留 smali 目录便于调试

输出:
    <out_dir>/<APK_HASH>.txt          每个 APK 一个指令文件
    <out_dir>/<APK_HASH>/             smali 目录 (默认处理完删掉)
"""

import argparse
import os
import os.path as osp
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List

from androserum.data.instruction_generator import SmaliInstructionGenerator


def _project_root() -> Path:
    """Walk up from this file to the nearest ``pyproject.toml`` (project root)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError(
        f"could not locate project root (no pyproject.toml found above {here})"
    )


BAKSMALI_JAR = str(_project_root() / "assets" / "baksmali-2.5.2.jar")


def disassemble(apk_path: str, smali_out_dir: str) -> None:
    """调用 baksmali 反编译 APK."""
    if not osp.exists(BAKSMALI_JAR):
        raise FileNotFoundError(
            f"baksmali jar 不存在: {BAKSMALI_JAR}\n"
            "请先下载: "
            "https://bitbucket.org/JesusFreke/smali/downloads/baksmali-2.5.2.jar "
            f"并放到 {DATA_DIR}/"
        )
    os.makedirs(smali_out_dir, exist_ok=True)
    cmd = ["java", "-jar", BAKSMALI_JAR, "disassemble", apk_path, "-o", smali_out_dir]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"baksmali 失败 (rc={proc.returncode}):\n"
            f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
        )


def extract_instructions(smali_dir: str, txt_path: str) -> int:
    """从 smali 目录抽取所有 class/method/instruction, 写入 .txt.

    格式与 Data/data4pretraining.py 保持一致:
        <txt_basename>
        ClassName: Lcom/foo/Bar;
        MethodName: doSomething(II)V
        <instr>
        <instr>
        ...
        <空行>
        ClassName: ...
        ...
    返回写入的 class 数。
    """
    cls_count = 0
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(osp.basename(txt_path) + "\n")
        for cls in SmaliInstructionGenerator(SmaliRootDir=smali_dir, flag="class"):
            f.write(f"ClassName: {cls.name}\n")
            for method in cls.methods:
                f.write(f"MethodName: {method.name}\n")
                for instr in method.instructions:
                    f.write(instr + "\n")
                f.write("\n")
            cls_count += 1
    return cls_count


def process_one_apk(apk_path: str, out_dir: str, keep_smali: bool = False) -> bool:
    apk_path = osp.abspath(apk_path)
    if not osp.exists(apk_path):
        print(f"[FAIL] APK 不存在: {apk_path}", file=sys.stderr)
        return False
    if not apk_path.lower().endswith(".apk"):
        print(f"[FAIL] 不是 .apk 文件: {apk_path}", file=sys.stderr)
        return False

    apk_name = osp.splitext(osp.basename(apk_path))[0]
    smali_dir = osp.join(out_dir, apk_name)
    txt_path = osp.join(out_dir, apk_name + ".txt")

    if osp.exists(txt_path) and osp.getsize(txt_path) > 0:
        print(f"[SKIP] 已处理过: {txt_path}")
        return True

    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    print(f"[INFO] 反编译 {apk_path} ...")
    try:
        disassemble(apk_path, smali_dir)
    except Exception as e:
        print(f"[FAIL] {apk_name}: 反编译失败: {e}", file=sys.stderr)
        if osp.exists(smali_dir):
            shutil.rmtree(smali_dir, ignore_errors=True)
        return False

    print(f"[INFO] 抽取指令 -> {txt_path}")
    try:
        n_class = extract_instructions(smali_dir, txt_path)
    except Exception as e:
        print(f"[FAIL] {apk_name}: 指令抽取失败: {e}", file=sys.stderr)
        if osp.exists(txt_path):
            os.remove(txt_path)
        return False
    finally:
        if not keep_smali and osp.exists(smali_dir):
            shutil.rmtree(smali_dir, ignore_errors=True)

    elapsed = time.time() - t0
    size_kb = osp.getsize(txt_path) / 1024
    print(
        f"[OK]   {apk_name}: {n_class} classes, "
        f"{size_kb:.1f} KB, {elapsed:.1f}s"
    )
    return True


def collect_apks(target: str) -> List[str]:
    if osp.isdir(target):
        return sorted(
            osp.join(target, f)
            for f in os.listdir(target)
            if f.lower().endswith(".apk")
        )
    if osp.isfile(target):
        return [target]
    return []


def main() -> None:
    ap = argparse.ArgumentParser(
        description="APK -> smali -> instruction .txt (DexBERT 输入格式)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("target", help="单个 APK 文件, 或包含若干 APK 的目录")
    ap.add_argument(
        "--out_dir", default="processed", help="指令 .txt 输出目录"
    )
    ap.add_argument(
        "--keep_smali",
        action="store_true",
        help="保留反编译后的 smali 目录 (默认处理完删掉以省空间)",
    )
    args = ap.parse_args()

    apks = collect_apks(args.target)
    if not apks:
        sys.exit(f"[ERR] 没找到任何 .apk: {args.target}")

    print(f"[INFO] 待处理 APK: {len(apks)}, 输出: {args.out_dir}")
    ok = fail = 0
    for apk in apks:
        if process_one_apk(apk, args.out_dir, keep_smali=args.keep_smali):
            ok += 1
        else:
            fail += 1
    print(f"\nDone. OK={ok}, FAIL={fail}")


if __name__ == "__main__":
    main()

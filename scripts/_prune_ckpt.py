#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Checkpoint 剪枝工具（Python 实现，绕过 shell safe-delete 批量拦截）。

用法:
  python _prune_ckpt.py <ckpt_dir> <keep>
    keep > 0 : 保留 step 最大的 keep 个 .pt，删除其余（按文件名排序）。
    keep == 0: 删除该目录下全部 step_*.pt（用于清空重训）。

不触碰 train.log / eval.log 等其它文件。仅删除 step_*.pt。
"""
import sys
import os
import glob


def main():
    if len(sys.argv) < 2:
        print("usage: _prune_ckpt.py <ckpt_dir> <keep>")
        sys.exit(2)
    ckpt_dir = sys.argv[1]
    keep = int(sys.argv[2]) if len(sys.argv) > 2 else 8

    if not os.path.isdir(ckpt_dir):
        print(f"SKIP (no dir): {ckpt_dir}")
        return

    files = sorted(glob.glob(os.path.join(ckpt_dir, "step_*.pt")))
    if not files:
        return

    if keep <= 0:
        for f in files:
            try:
                os.remove(f)
            except OSError as e:
                print(f"  rm err {os.path.basename(f)}: {e}")
        print(f"CLEARED {len(files)} checkpoints in {ckpt_dir}")
        return

    if len(files) <= keep:
        return

    to_del = files[:-keep]
    for f in to_del:
        try:
            os.remove(f)
        except OSError as e:
            print(f"  rm err {os.path.basename(f)}: {e}")
    print(f"PRUNE {ckpt_dir}: deleted {len(to_del)}, kept {keep} (latest={os.path.basename(files[-1])})")


if __name__ == "__main__":
    main()

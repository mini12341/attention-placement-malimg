# -*- coding: utf-8 -*-
"""把 Hydrogenion/Malimg 的 malimg.npz 转成「家族文件夹 + PNG」结构。

输入: data/malimg.npz  (9339 张 32x32 灰度图 + 0-24 标签；可用 MALIMG_NPZ 覆盖)
输出: $MALIMG_32（默认 ./data/malimg_32）  (25 个家族子文件夹, 每个文件夹里是 PNG)

标签 0-24 按家族名首字母顺序排列(已用样本数量核对过)。
"""
import os
import numpy as np
from PIL import Image

from paths import DATA_32, ROOT

SRC = os.environ.get("MALIMG_NPZ", os.path.join(ROOT, "data", "malimg.npz"))
DST = DATA_32

# 索引 0-24 对应家族名(首字母序)
FAMILIES = [
    "Adialer.C", "Agent.FYI", "Allaple.A", "Allaple.L", "Alueron.gen!J",
    "Autorun.K", "C2LOP.gen!g", "C2LOP.P", "Dialplatform.B", "Dontovo.A",
    "Fakerean", "Instantaccess", "Lolyda.AA1", "Lolyda.AA2", "Lolyda.AA3",
    "Lolyda.AT", "Malex.gen!J", "Obfuscator.AD", "Rbot!gen", "Skintrim.N",
    "Swizzor.gen!E", "Swizzor.gen!I", "VB.AT", "Wintrim.BX", "Yuner.A",
]

def main():
    os.makedirs(DST, exist_ok=True)
    d = np.load(SRC, allow_pickle=True)
    arr = d["arr"]

    counts = {}
    for i, row in enumerate(arr):
        img = row[0]                 # (32, 32) uint8 灰度图
        label = int(row[1])          # 0-24
        fam = FAMILIES[label]
        folder = os.path.join(DST, fam)
        os.makedirs(folder, exist_ok=True)
        Image.fromarray(img, mode="L").save(
            os.path.join(folder, f"{fam}_{i:05d}.png"))
        counts[fam] = counts.get(fam, 0) + 1

    print("总保存张数:", len(arr))
    for fam in FAMILIES:
        print(f"  {fam}: {counts.get(fam, 0)}")


if __name__ == "__main__":
    main()

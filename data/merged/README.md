# 合并语料 · Merged Corpus

把 25 个任务的数据合并成**一个大训练集**，用**一个多任务模型**统一训练。

## 数据现状

| 指标 | 数量 |
| --- | --- |
| 任务数 | 25 |
| 训练样本 | 541,443 |
| 验证样本 | 73,268 |
| 测试样本 | 124,847 |
| 合计 | 739,558 |
| 原始体积 | 771.5 MB |
| 合并后体积 | 2.29 GB（32×32×3 uint8） |

### 各任务规模

| 任务 | 类型 | 头 | train | val | test | 原始尺寸 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| adrenalmnist3d | cls | cls | 1,188 | 98 | 298 | 28×28×28 |
| alzheimer-mri | cls | cls | 4,610 | 510 | 1,280 | 64×64 |
| bloodmnist | cls | cls | 11,959 | 1,712 | 3,421 | 28×28×3 |
| brain-tumor-mri | cls | cls | 3,759 | 1,073 | 537 | 64×64 |
| breast-ultrasound-birads | cls | cls | 1,352 | 148 | 375 | 64×64 |
| breastmnist | cls | cls | 546 | 78 | 156 | 28×28 |
| chest-ct-lung-cancer | cls | cls | 612 | 72 | 315 | 64×64 |
| chest-xray-pneumonia | cls | cls | 5,216 | 16 | 624 | 64×64 |
| chestmnist | multilabel | multilabel | 78,468 | 11,219 | 22,433 | 28×28 |
| dermamnist | cls | cls | 7,007 | 1,003 | 2,005 | 28×28×3 |
| fracturemnist3d | cls | cls | 1,027 | 103 | 240 | 28×28×28 |
| idrid-diabetic-retinopathy | cls | cls | 374 | 39 | 103 | 64×64 |
| nodulemnist3d | cls | cls | 1,158 | 165 | 310 | 28×28×28 |
| oasis-age-reg | reg | reg | 382 | 47 | 47 | 64×64 |
| octmnist | cls | cls | 97,477 | 10,832 | 1,000 | 28×28 |
| organamnist | cls | cls | 34,581 | 6,491 | 17,778 | 28×28 |
| organcmnist | cls | cls | 13,000 | 2,392 | 8,268 | 28×28 |
| organmnist3d | cls | cls | 972 | 161 | 610 | 28×28×28 |
| organsmnist | cls | cls | 13,940 | 2,452 | 8,829 | 28×28 |
| pathmnist | cls | cls | 89,996 | 10,004 | 7,180 | 28×28×3 |
| pneumoniamnist | cls | cls | 4,708 | 524 | 624 | 28×28 |
| retinamnist | cls | cls | 1,080 | 120 | 400 | 28×28×3 |
| synapsemnist3d | cls | cls | 1,230 | 177 | 352 | 28×28×28 |
| tissuemnist | cls | cls | 165,466 | 23,640 | 47,280 | 28×28 |
| vesselmnist3d | cls | cls | 1,335 | 192 | 382 | 28×28×28 |

> 注：`chest-xray-pneumonia` 的验证集只有 16 张，是原始数据集自带的划分；
> 合并训练时它的验证指标方差会很大。

## 合并规则

| 步骤 | 处理 |
| --- | --- |
| 分辨率 | 全部重采样到 32×32（`--size` 可改，64 会涨到 9 GB） |
| 通道 | 灰度复制成 3 通道；RGB 保留颜色 |
| 3D 体数据 | 沿深度做最大密度投影（MIP）转成 2D，便于共用一个 2D backbone |
| 标签空间 | 每个任务一个独立的 head：cls → CrossEntropy；chestmnist → BCE（14 维）；oasis → MSE（标准化） |
| 采样 | 默认按任务均衡采样，避免 tissuemnist（165k）淹没 idrid（374） |

## 用法

```powershell
# 1) 重建数据集（先跑 inventory_datasets.py 生成清单）
python inventory_datasets.py
python merge_datasets.py            # 默认 32x32；--size 64 / --dry-run

# 2) 训练多任务模型
python train_merged.py --epochs 10
python train_merged.py --epochs 20 --no-balanced        # 按比例采样
python train_merged.py --tasks pneumoniamnist,organamnist # 只训部分任务

# 3) 也可以直接在训练台里启动（数据集下拉里选 "merged"）
```

产物：

```
data/merged/merged.npz       合并后的图像与标签
data/merged/merged_meta.json 每个任务的 head 类型、类别数、样本数
checkpoints/merged/          权重与 summary.json（含 per_task 明细）
```

## 模型结构

```
输入 3×32×32
  → Conv(3→32)
  → 4 × ResNet block (stride 2 ×3)  → width×8 = 256
  → AdaptiveAvgPool + Dropout
  → 25 个任务各自的 Linear head
```

参数量约 2.75 M，与单任务模型同量级（2.7 M）。

# 训练台 · Training Console

医学影像训练套件（`train_medmnist.py` + `run_suite.py`）的本地可视化控制台。
纯标准库实现，无第三方依赖。

## 启动

```powershell
python console\server.py --port 8770
```

或双击 `console\start_console.bat`（自动打开浏览器）。
浏览器访问 <http://127.0.0.1:8770/>。

## 功能

| 区域 | 说明 |
| --- | --- |
| 顶部状态条 | GPU 型号 / 显存占用 / 利用率 / 温度，实时刷新 |
| 启动训练 | 选择数据集与超参（epochs / batch / lr / seed / iters），一键启动 |
| 任务队列 | 历史与运行中任务，点击切换查看对应日志 |
| 结果总览 | 全部任务的结果表，支持搜索、按类型筛选、按指标排序、水平条 |
| 训练曲线 | 选中任务后绘制训练损失 / 验证损失 / 验证指标（自绘 SVG，悬停看逐轮数值） |
| 实时日志 | 增量拉取训练输出，错误行标红，任务结束自动回写结果 |

## 迭代数与断点续跑

| 字段 | 说明 |
| --- | --- |
| 迭代数 iters | `0` = 按轮数训练（`--epochs`）。填 `N` 则改为跑 `N` 个优化步，学习率在这 `N` 步内做 cosine 衰减，默认每 `N/10` 步评估一次。 |
| 继续上次训练 | 从 `checkpoints/<task>/last.pt` 断点续跑：模型、优化器状态、训练历史一并恢复，**预算累加而不是覆盖**。 |

续跑时如果新结果不如已有的 `best.pt`，旧的最优成绩不会被覆盖。

> **注意（学习率陷阱）**：断点续跑时优化器状态里保存的学习率是上一轮训练结束时的值
> （cosine 调度到末尾通常为 `0`），而 `CosineAnnealingLR` 是从当前值**乘法递推**的，
> 直接恢复会把学习率钉死在 `0`，训练看似在跑实则完全不更新参数。
> 两个训练脚本都会在构建新调度器前用 `--lr` 重新播种学习率，
> 并按**本次**预算重建调度器（不沿用旧的 `T_max` / `last_epoch`）。
> `test_resume_lr.py` 是这条路径的回归测试。

## 覆盖保护

重跑一个已有结果的任务时，后端返回 `409` 并在页面上弹出确认框。
确认覆盖后，旧产物会被自动备份：

```
checkpoints/<task>/summary.json -> summary.prev.json
checkpoints/<task>/best.pt      -> best.prev.pt
checkpoints/<task>/last.pt      -> last.prev.pt
```

也可以直接勾选「覆盖已有结果」跳过确认。

## HTTP 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/overview` | 数据集清单 + 结果 + GPU + 统计 |
| GET | `/api/task/<name>` | 单任务 summary 与逐轮 history |
| POST | `/api/train` | 启动训练，body: `{task, epochs, batch_size, lr, seed, iters, resume, overwrite}` |
| GET | `/api/jobs` | 任务列表与状态 |
| GET | `/api/jobs/<id>/log?since=N` | 增量日志（`cursor` 为下次游标） |
| POST | `/api/jobs/<id>/stop` | 终止运行中的任务 |
| GET | `/api/gpu` | GPU 状态 |

同一时间只允许一个训练任务运行（第二个请求返回 `409`）。

## 合并语料任务

数据集下拉里多出一个 `merged` 选项，对应 `train_merged.py`：在 25 个任务合并成的大语料上
训练**一个共享 backbone + 25 个任务 head** 的多任务模型。
先执行 `python merge_datasets.py` 生成 `data/merged/merged.npz` 后该选项才会出现。
详见 `data/merged/README.md`。

## 文件

```
console/
  server.py          后端：任务管理、GPU 探测、结果聚合
  static/index.html  前端单文件（内联 CSS/JS，无 CDN 依赖）
  start_console.bat  Windows 一键启动
```

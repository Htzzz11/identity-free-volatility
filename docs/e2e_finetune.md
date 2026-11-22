# 端到端 fine-tuning：怎么跑，现在的状态

## 状态（2026-11-22）

- 代码写完了，**但训练循环还没有在任何机器上跑通过**。
- 在 Mac（M4 Max，MPS）上，第一步训练就在 Apple 的 GPU 后端里崩溃：
  `MPSNDArrayMatrixMultiplication: Destination NDArray and Accumulator NDArray cannot have different datatype`。
  单独对 DeBERTa 做 forward 加 backward 是正常的，所以问题出在整条链路的某个组合上，还没定位。
- **没有在 CUDA 上测过。** 崩溃点在 MPS 后端内部，CUDA 上很可能没有这个问题，但这只是推测。
- 已经验证过的部分：分词和切块与冻结缓存完全一致（块数相同，同一场电话会的块向量和缓存逐位相同）；第一步的冻结训练能跑。

## 它做什么

和 `models/hier.py` 同一个结构，但 DeBERTa 也一起训练。每个测试季度：

1. 在缓存的冻结块向量上训练上层（和 `hier.py` 完全一样）；
2. 从预训练的 DeBERTa 和这个上层出发，在全文上端到端微调。一次 forward 是一场电话会，梯度累积 4 场更新一次，3 个 epoch；
3. 取 val MSE 最好的权重（第 1 步的冻结模型算作第 0 个 epoch 的候选），预测测试季度并存盘。

超参数在 `models/e2e.py` 顶部：encoder 学习率 1e-5，上层 1e-4，warmup 10% 后线性衰减。都没有调过。

## 怎么跑

```bash
uv sync
# data/ 不在 git 里，要先拷过来：至少需要 data/raw/dec_repo、data/raw/dec_drive/DEC.json
uv run python scripts/encode_chunks.py --corpus dec    # 冻结块向量缓存，GPU 上几分钟到十几分钟
uv run python scripts/run_e2e.py --corpus dec          # 从 2021Q1 开始，一个季度一个季度跑
uv run python scripts/run_e2e.py --corpus dec --report # 在已完成的季度上和所有 baseline 比
```

先用小样本确认能跑通（几分钟）：

```bash
uv run python scripts/run_e2e.py --corpus dec --seed 9 --only 2021Q1 --max-train 16 --max-eval 6
```

## 随时可以停

- 每完成一个季度，写一个 `results/preds_partial/dec/hier_e2e_deberta/seed<k>/<季度>.parquet`，下次启动会跳过。
- 季度中间每 100 场电话会和每个 epoch 结束各存一次断点，在 `checkpoints/` 下。再跑同一条命令就从断点继续。
- 一个季度完成后断点会被删掉。断点每个约 2 GB。

## 显存

一步训练是一场电话会的全部块（平均 24 块，每块 512 token），带梯度，fp32。训练时超过 40 块的电话会随机丢块到 40 块（DEC 里占 1.4%），验证和测试始终用全文。

没有用混合精度。显存不够时调小 `--max-train-chunks`，例如 16GB 的卡先试 `--max-train-chunks 8`。调小以后要在报告里写明。

## 耗时估计

Mac 上实测的训练速度是每秒 4.2 块，12 个测试季度全跑完约 46 小时。CUDA 上没有实测数字。

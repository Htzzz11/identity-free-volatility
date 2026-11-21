# Songhan 部分的计划与逐步核对

范围：proposal 里 Songhan 负责的数据、标签和 baselines（11-23 milestone）。
每一步写了「做了什么」和「怎么核对的」。核对于 2026-11-20 全部重新跑过一遍。
核对中发现并修掉了两个问题，见第 4、5 步。

复现命令见 README 的 Setup 一节。所有数据只在本地 `data/`，没有进 GitHub。

## 进度

- [x] 1. 下载 STRUX 和 DEC
- [x] 2. 用 yfinance 缓存复权收盘价
- [x] 3. 确定标签公式
- [x] 4. 恢复发布时间（盘前 / 盘后）并给 STRUX 打标签 **（核对时发现 bug，已修）**
- [x] 5. 筛选 ≥20 场的 ticker **（核对时发现 3 场重复，已剔除）**
- [x] 6. rolling-by-quarter 划分
- [x] 7. PEV / STPEV
- [x] 8. Random(Ticker)、Random(All)、OpenAI embedding + MLP
- [x] 9. TF-IDF + ridge
- [x] 10. 统一的预测文件格式和指标
- [x] 11. 推代码、开 PR
- [ ] 12. LLM-summary baseline（脚本已写好，等确认模型、费用和 API key，目前没有花钱）

## 逐步核对

### 1. 下载数据
- 做了什么：STRUX 来自 Hugging Face，DEC 的标签来自作者 GitHub，DEC 的转录和 OpenAI embedding 来自作者的 Google Drive。脚本是 `scripts/download_data.py`。
- 核对：
  - STRUX 11,411 场、858 个 ticker，没有重复的 (ticker, 日期)。
  - DEC 1,800 行，90 个 ticker，每个正好 20 场，标签没有缺失。
  - DEC.json 的 1,800 个 id 和 DEC.csv 完全一致；embedding 是 1800 × 3072，id 顺序和 DEC.csv 一致，没有 NaN。
  - `git ls-files` 里没有任何 `data/`、`results/` 下的文件；PR 的 diff 里搜不到转录文本；`.env`、`data/`、`results/` 都在 .gitignore 里。

### 2. 价格缓存
- 做了什么：`prices.py` 下载 2016-06 到 2024-12 的复权收盘价，存成 `data/prices/close.parquet`。
- 核对：
  - 747 个 ticker 有价格，日期索引有序无重复，序列中间没有缺失日，没有非正价格。
  - 111 个 ticker 在 Yahoo 上已经拿不到（退市或被收购）。其中 7 个属于 ≥20 场的集合：COOP、DOMO、DVAX、EA、EXAS、SPLK、WBA。MMC、FI、BK 三个是改了代码，已映射到 MRSH、FISV、BNY。

### 3. 标签公式
- 结论：`v = log sqrt( (1/τ) · Σ_{i=0}^{τ-1} (r_{t+i} − r̄)² )`，r 是**简单收益率**。proposal 里的写法（没有 1/τ、求和 τ+1 项）是错的。
- 核对：
  - 用 DEC 自带的收益率列按这个公式算，四个 horizon 都和 DEC 的标签列完全一致（误差 < 1e-9）。
  - Target Q3 2017，τ=3：算出 −2.726。换成 log 收益率是 −2.698，去掉 1/τ 是 −2.177，都对不上。
  - 这两条都写成了单元测试（`tests/test_labels.py`）。
- 备注：τ=3 的标签本身噪声很大。例如 GILD 2018-10-26 连续三天涨幅几乎相同，标准差接近 0，标签是 −8.39。这是定义带来的，不是数据错误。

### 4. 发布时间恢复和 STRUX 标签
- 做了什么：STRUX 只有电话会日期 D。标签需要知道第一个反映财报的交易日是 D 还是 D 的下一个交易日。来源按优先级：DEC（重合的部分）、Yahoo 的财报时间戳、转录开头的问候语（good morning / afternoon）、同一 ticker 的多数情况。
- **核对时发现的 bug：** 一开始我直接把 DEC 的 `beforeAfterMarket` 套在电话会日期上。但这个标记是相对 DEC 数据商的「财报发布日」的，不是相对 `day_earnings` 列（电话会日期）。O、EOG、CCI、PPG 这类公司是前一天盘后发布、第二天早上开会，套上去之后我们的窗口晚了一天。
  - 之前我说「DEC 这几家的标签窗口晚了一天」，**这个说法是错的**，错的是我们，DEC 的标签是对的。已更正 PR 描述。
  - 修法：DEC 来源不再用原始标记，改为看 DEC 自己的收益率窗口从哪一天开始（`strux._dec_window_flags`）。
  - 影响：4,127 个标签里改了 63 个（O 19、EOG 18、CCI 17、PPG 7、EQIX 1、PLD 1）。
- 修完之后的核对：
  - 和 DEC 重合的 1,360 场，我们的标签与 DEC 的差距小于 0.05 的比例：τ=3 是 99.8%，τ=30 是 99.9%（修之前是 94%–98%）。
  - 不用 DEC 时各来源的准确率（以 DEC 的窗口为准，1,357 场）：Yahoo 99.7%（覆盖 69%），问候语 97.0%（覆盖 71%），整条 fallback 链 98.1%（覆盖 95%）。这是 DEC 没覆盖的 ticker 上可以预期的准确率。
  - 独立重算：重新从 Yahoo 下载价格，对随机 25 场 × 4 个 horizon 用另一段代码重算，最大差小于 1e-4。
  - 最终语料里，第一个交易日减电话会日期：0 天 2,616 场，1 天 1,496 场，2–4 天 12 场（周末和节假日），没有负数。

### 5. 筛选 ≥20 场的 ticker
- 结果：179 个 ticker、4,281 场；能打上四个标签的 4,127 场；最终进语料 4,124 场、172 个 ticker。
- **核对时发现的问题：** STRUX 里有 3 场是同一场电话会换了个日期又出现一次（AMZN 2018-01-31、BIDU 2018-02-21、MA 2019-01-29），日期没有任何财报日历能对上。已剔除（`strux.duplicate_suspects`）。
- 核对：dense 集合里每个 ticker 至少 20 场；没有完全相同的转录文本。

### 6. 划分
- 做了什么：DEC 直接用作者给的划分列。STRUX 按同样的规则重建：测试集是一个日历季度，之前所有的电话会按 2:1 随机分成 train / val。
- 核对（DEC 和 STRUX 都是 0 处违规）：
  - train / val 的季度都严格早于测试季度，日期也严格早于测试集最早的日期。
  - train、val、test 互不重叠。
  - 从第一个测试季度起，每场电话会恰好当一次测试样本。
  - STRUX 里有 20 个测试样本的 ticker 之前没有历史，STPEV 对它们退回到 PEV。

### 7. PEV / STPEV
- 核对：
  - STPEV 和 DEC.csv 里自带的 `stpev_mean_lv3f3` 列逐行一致。
  - DEC 2021–2023 平均 MSE：STPEV 0.252（论文 0.252），PEV 0.310（论文 0.307）。STPEV 逐季度完全一致；PEV 逐季度最多差 0.011，论文没有说明 PEV 用哪段历史求平均，我试了 8 种定义，现在用的是最接近的。

### 8. MLP 类 baseline
- 做了什么：把作者的 TMLP 移植成 `baselines/mlp.py`，超参数照搬（Adam 1e-3，每个 epoch 减半，batch 16，最多 10 个 epoch，patience 3），每个 horizon 一个模型，3 个 seed。
- 核对：和作者 repo 里自带的结果文件逐格（19 个季度 × 4 个 horizon = 76 格）比较：

  | embedding | 我们 | 作者文件 | 相关系数 |
  |:--|--:|--:|--:|
  | OpenAI | 0.324 | 0.321 | 0.997 |
  | Random(Ticker) | 0.313 | 0.311 | 0.993 |
  | Random(All) | 0.361 | 0.365 | 0.978 |

- 备注：论文 Table 4 里 OpenAI 这一行（2021–23 平均 0.263）比作者自己的结果文件（0.278）和我们（0.280）都好一点，这是论文表格和他们公开代码之间的出入。

### 9. TF-IDF + ridge
- 做了什么：unigram + bigram，词表和 IDF 只用测试季度之前的数据，alpha 在 val 上选。
- 核对：
  - 无泄漏：对 2019Q3，直接只在历史数据上 fit 一个 sklearn 的 `TfidfVectorizer`，特征数相同（43,359），预测最大差 1.7e-6。2019Q4 特征数达到上限 50,000，预测差 0.008，来自词频并列时的取舍，不是泄漏。
  - 打乱训练标签后，2021Q1 的 MSE 从 0.170 变成 0.259，比 PEV（0.216）还差，说明模型没有别的途径拿到测试标签。

### 10. 预测文件和指标
- 核对：
  - 每个预测文件都是每个测试样本恰好一行，没有 NaN（DEC 1,710 行，STRUX 3,759 行）。
  - TF-IDF 在 DEC 2021–23 的 MSE 手算一遍是 0.243 / 0.519 / 0.215 / 0.137 / 0.100，和 `evaluate.summarize` 一致。

### 11. PR
- https://github.com/Htzzz11/identity-free-volatility/pull/1
- 核对：8 个单元测试通过；PR 里只有代码；PR 描述里的数字已按修正后的结果更新。

### 12. LLM-summary baseline（未完成）
- 已有：`scripts/llm_summaries.py`（走 Batch API；不加 `--yes-spend` 不会发任何计费请求）和 `baselines/summary.py`（本机开源 embedding + 同一个 MLP）。
- 还缺：选定模型和确认费用，以及本机的 `ANTHROPIC_API_KEY`。

## 最终结果（四个 horizon 的平均 MSE）

| 模型 | DEC 2021–23 | 论文 | STRUX 2021–24 |
|:--|--:|--:|--:|
| TF-IDF + ridge | 0.243 | 无 | 0.276 |
| STPEV | 0.252 | 0.252 | 0.283 |
| Random(Ticker) + MLP | 0.267 ± 0.001 | 0.265 | 0.296 ± 0.003 |
| OpenAI embedding + MLP | 0.280 ± 0.002 | 0.263 | 无 embedding |
| PEV | 0.310 | 0.307 | 0.416 |
| Random(All) + MLP | 0.331 ± 0.003 | 0.323 | 0.438 ± 0.003 |

DEC 的数字不受第 4、5 步修正的影响（DEC 直接用它自己的标签）。STRUX 的数字是修正后重跑的，和修正前相比变化不超过 0.001。

## 没有核对到的地方

- DEC 没覆盖的 STRUX ticker，发布时间只能靠 Yahoo 和问候语，按上面的准确率估计大约有 1%–2% 的标签会差一天。
- 没有逐个确认 111 个拿不到价格的 ticker 是退市还是改了代码，只处理了已知的三个改名。

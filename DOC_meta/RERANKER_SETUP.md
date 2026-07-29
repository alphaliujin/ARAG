# Cross-Encoder Reranker (B2)

为提升 DocScan 比对准确率（特别是降低"主题相同但内容不同"假阳性），在 `combined_similarity` 之上加了一个 **cross-encoder reranker** 精排层。

## 当前状态：未启用（优雅降级中）

后端启动会输出：
```
[RERANKER] disabled — reranker model dir not found: <project>/bge-reranker-v2-m3-local
```

DocScan 比对仍正常工作，走原启发式 `combined_similarity`，无需任何配置。

## 启用步骤

### 1. 下载模型（外网机器一次性）

模型：`BAAI/bge-reranker-v2-m3`（约 568 MB）。

```bash
# 任选其一 (外网):
git clone https://huggingface.co/BAAI/bge-reranker-v2-m3 bge-reranker-v2-m3-local
# 或:
huggingface-cli download BAAI/bge-reranker-v2-m3 --local-dir bge-reranker-v2-m3-local
# 或国内镜像:
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download BAAI/bge-reranker-v2-m3 \
  --local-dir bge-reranker-v2-m3-local
```

### 2. 放到项目目录

把整个 `bge-reranker-v2-m3-local/` 目录放在：
```
/Users/alpha/Desktop/workspace/ARAG_V0.2/bge-reranker-v2-m3-local/
```

或放到任意位置，然后在 `backend/.env` 设置：
```
RERANKER_MODEL_PATH=/path/to/bge-reranker-v2-m3-local
```

目录里需包含：`config.json`、`tokenizer.json`（或 `sentencepiece.bpe.model`）、`pytorch_model.bin` 或 `model.safetensors`。

### 3. 重启后端

```bash
./stop.sh && ./start.sh
```

第一次 DocScan 比对时会触发模型加载（首次约 2~5 秒），之后常驻内存，加载日志：
```
[RERANKER] loaded /Users/alpha/.../bge-reranker-v2-m3-local on mps
```

设备自动选 `mps` (Apple Silicon) / `cuda` / `cpu`，由 `settings.DEVICE` 控制。

## 性能开销

| 设备 | 100 chunks × 3 密级 × top-10 | 备注 |
|---|---|---|
| MPS (M1/M2) | ~30~60 秒 | 推荐 |
| CUDA | ~10~20 秒 | 取决于显卡 |
| CPU | 5~15 分钟 | 不推荐生产用 |

每次 `compare` 调用约 300 次 reranker batch（每批 10 对）。如果觉得慢，把 `n_results` 从 10 调到 5，开销减半，召回略降。

## 评分融合策略

reranker 分数（0~1，sigmoid 后）与启发式 `combined_similarity` 融合：

| reranker | 融合 |
|---|---|
| ≥ 0.6 | `0.7 * rerank + 0.3 * combined` （强相关，rerank 主导） |
| < 0.2 | `min(combined, 0.4)` （强不相关，**压低假阳性**） |
| 0.2~0.6 | `0.5 * rerank + 0.5 * combined` （不确定，平均） |

reranker 不可用时直接用 `combined`，零回归。

## 关闭

删除 `bge-reranker-v2-m3-local/` 目录即可（或清空 `RERANKER_MODEL_PATH` 后重启）。

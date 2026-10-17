# 第 3 章学习笔记：用户记忆与知识库检索

> 实验日期：2026-10-16/17 · 环境：Windows + Python 3.13（纯 CPU，无 GPU）
> 代码仓库：`nailong676/ai-agent-book-chapter3`（私有）

---

## 一、本章知识地图

```
用户输入
   ├── 用户记忆（跨会话持久化，精炼笔记）── 侧重「写」：更新、冲突解决、时间戳
   └── 知识库 RAG（文档检索）────────────── 侧重「读」：chunk → embed → retrieve → fuse → rerank
```

三层易混淆概念辨析：

| 层次 | 本质 | 生命周期 | 本实验对应物 |
|---|---|---|---|
| 聊天记录 | 原始完整对话历史 | 单会话 | `data/conversations/` |
| 当前上下文 | 模型实际看到的窗口（可截断/压缩） | 单会话 | prompt 中注入的部分 |
| 长期用户记忆 | 提炼后的结构化笔记 | 跨会话 | `data/memories/`（带时间戳与 session_id） |

---

## 二、实验 3-5：sparse-embedding（BM25 与倒排索引）

### 做了什么
- 在 10 篇内置英文语料上自建倒排索引 + BM25，复现「逐词 IDF / TF / BM25 贡献」日志（`--explain`）
- 带标注评测（`--eval`，k=5）：

| 查询 | recall@5 | 结论 |
|---|---|---|
| model distillation / HTTP 404 error / XK9-2B4-7Q1 / BM25 ranking function | 1.00 | 精确关键词、错误码、专有名 → BM25 极佳 |
| **cat** | **0.00** | 文档只写 kitten / feline → **同义词短板，漏召回** |

- 宏平均：recall@5=0.800，precision@5=0.700，MRR=0.800
- SPLADE（`naver/splade-cocondenser-ensembledistil`，418MB，本地加载）：查询 `cat` 成功召回 kitten/feline 文档 → **学习型稀疏检索补齐 BM25 同义词短板**
- `server.py`（:4241）全流程跑通：/index /search /stats /index/structure /document

### 要点
- BM25 是词面匹配：得分 = Σ IDF(t)·TF 饱和·长度归一化；解释日志能看清每一分来自哪个词
- 检索不到同义词文档不是 bug，是词项匹配的边界
- 调 k1/b 时要同时观察长文档与短文档的排名变化

---

## 三、实验 3-6：retrieval-pipeline（chunk → sparse → dense → RRF → rerank）

### 逐阶段对照（每次只改一项，同一查询集 12 条）

| 阶段 | Recall@3 | MRR | 观察 |
|---|---|---|---|
| 1. 纯 BM25 | 0.8333 | 0.7917 | 语义改写、跨语言查询翻车（0.00） |
| 2. +Dense（MiniLM-L6-v2） | 1.0000 | 0.9167 | 近似编码混淆：XR-7003 排到兄弟型号后 |
| 3. +RRF 融合 | **1.0000** | **1.0000** | 两路弱点互补 |
| 4. +Rerank（bge-reranker-base） | 1.0000 | 0.9583 | 反把 HTTP-400 正确文档降到第 2 |

### 单查询追踪 `XR-7003`
```
BM25:   #1 xr_7003                     ← 精确匹配
Dense:  #1 xr_7001  #2 xr_7003         ← 稠密先排到兄弟编码
RRF:    #1 xr_7003                     ← 融合把精确匹配推回第 1
Rerank: #1 xr_7003 (7.24)              ← 稳居第 1
```

### chunk-size 消融（其余条件不变）

| chunk_size | 块数 | Dense MRR | Hybrid-RRF MRR |
|---|---|---|---|
| 80（太小） | 32 | 0.8611 ↓ | 0.9444 ↓ |
| 280（默认） | 22 | 0.9167 | **1.0000** |
| 1000（太大） | 20 | 0.9167 | 1.0000 |

→ 块太小：语义切碎，dense 明显受损；BM25 在所有配置下不变（只影响 dense 路）。

### 服务端（dense:4240 / sparse:4241 / pipeline:4242）
- `test_client.py` 五组教学用例全部符合预期：语义/跨语言 dense 胜，精确码/人名 sparse 胜，hybrid 全部 100%
- `demo.py` 6 条查询（API key、概念、404、跨语言、人名、语义上下文）hybrid+rerank top1 全对

### 两条数据流程图（RAG 主线）

**流程一：文档如何变得可检索（离线建库）**

```
原始文档
  → 字符窗口切分（chunk_size=280, overlap=40；20 篇 → 22 块）
  → 双路表示：
    ├── 稀疏路：分词 → 倒排索引（term → doc 集合，BM25 统计 DF/TF/文档长度）
    └── 稠密路：Embedding 模型（MiniLM-L6-v2 / bge-m3）→ 向量 → 索引（FlatIndex 暴力余弦）
```

**流程二：提问后如何检索并回答（在线查询）**

```
query
  → 并行两路召回：BM25 top-k ＋ dense top-k（各自候选列表）
  → RRF 融合（按排名倒数融合，不看分数尺度）
  → bge-reranker 精排（query-候选对逐一打分）
  → top 证据块注入 prompt 上下文
  → LLM 基于证据生成回答
```

**记忆线：一次偏好更新的全程追踪（user-memory 实证）**

```
用户说"我搬到了上海"
  → 对话 Agent 正常应答（本轮不被打断）
  → 后台记忆处理器读取最近会话 → LLM 决策操作类型
  → 无旧城市事实 → ADD 新笔记（若有旧值则 UPDATE）
  → 写入 data/memories/*.json（content + tags + created_at/updated_at + session_id）
  → 下次会话开启时笔记注入 Agent 上下文
  → 问"我住哪个城市" → 回答"上海"（实证通过）
```

### 要点
- 没有单一最优；混合通常更好；RRF 对分数尺度不敏感，weighted 融合要注意两路分数尺度
- **重排不保证总比融合好**——候选已全对时它只可能重序；判断重排增益要用更大候选集
- 小语料满分不能外推到其他语料

---

## 四、user-memory 实验（跨会话记忆）

### 架构
对话 Agent（前台，注入记忆上下文）+ 记忆处理器（后台，每轮提取/更新笔记），分离设计。

### 过程与结果
- Session 1：3 轮对话 → 提取 3 条笔记（名字 / 职业 / 偏好）
- 撞坑：账号组织级限流 **3 RPM**，连发导致 Session 2 全部 429
- 改法：自写 `paced_memory_demo.py`（35s 节流 + 429 退避重试），新会话 4 轮：

| 轮次 | 输入 | 记忆操作 |
|---|---|---|
| 召回 | "What do you remember about me?" | 正确答出 developer/Python/ML（跨会话注入成功） |
| 更新 1 | "moved from Beijing to Shanghai" | **ADD** location 笔记（旧值不存在，非 UPDATE） |
| 更新 2 | "switched from PyTorch to JAX" | **ADD** preferences 笔记 |
| 验证 | "Which city? Which framework?" | 正确回答 Shanghai + JAX |

### 要点
- 「记住用户偏好」侧重写的确定性：小数据量、精确命中、可解释更新 → 结构化存储；「查询产品说明书」侧重语义召回 → 必须 chunk+embed+retrieve
- 事实更新不是 append-only：新值覆盖当前态，旧值保留为带时间戳历史；无旧值时才是 ADD
- 后台记忆处理使每轮实际 LLM 调用翻倍，压测/演示必须按 RPM 预算 pacing

---

## 五、五个理论问题（带实验证据）

1. **聊天记录 / 上下文 / 长期记忆是同一回事吗？** 不是。记录是源、上下文是工作区、长期记忆是精炼持久层（见第一节表）。
2. **"住在北京"→"搬到上海"怎么存？** 新值覆盖当前态 + 旧值带时间戳留痕；个性化用当前值，历史值回答"什么时候搬的"类问题。实验实证：无旧值时系统 ADD 新笔记。
3. **记偏好 vs 查说明书能否用相同检索技术？** 技术上可以，需求权重不同：前者结构化精确读写，后者必须语义召回。证据：`HTTP-403` 靠 BM25 精确命中（dense 混淆兄弟错误码），`cat→kitten` 靠 dense/SPLADE。
4. **块太大/太小的问题？** 太小：语义切碎、索引膨胀（Dense MRR 0.9167→0.8611）；太大：向量被多主题稀释、浪费上下文 token。经验起点：几百字符 + 10~15% 重叠。
5. **RAG 答错的三层归因？** 按序排查：① 没检索到证据（查 recall，如 cat 漏召回）→ ② 检索到错误证据（查 ranking，如 xr_7001 排到 xr_7003 前）→ ③ 模型没用对证据（对比注入原文与回答）。顺序不能反。

---

## 六、环境踩坑记录（Windows CPU 复现指南）

| 坑 | 解法 |
|---|---|
| HuggingFace 直连不通 | hf-mirror + curl 单线程续传；限流后改 ModelScope 本地下载 |
| Windows 无 annoy/hnswlib 预编译 wheel | `indexing.py` 改可选依赖 + 纯 numpy `FlatIndex` 回退 |
| 系统代理 127.0.0.1:7990 拦截 localhost | 客户端/服务端均设 `NO_PROXY=localhost,127.0.0.1` |
| HF 模型默认联网拉取 | `config.py` 改本地 `models/` 目录优先，CPU 关 fp16 |
| SPLADE 依赖 | torch 2.14.1+cpu + transformers，模型 418MB 本地优先加载 |
| PowerShell | 无 heredoc（git commit 用多个 `-m`）；git 的 stderr 进度被显示成"错误"但实际成功 |
| Kimi API 3 RPM 限流 | 调用间隔 ≥35s + 429 退避重试 |

安全：API Key 仅存 `user-memory/.env`（未入库）；GitHub 推送时 `.env` / 模型权重（~3.8GB）/ 运行时数据全部由 `.gitignore` 排除。

---

## 七、一句话总结

> BM25 赢在精确词项，dense 赢在语义泛化，RRF 让两者互补，rerank 只负责精排候选池；而用户记忆是另一套以「写入与更新」为核心的系统，跨会话靠精炼笔记注入而非全量历史。

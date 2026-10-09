# Phase 0 audit：本地 MA-RAG

檢查日期：2026-10-09～10（Asia/Taipei）。目標為此資料夾，未 clone 外部 repository，未修改另一個 agentic-rag-failure-detection 專案。
修改前 `git status --short` 為空；來源 commit：`daa94d2471ce83065eb7e969d242040556034ba6`。目前修改未 commit／push。

## 已驗證的架構

外層 `main.py`：`START → planer_node → plan_executor_node`，只執行一次 planning/execution，沒有 replanning loop。拼字 `planer_node` 原樣保留。

內層 executor：`START → task_definer → single_task_execute → task_definer`。`task_definer` 在停止時產生 summary 並路由至 END。

RAG subgraph：`START → retrieve → extract → generate → END`。
Aggregate 是 `single_task_execute` 內的分支，不是新增 graph node。Summary 是 `task_definer` 內的 LLM call。

| Component | 原始位置／prompt prefix | Input | Output schema | Temperature |
|---|---|---|---|---|
| Planner | `agents/plan.py`／`planing_*` | question、past plan summaries | `PlanFormat` | 0.3 |
| Step Definer | `agents/step_definer.py`／`step_*` | plan、current step、previous answers | `StepTaskFormat` | 0.3 |
| Extractor | `agents/rag.py`／`extract_*` | passage、question | 原本即為 free text | 0.0 |
| QA Generator | `agents/rag.py`／`qa_*` | doc-ID tagged extraction notes、question | `QAAnswerFormat` | 0.3 |
| Aggregate | `agents/plan_executor.py`／`aggregate_*` | concrete task（包含 definer 放入的先前資訊） | `QAAnswerFormat` | 0.0 |
| Plan Summary | `agents/step_definer.py`／`summary_*` | original question、plan、task/query/answer/rating memory | `PlanSummaryFormat` | 0.0 |

Extractor 逐 retrieved document 呼叫一次，保留 `[note]` 包裝以及 QA context 的 `doc_{id}: {note}` 格式。Aggregate 不檢索，也不呼叫 Extractor。

原始 schema／state 位於 `src/utils.py`：`GraphState` 管理 question、plan、累加的 past_exp；`PlanExecState` 管理累加的 step_question/output/docs_ids/notes、summary、stop；`RagState` 管理 documents、doc_ids、notes、final_raw_answer。原始 `GraphState.final_answer` 沒有 node 賦值，最終答案實際在 `past_exp[-1].plan_summary.answer`；此次保留。

原始 Structured Output 使用 LangChain `with_structured_output`；Extractor 用 `StrOutputParser`。目前使用共用 backend，Ollama 仍送出相同 Pydantic JSON schema，並在回傳後以原始 Pydantic schema 驗證；見 migration 文件。

## Success 與停止規則

原始條件：所有 plan steps 已完成，**或**最後一步 `success.lower() == "no"`，便進行 summary；其他情況繼續下一步。沒有基於 rating 的 routing。僅 `type == "aggregate"` 走 Aggregate，其餘 type 仍走 RAG，未改成 enum 限制。

已加入明確 serialization equivalents；合法 yes/no 行為不變。`"False"`／`false`／`0` 現在正確對應 no；原始程式對字串 False 會繼續、對 boolean 則無法呼叫 lower。未知值會停止 affected run 並保留 validation error，不生成重試或強制摘要。詳見 success audit。

## Retriever audit

| 項目 | 實際程式／驗證結果 |
|---|---|
| Embedding model | `hf/gte-multilingual-base`，未更換 |
| Index | `faiss.IndexFlatIP`，未更換 |
| Score | inner product；不能僅憑 index type 宣稱 cosine |
| Query normalization | `RetrieveTopChunk.__call__` 中 `F.normalize(..., p=2, dim=1)` |
| Document normalization | `corpus/embed_corpus.py` 有相同 L2 normalization；實際載入 shards 還必須檢查 norms |
| Top-k | `RetrieveTopChunk` 預設 10；main 沒有覆寫 |
| 排序 | 原 FAISS search 與 dict insertion order，未增加排序／filter／reranking |
| 分數原先消失處 | `Retriever.search` 回傳 doc-ID→score dict；`RetrieveTopChunk` 只迭代 ID，回傳 texts、IDs，丟棄 scores |
| 新 audit | logging 啟用時逐 shard 驗證全體 document norms，SHA256 fingerprint 依原載入順序記錄；query norms 從實際 search input 檢查 |
| 實機 corpus score | **未驗證**，本資料夾未找到 `save_embs/gte-ml-base/dpr100*` 或 `emb_corpus` shards |
| Synthetic verification | CPU FAISS 真實 IndexFlatIP 的 normalized／non-normalized vectors 測試通過，logging 開關的回傳與 ranking 相同 |

僅兩側 norms 都經驗證、無 invalid FAISS index position、score 範圍有效時，才計算 `1-s1` 與 `1-mean(scores)`；否則保留 raw IP scores，cosine uncertainty 為 null。Margin 為 raw ranking gap，並非機率。

其他原始問題：

- `embed_corpus.py` 寫入 `emb_corpus/gte-ml-base`，main 卻讀 `save_embs/gte-ml-base`。
- Index loader 無 shards 時仍進入 GPU move；安裝的 `faiss-cpu==1.8.0`：`get_num_gpus()==0` 且沒有 `StandardGpuResources`，無法執行原 main 的 GPU index path。
- FAISS 回傳 -1 的不足 top-k case，原程式會 indexing 到 doc_lookup 最後一項；dict 也會合併重複 IDs。此次未變更搜尋結果，保留完整 `faiss_hits` 與 invalid position flag，這種情況不認定 cosine 有效。
- `outputs.last_hidden_state[:, 0][:dimension]` 的 dimension slicing 實際作用於 batch axis。此次未修正 embedding 算法。
- MedCPT adapter 不做 query normalization 且另有候選過濾；main 未使用它，此次未新增該支線的 uncertainty 計算。

## 原始環境與儲存問題

原主程式的 dataset 為 Linux `/scratch2/...` 硬編碼，尚未驗證 Windows 的資料與 corpus store。`--model` 是 legacy CLI/output label，實際 backend 現在取自環境配置。原本 default end_index 很大，此次沒有執行 main。

專案 pyproject 原為 Python 3.14 scaffold、dependencies 為空，但 requirements 為舊版 PyTorch／FAISS stack。本機 `.venv311` 可用，固定 3.11 路徑完成驗證；修正原有 Python 3.12+ 才接受的 nested f-string 引號，產生文字不變。將 ir_datasets import 延至實際 corpus/dataset loader，避免只載入 schemas 就初始化家目錄 cache。

原始檔以每題 JSON 儲存完整 graph output；主程式 catch exception 後只 print。新增 sidecar 保存 failed runs，legacy output 結構僅添加必要 success audit 欄位；新配置輸出資料夾加 backend configuration ID，舊檔不覆寫。原始 Aggregate 的 docs_ids/notes 累加列表沒有 placeholder；未改它，改用 sidecar step IDs 對齊。

`rating: int = Field(default=None)` 原 schema 保留。省略 rating 時 Pydantic 會使用 None；此次未擴充 schema 或把 rating 當成 gold correctness。

## Preservation checks

`python scripts/audit_contract.py` 已比對本地來源 commit：prompt literals、四個 Pydantic output class AST、outer／executor／RAG graph node/edge declarations 全部一致。這是 static contract check，加上 synthetic execution tests；不等於原始 OpenAI 成績的重現。

尚未執行任何真實 MA-RAG question、HotpotQA batch、failure prediction training 或 teacher-forced scoring。

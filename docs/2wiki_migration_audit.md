# 2WikiMultiHopQA 遷移 Phase 0 audit

日期：2026-10-10。專案：`D:\school\研究所\碩論\paper_codes\MA-RAG`。

## 1. 範圍與結論

本次依需求 Execution Policy，只進行 Phase 0：檢查目前工作樹、既有測試與公開原始來源，提出可審查的遷移方案。本次新增本文件，未開始 2Wiki adapter、corpus builder、embedding builder、evaluation 或實驗 runner。未下載資料或模型、未執行完整 graph、未提交或推送。先前 Phase 1.5 的未提交修改保留。

**已驗證**：目前 prompts、四個 Pydantic schema 與 graph 宣告保持原版契約；133 個既有離線測試通過；六個 Qwen3 component probes 有 logprobs，但 coverage 都是 partial。CPU FAISS 可執行合成向量測試。

**阻礙**：沒有本地 2Wiki 資料；沒有原始 DPR shards/docstore 或原始 `hf/gte-multilingual-base`；原始 embedding 的遠端身分與 revision 未確認。尚無真實 GTE cosine 驗證、固定 pooled corpus、索引 manifest、離線 correctness evaluator 或 failure-prediction runner。

**建議**：新增資料／檢索資產 adapter 與離線分析層，保留 agents 與 prompts。研究定位為 2Wiki pooled-context adaptation，不宣稱重現 DPR-W100 benchmark。

## 2. Git 與既有修改

- 分支：`research/qwen3-uncertainty`。
- HEAD：`57d6e660781a3a1f3e5c23c0e6034acca69af0f9`。
- 既有未提交 tracked 修改：`corpus/retrieve.py`、`main.py`、`src/invocation.py`、`src/uncertainty.py`、`src/utils.py`。
- 既有 untracked 程式：`scripts/phase1_5_signals.py`、`src/assets.py`、`src/dataset_io.py`。
- 本次新增：`docs/2wiki_migration_audit.md`。測試暫存輸出位於 `outputs/`，不作研究資產。

上述程式是先前 Phase 1.5 的進行中修改，不能當成已完成的 2Wiki 支援。它們包含本地 dataset path、資產存在檢查、CPU FAISS 選項、cosine audit 與 logprob coverage；目前通過既有回歸測試，但新增行為仍缺專屬測試。沒有重設、覆寫或丟棄既有修改。

## 3. 資料、檢索與執行流程

|位置|目前行為／已確認問題|遷移處理提案|
|---|---|---|
|`src/utils.py`、`src/dataset_io.py`|原 active loader 的 Linux dataset 路徑已在 Phase 1.5 工作樹替換；支援顯式路徑、dataset-specific 環境變數及 `DATASET_PATH`。JSON/JSONL/Arrow 讀取尚非 2Wiki 正規化；同名 `load_dataset` 前段定義被後段覆蓋|新增明確 dataset adapter，不靠檔名猜 schema；保留 HotpotQA/KILT 選項|
|`main.py`|預設仍按 KILT `id/input`，可選 HotpotQA `_id/question`；沒有 2Wiki backend、來源 revision、corpus/index 版本檢查|runner 先 normalize，再僅傳 question 給既有 graph|
|`src/assets.py`|要求已有 DPR PickleLz4FullStore 與本地模型，不自動建立／下載|保留 DPR 路徑，另加 pooled docstore；不可把 2Wiki 假裝成 DPR|
|`corpus/embed_corpus.py`|`ir_datasets` corpus 為 `dpr-w100/natural-questions/train`，使用 passage `.text`；模型字串 `hf/gte-multilingual-base`；doc token 上限 512、batch 512、CLS pooling、L2 normalization|獨立新 builder，明確 config，避免偷偷改原 baseline|
|原 embedding shards|writer 輸出 `emb_corpus/gte-ml-base/dpr100_*`，reader 舊預設 `save_embs/gte-ml-base/dpr100*`，路徑不一致|全部配置化；新資產獨立目錄|
|`corpus/retrieve.py`|讀 pickle `(ids, embeddings)`，建立 `IndexFlatIP`，以 row 對應 ID；目前有 CPU 選項、GPU capability error 與驗證資訊；舊式 shard 載入沒有完整 manifest/version contract|保存 index、row-ID mapping、docstore、manifest；載入前驗證一致性|
|`RetrieveTopChunk`|query 原 max_length 8192，CLS + L2；Top-10 取文，telemetry 留 raw scores|沿用工具接口，供應 pooled text；保留 FAISS 原分數與順序|
|`main.py` 結果|`<exp>_<model>_<dataset>_<configuration_id>/<question_id>.json`；存在就跳過；例外只印出，trajectory 開啟才有持久化失敗資訊|新 runner 用唯一 run 目錄、逐題 status，不覆寫原輸出|

補充：原 query/doc 的 `last_hidden_state[:, 0][:dimension]` 第二個切片沿 batch 軸，不是 feature 軸；目前 768 維下不能視為可配置降維。原 doc 512/query 8192 長度策略不同，原 batch 512 不適合直接搬到 12GB 環境。新 builder 必須明訂這些參數，不能把修正混入舊 benchmark。

原始 DPR-W100 是大型 Wikipedia 100-word passage corpus。本地搜尋專案、相鄰論文專案及預設 HF cache 未找到必要資產；使用者也已確認沒有其他位置。搜尋受到少數舊暫存目錄權限限制，因此不宣稱掃描所有磁碟。[DPR-W100 官方說明](https://ir-datasets.com/dpr-w100.html)

## 4. Agent 契約與六個 LLM 呼叫點

|層級|原流程|
|---|---|
|外層|START → `planer_node` → `plan_executor_node`；沒有新增 replanning|
|executor|START → task_definer → single_task_execute → task_definer；由 task_definer 決定 summary／結束|
|RAG|START → retrieve → extract → generate → END|
|分支|task type 精確等於 `aggregate` 走 Aggregate；其餘走 RAG；summary 在 task_definer 內執行|

|Component|位置|temperature|輸出／次數|
|---|---|---|---|
|Planner|`agents/plan.py`|0.3|PlanFormat，每題通常一次|
|Step Definer|`agents/step_definer.py`|0.3|StepTaskFormat，每個執行 step 一次|
|Extractor|`agents/rag.py`|0|純文字，每個返回 document 一次|
|QA Generator|`agents/rag.py`|0.3|QAAnswerFormat，每個 RAG step 一次|
|Aggregate|`agents/plan_executor.py`|0|QAAnswerFormat，每個 aggregate step 一次|
|Plan Summary|`agents/step_definer.py`|0|PlanSummaryFormat，正常終止時一次|

所有六處經 `src/invocation.py`；集中 backend 可重用。Ollama 維持 `qwen3:8b`、thinking=false、num_ctx=8192、timeout=180 秒；structured output 使用原 schema 的 native JSON format，再以原 Pydantic 驗證；Extractor 保持文字。OpenAI backend 保留且本次未使用付費 API。

`scripts/audit_contract.py` 對原始 commit `daa94d2471ce83065eb7e969d242040556034ba6` 比較 prompt literal、schema AST 與 graph 宣告，五項皆 preserved=true。這是靜態契約確認，不等同所有執行情況的語意形式證明。

`src/success.py` 保留 raw 與 normalized success；未知值明確失敗，不補造 yes/no。它代表 agent 內部是否可回答，不能用作最終 correctness label。最終答案應取 `past_exp[-1]["plan_summary"]["answer"]`，原 GraphState 的 `final_answer` 不是可靠已填入欄位。新 runner 要驗證 summary 是否存在，缺少就記錄 execution failure。

## 5. 不確定性、logprobs 與 cosine 驗證狀態

NLL 既有欄位為 nll_sum、nll_mean、nll_max、nll_std（population std）與 token_count；只使用實際返回的 generated-token logprobs，非法／缺漏不補值。未驗證 cosine 時 Top-1／Top-k cosine uncertainty 為 null；score margin 可保留為 raw IP margin。cosine 的 `1-score` 不是校準失敗機率，可能超過 1。

先前 Phase 1.5 六個獨立 production-wrapper probes 的實測證據：`outputs/phase1_5/signals_20261010.json`。這些使用簡短工程問題，沒有執行完整 graph。

|Component|logprobs_count|eval_count|coverage_ratio|狀態|
|---|---:|---:|---:|---|
|Planner|29|30|0.9667|partial|
|Step Definer|19|20|0.9500|partial|
|Extractor|44|45|0.9778|partial|
|QA|52|53|0.9811|partial|
|Aggregate|34|35|0.9714|partial|
|Summary|23|24|0.9583|partial|

六處 logprob token 拼接與可見 content 一致。plain stop 2/3、custom stop 1/2、structured stop 6/7；length-limited control 為 4/4。`/api/generate` control 亦為 2/3，返回 context 不提供缺失 logprob。這支持「停止／特殊 token 計數不同」的推論，尚不能確定 Ollama 0.40.0 缺失 token 的精確身分。先前 QA 49/50 不應解讀成可補一筆 EOS 機率；coverage ratio 也不是文字完整性的證明。NLL scope 必須記為 returned_generated_tokens_only，保留 partial。

實測模型 digest：`500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41`；Q4_K_M，Ollama 0.40.0。這是當時 probe 設定，後續 run 仍需重新記錄。

目前只確認編碼程式含 L2 normalization，以及既有合成向量 FAISS 測試可通過；**沒有真實 GTE query/doc embeddings，所以尚無研究資產的 cosine 驗證結果**。IP 等於 cosine 需 query、doc 均為單位向量。[FAISS metric 官方說明](https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances)

Phase 3 驗收需在真實小樣本檢查 finite/norm，取 FAISS 返回 row 對應向量，以 float64 NumPy `dot/(norm(q)*norm(d))` 獨立計算，記錄最大誤差與 tolerance（提案 1e-5），保留 raw IP 不重排、不 clip。Index 少於 k、重複 ID、row count 不符及版本衝突都應明確處理；不能讓 FAISS `-1` 對應到最後一筆文件。

## 6. Trajectory、對齊與 evaluation

optional trajectory 位於 `outputs/trajectories/<configuration_id>/<run_uuid>/trajectory.json`，含 question/run/invocation/component/step/document/rank/route、prompt hash、raw/parsed output、token usage、logprobs、NLL、latency、error，以及 retrievals/step masks。以 context 隔離每題並 atomic checkpoint；logging 關閉時不應改變路由。可重用，新增 dataset revision、corpus checksum、index version 與 manifest hash 即可。

Aggregate 不產生 retrieval；原 `docsids/notes` 累積列表不能單靠位置與所有 step 對齊，須使用 telemetry 的 step_id/route/mask。缺失／未執行 retrieval 必須是 null + occurred=false，不能變成零 uncertainty。需另記 actual_returned_count。

目前沒有可用的離線 EM/token-F1、support evidence coverage、question-level feature aggregation 或 classifier evaluation。提案：與官方 answer normalizer 相容，明確選定 single-reference 或官方 alias-aware 評估；failure=1−EM。證據 coverage 透過原 context-entry→doc ID mapping，分開報告 corpus 可涵蓋比例與實際檢索 Recall@K；不要用 title 相同就當 evidence 命中。多次 retrieval 同時報 per-call 與 union 指標。Agent success、QA correctness、evidence coverage 分欄保存。

question-level features 只來自模型／檢索可觀測訊號，按 component 聚合 count/mean/max/std 與 availability mask；不能把 gold、EM、supporting facts、錯誤答案對照文字或 corpus 的 gold 覆蓋率放入 predictor。execution exception 與有答案但 EM=0 要分欄，未產生答案的評估政策預先固定。

## 7. 2Wiki schema 與待批准設計

官方作者 repo 說明的 raw fields：`_id`、`question`、`answer`、`context=[[title, sentences], ...]`、`supporting_facts=[[title, sent_id], ...]`、`evidences` triples、`type`、`entity_ids`。官方 test 缺 answer/support/evidence。官方另有 2021-04-07 sentence segmentation 修正版及 alias-aware evaluation 更新。[官方 dataset 說明](https://github.com/Alab-NII/2wikimultihop)

**尚未取得並檢查實際資料檔**；上述是來源文件 schema，不能宣稱已驗證所有樣本、split counts、unique IDs 或 answerability。Phase 1 先檢查選定 archive 的實際 bytes、checksum、欄位及 sentence-index 一致性，再寫 transformation。不可混用不同版本的 context 與 supporting facts。

|待批准決策|建議與影響|
|---|---|
|資料來源|採官方作者 2021-04-07 修正版；取得前確認檔案大小與下載範圍，記錄 URL、版本日期、SHA256；不自動改用 HF 重包裝版本|
|主要 QA 樣本|有標註的 validation/dev；official test 不作 failure labels。下游 predictor 再依 question ID 分 train/validation/test，這與 corpus source splits 是不同概念|
|schema|正規化 question_id/question/gold_answer/question_type/source_split/context_entries/supporting_facts/evidence_annotations；保存完整 raw、source file/checksum/row index；ID 不改寫，跨 split 重複明確報錯|
|answerability|依實際 variant 定義；缺標註不等於 unanswerable。空答案、缺欄位先報錯或依明確 unlabeled schema 處理，不自行猜測|
|corpus policy|建議主 benchmark 採 `train_plus_validation`，只用 context，不依 gold 選文；明列為 pooled-context benchmark。另支援 `train_only` 作對照；在最終 evaluation 前凍結 policy/checksum|
|passage 與去重|每 context entry 一篇；保留原 title、sentences。建議 embedding text=`title + newline + sentences`（sentence 分隔符固定）；hash 用 Unicode NFC + canonical whitespace 的完整 title/text，不只 title；保留原文、所有來源 mapping，hash collision 再比內容|
|GTE 身分|原 `hf/gte-multilingual-base` 無法證實等於遠端模型。提案新 adaptation 選 `Alibaba-NLP/gte-multilingual-base`，明確記為待驗證替代，固定 model/tokenizer/custom-code commit SHA，不能用可變 `main` 作研究版本|
|編碼策略|提案 dense CLS 768 維、float32 入 FAISS、query/doc 同一 encoder、L2；先用 max_length=512、batch=8 做小樣本截斷／資源評估，再凍結參數。此截斷政策不同原 query 8192，需批准並記錄，不暗改 HotpotQA baseline|
|答案評估|建議以官方 v1.1 alias-aware answer EM/F1 為主，保存 alias 資產版本；若缺相容 alias 檔則先阻擋，或經批准另命名 single-reference 指標|

公開 GTE model card 說明 305M、768 維、最大 8192 tokens 與 CLS/L2 使用方式，僅支持候選配置，不能證明原本缺失本地模型的身分。官方 card 範例有 *100 顯示分數，本研究 FAISS cosine **不乘 100**。[GTE model card](https://huggingface.co/Alibaba-NLP/gte-multilingual-base)

官方 evaluator 的 answer normalization 包含 lowercase、移除標點與冠詞、空白正規化；v1.1 包含 aliases 處理。應引用並測試相容性，不直接把第三方 HotpotQA scorer 當相同版本。[官方 evaluator](https://github.com/Alab-NII/2wikimultihop/blob/main/2wikimultihop_evaluate_v1.1.py)

## 8. 最小檔案變更提案（尚未實作）

|階段|新增／調整|驗收|
|---|---|---|
|1|`src/datasets/two_wiki.py`、dataset registry、`tests/test_two_wiki.py`、schema validation command、`docs/2wiki_dataset_and_corpus_design.md`|實際 schema、唯一 ID、malformed/缺檔、無 gold 傳入 agent、HotpotQA 回歸|
|2|`corpus/build_pooled_corpus.py`、manifest/docstore schema、小型 fixture tests|先 dry-run；去重穩定、title 同名異文、句界與 source mapping、重跑 checksum 相同|
|3|新 `corpus/build_local_index.py`、共用 encoding config、`corpus/retrieve.py` local-index loader|CPU exact FlatIP、小索引排名/真 cosine、resumable shard checkpoint、row-ID/count/version mismatch 拒絕|
|4|獨立 `scripts/run_two_wiki.py`、配置範例、`src/telemetry.py` additive metadata|一題完整 graph，final answer/status 可靠、Aggregate mask、零 gold leakage；不改 agent prompts/graph|
|5|`src/evaluation.py`、`src/features.py`、離線測試|EM/F1/alias、evidence mapping、partial NLL、missing masks、重複呼叫聚合、execution failures|
|6|實驗 sampling/config 與 predictor runner|固定 seed/type strata、question-disjoint splits、train-only preprocessing；Dummy/LR/RF，XGB 可選；AUROC/AUPRC/Brier/logloss/ECE、single-class 明確處理|

新資產提案：`data/2wiki/<source_checksum>/`、`artifacts/2wiki/<corpus_checksum>/<encoder_config_hash>/`、`outputs/2wiki/<run_uuid>/`。原始資料唯讀，manifest 未完成不得當成可載入 index；resume 只接續同一 corpus/model/config，逐 batch durable checkpoint。文件 hash、模型 commit、套件版本、seed、prompt hash、Git dirty 狀態都寫 run metadata。

## 9. Windows 與資源估算

使用 `.venv311\Scripts\python.exe`（Python 3.11）；`.venv` 的 Python 3.14 不作此專案基準。實際 dependencies 在 `requirements.txt`；`pyproject.toml` 的空 dependencies 與 scaffold console entry 不足以建立實驗環境。需明示由 repo root 執行，以免 `agents/src` imports 失敗。路徑使用 pathlib／引號；UTF-8、中文路徑、JSON 原 ID 轉檔名與 invalid Windows characters 都需測試，新 runner 可用 run manifest 做安全檔名 mapping。

安裝的 `faiss-cpu` 1.8.0 可以跑 CPU 測試；沒有 `StandardGpuResources`，不能假定裝 PyTorch CUDA 就有 FAISS GPU。CPU 是必要支援，GPU 非必要。HotpotQA HF Arrow cache 與目前 datasets 版本曾有 Feature `List` metadata 不相容；新 raw JSON 2Wiki 路徑不應依賴該 cache，既有 Arrow compatibility 仍需專屬測試。

FlatIP 的向量儲存成本是 `N * d * 4` bytes，以下按 768 維計算，只含向量，不含 Python／docstore／ID／建置副本。[FAISS index 官方表](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)

|去重 passage 數|單份 float32 vectors／FlatIP|embedding 檔 + index 磁碟最低量|
|---:|---:|---:|
|10,000|29.3 MiB|58.6 MiB|
|100,000|293 MiB|586 MiB|
|500,000|1.43 GiB|2.86 GiB|
|1,000,000|2.86 GiB|5.72 GiB|

另加 raw archive、解壓 JSON、pooled corpus、ID mappings、模型 cache 與 trajectories；無實際 dry-run 前不能給精確總磁碟需求。建置若同時持有 embedding matrix/index/副本，RAM 可達向量成本 2–3 倍以上；採 memmap 或分批 add，避免 Python 全量 JSON/list 放大。DPR 21M 級的 768 維單份已約 60 GiB，超過 32GB RAM，不是本次路線。

Qwen3 現有模型磁碟約 5.23 GB；8192 context 的 probe 回報 VRAM 約 6.30 GB。候選 GTE 305M 純權重約 0.61GB fp16／1.22GB fp32，另有 activations、CUDA workspace。12GB 不保證兩者加長 sequence/batch 同時可跑；建議先獨立離線 embedding，再載入 Qwen，檢索 query encoder 以小 batch/CPU 或經量測的 GPU 配置。原 batch512 不沿用。模型下載大小仍須依固定 revision 檔案清單確認。

完整題目的正常 LLM 呼叫數約 `2 + 2*S + sum(k_r)`（Planner+Summary、每 step 的 Definer 與 QA/Aggregate、每 RAG 返回文件一個 Extractor），不含錯誤與 retry；Top10 時為 `2+2*S+10*R`。例如兩個 RAG 加一個 Aggregate 共 28 calls；1,000 題相同結構約 28,000 calls。實際 steps、token length、timeout 由一題再 10 題量測，不能由短 probe 推算可靠總工時。embedding 時間需 dry-run N/token 長度與小 batch throughput 才能估算。

## 10. 已執行驗證與限制

- `python scripts/audit_contract.py`：五項 preserved=true。
- 既有離線 pytest：**133 passed in 5.82s**。命令：`.venv311\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp=outputs/audit_2wiki_verified_tmp_20261010 --tb=short`。
- 前兩次受 Windows 暫存目錄 PermissionError 阻擋；取得 sandbox 外執行許可後成功，沒有修改測試或放寬測試斷言。
- 現有 tests 涵蓋 backend/success/NLL、合成 FAISS norms/ranking、graph routing、多文件 Extractor、logging toggle、atomic persistence、錯誤保留；使用 mocks，阻擋模型／網路呼叫。
- 不涵蓋：2Wiki schema／去重／resume／persisted index version、真實 GTE cosine、正式一題 end-to-end、EM/F1／classifier calibration。通過 133 tests 不能宣稱以上已完成。

## 11. 優先實作與批准界線

1. **先批准**第 7 節資料來源、schema、corpus policy、編碼／評估方向；下載資料／模型前另列實際 URL、大小與位置取得授權。
2. Phase 1：檢查實際 raw samples，完成 adapter/schema tests 與 validation report；保留 HotpotQA。
3. Phase 2：先 dry-run statistics，審查 N、duplicates、token lengths、RAM/disk，再凍結 corpus manifest。
4. Phase 3：先小型工程 index 與真實 GTE 小樣本 cosine 檢查，再審查全量 pooled embeddings 資源與執行時間；工程 index 與研究 index 分開標示。
5. Phase 4：準備一題 labeled dev 的 dataset path、corpus/index hashes、預期 calls、模型設定、資源與輸出路徑；依前次要求，**執行完整 end-to-end 前等待批准**。
6. Phase 5：離線 correctness/evidence/features 驗證，凍結 failure 與 aggregation policy。
7. Phase 6：準備固定 seed 的 10 題 smoke、100 題 pilot、1,000 題 main 配置；pilot/main 與長 GPU 工作均需批准。不得從一題驗證自動擴展為大規模執行。

目前停止於 Phase 0；Phase 1 尚未開始。

# Phase 1 instrumentation 與交付結果

## 已實作

Logging 由 `TRAJECTORY_ENABLED` 控制，預設關閉；不參與 agent routing。啟用後每題建立獨立 UUID run directory：

```text
outputs/trajectories/<configuration_id>/<run_uuid>/trajectory.json
```

每次 LLM invocation 開始／完成／失敗、retrieval 與 step 更新都 checkpoint。寫入採 temp file、flush、fsync、atomic replace。僅覆寫同一 run 的 checkpoint；不同 runs 不重用路徑。Sidecar 保存 failed runs；legacy result output 保留另存，增加 configuration suffix 避免與原 baseline 混合。

非有限浮點 raw 值以 `{"serialization_type":"nonfinite_float","value":"nan/inf/-inf"}` 明確保存，避免寫入非法 JSON 或把它當成零；NLL utility 仍標記 invalid。

Logging disabled 的 graph output、發送 prompts／schemas／kwargs、calls 數量與 logging enabled 相同，已有 mock regression test。Logprob request 由獨立 `LLM_LOGPROBS` 控制，與 sidecar 開關無關。Logging 增加本地 I/O、metadata queries、index norm/hash audit 時間，因此不是 latency-neutral；不改 inference decisions 或 retrieval ranking。

## Invocation records

每次保存 run_id、question_id、invocation_id、component、step_id、document_id、retrieval_rank、route、model_id、temperature、thinking_mode、prompt_hash、input、完整 raw AIMessage、parsed_output、token_usage、token_logprobs、logprobs_status、NLL、latency_ms、status、structured error、timestamps。

IDs 是 run 內穩定的 `step:0`／`step:1` 與 `<run_uuid>:call:<index>`。Planner／Summary 是 question 層級，step_id 為 null；Summary 不虛構最後一個 step ID。Extractor 每個 passage 都是獨立 invocation，附原 document ID 與一開始為 1 的 retrieval rank。

沒有記錄完整環境變數、API credentials 或 request headers。附加 raw output 不取代 agent 實際接收到的 output，也不剝除 reasoning content。Ollama thinking=false 的 wire 配置由 backend 設定，實機對照證據見 migration。

## Retriever records 與 masks

保存 query、top_k_requested／returned、ordered_document_ids、raw_faiss_scores、cosine_similarities（驗證通過才有）、full ordered `faiss_hits`（包括 invalid positions）、embedding model、index type／version、query/document norm validation、similarity validity、Top1／TopK mean uncertainty、score margin、latency／status／error。

Cosine validity 使用實際 loaded shards 與 query norms，容差 rtol=atol=1e-5，沒有重新 normalize index 或改排序。index_version 來自逐 shard SHA256 與原載入順序；這可能增加大型 corpus 的初始化 I/O，只在 logging/audit 啟用時執行。

- Empty retrieval：returned=0，uncertainty／margin=null。
- 只有一筆：可有 Top1／mean，margin=null。
- 兩側 normalization 無法證實：保留 raw IP，Top1／mean=null；可有 raw score margin。
- 無 score 的自訂 retriever：記錄 `scores_unavailable`，不假造 cosine。
- Retrieval exception：保存 failed retrieval event，propagate 原錯誤，不 retry。

每步有明確 route 與 `execution_masks`：retriever／extractor／qa／aggregate。Mask 表示該 component 是否實際被呼叫（包括 failed attempt）；完成與否看 status。Aggregate 沒有虛構 retrieval/extractor event；空 retrieval 沒有 extractor call。未執行 component 沒有 NLL，不能補零。

## NLL utility

`src/uncertainty.py::nll_from_logprobs()` 接受實際 generated token logprobs，計算負 logprob 的 sum／mean／max／population std／token_count。空序列回 empty，無 records 回 missing，未支援回 unsupported；non-finite、正 logprob、錯誤型別回 invalid。Unknown provider shape 回 returned_unverified。

以 API 回傳的 token records 為分母；沒有 re-tokenization、補 EOS、teacher forcing 或 self-rated confidence。Structured Output 的 scope 為 full_response（實際回傳 JSON token records），不是 answer-field NLL；coverage count 與 provider eval_count 另存。真實 probe 發現少一筆 logprob，因此不可解讀為所有內部 tokens 的完整 NLL。詳見 migration。

## Experiment metadata

保存 configuration ID、backend、exact tag／digest、Ollama version、quantization、thinking、requested context、`/api/ps` 的 loaded context、temperatures、output mode、prompt hashes、model template hash／parameters、package versions、retriever/index metadata、Git SHA／dirty flag、implementation file hashes、dataset ID、UTC timestamp。

Metadata query 失敗時保存 `metadata_error`，不宣稱 provenance 已驗證；不為 metadata 失敗改 agent 路由。要做正式實驗應先確認這些欄位齊全。

## 驗證結果與 artifacts

最終離線 pytest 結果見 `outputs/validation/offline_tests.xml`。涵蓋所有 success variants、unknown／missing、兩個分支 routing/raw preservation、backend selection、mock Structured Output、missing logprobs／NLL numerical checks、retrieval features、empty／single result、actual synthetic CPU FAISS normalization/ranking、multi-document extraction、step alignment、logging off equivalence、JSON persistence、unique runs、retrieval errors。

最終結果：**133 passed in 6.08s**。實測／未驗證項目的總表見 [validation report](validation_report.md)。

一般 tests 不需 Ollama 或 GPU，並攔截 Ollama/urllib calls；使用 synthetic data 與 CPU FAISS。Tests 和 examples 明確關閉 remote LangSmith tracing。實機 probes 與 offline tests 分開執行。

`docs/examples/synthetic_trajectory.json` 是真正 instrumented graphs 配合 mock LLM／synthetic retriever 產生的範例，不是 Qwen3 實驗成績。可產生新的 example：

```powershell
.\.venv311\Scripts\python.exe scripts\synthetic_example.py --output outputs\examples\new_synthetic_trajectory.json
```

`scripts/audit_contract.py` 已驗證原 prompts、Pydantic output schemas、graph topology 保留。這是 declarations 的 static comparison，不是整個程式語義的形式化證明；routing equivalence 另有 branch execution tests。

## Changed files

修改：`.env.sample`、`.gitignore`、`pyproject.toml`、本機 ignored `.python-version`、`requirements.txt`、`main.py`、`agents/plan.py`、`agents/step_definer.py`、`agents/rag.py`、`agents/plan_executor.py`、`corpus/retrieve.py`、`src/utils.py`、`README.md`。

新增：

- `src/backend.py`、`src/invocation.py`、`src/ollama_api.py`、`src/success.py`、`src/telemetry.py`、`src/uncertainty.py`。
- `pytest.ini`、`requirements-dev.txt`、`tests/conftest.py`、`tests/test_backend.py`、`tests/test_success.py`、`tests/test_uncertainty.py`、`tests/test_retriever.py`、`tests/test_workflow.py`。
- `scripts/probe_ollama.py`、`scripts/synthetic_example.py`、`scripts/audit_contract.py`。
- 四份 audit/migration/instrumentation documents、本 synthetic example、validation report。

`outputs/` 已加入 gitignore；實機相容性證據與 JUnit XML 保存在本機，不 commit 大型資料或個別實驗 outputs。

## 明示 baseline differences／未驗證事項

已改變的 configuration：OpenAI→local Qwen3 GGUF、明示 thinking=false、context=8192、native JSON schema transport、logprobs request；新增 success equivalents 與 ambiguous fail-closed validation；legacy output path 添加 configuration ID。原有 component temperatures 保留。原 prompts、embedding model、index type、top-k、ranking、branch conditions 與 completion policy未重新設計。

沒有新增 stopping cap、uncertainty-aware intervention、correctness labels、predictor、HF scoring backend、dataset batch。Ollama path 沒有 generation retry；optional OpenAI 保留原 max_retries（planner default 2，其餘 5）。

尚未驗證真實 corpus 的 cosine、full pipeline context/peak VRAM、OpenAI optional backend 的實際 generation、跨模型 exact reproduction。既有 Windows dataset/index/GPU FAISS 問題見 Phase 0。Backend 已可做 component smoke，但完整一題 end-to-end 需先解決資產與 FAISS 相容性，且需使用者明確同意。

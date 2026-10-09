# Qwen3 migration 與實機相容性

本次實作位於原始 MA-RAG；新 baseline ID 為 `ma_rag_qwen3_8b`。模型、thinking、context 與 output transport 已改變，不能宣稱 exact reproduction of original OpenAI results。

## 實作與配置

六個 invocation sites 統一走 `src/backend.py` 與 `src/invocation.py`；model default 只在共用 factory 定義一次。Ollama 不需要 OpenAI key，也沒有 model fallback／pull。原 OpenAI backend 保留，支援 `OPENAI_API_KEY` 與 legacy `API_KEY`，本次沒有付費 API 呼叫。

```env
LLM_BACKEND=ollama
OLLAMA_MODEL=qwen3:8b
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_THINKING=false
OLLAMA_NUM_CTX=8192
LLM_TIMEOUT_SECONDS=180
LLM_LOGPROBS=true
TRAJECTORY_ENABLED=false
LANGCHAIN_TRACING_V2=false
LANGSMITH_TRACING=false
```

Existing `.env` 未覆寫；請依 `.env.sample` 更新所需設定。Logging 以 local sidecar 實作，不依賴 LangSmith。Tests／synthetic example／compatibility probe 明確關閉 LangSmith tracing，避免繼承 `.env` 的遠端 logging。

## 已實測（2026-10-09～10）

| 項目 | 實際結果 |
|---|---|
| Ollama server／CLI | 0.40.0 |
| Exact model | qwen3:8b，8.2B，GGUF Q4_K_M |
| Digest | `500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41` |
| Integration | langchain-ollama 0.3.6、ollama SDK 0.6.3、langchain-core 0.3.74 |
| Graph／schema packages | langgraph 0.6.5、Pydantic 2.11.7 |
| Thinking | ChatOllama `reasoning=False` 對應 wire `think:false`；原生 API 對照 probe 的 default 有 thinking、false 無 thinking |
| Native structured output | JSON schema generation 與 Pydantic parsing 通過 |
| Production QA component smoke | 使用原 QA prompts/schema、synthetic context；answer=2，raw success="True" → normalized yes，trajectory 寫入成功 |
| Native generated-token logprobs | **available**，plain／structured／thinking-disabled 都有實際 token+logprob records |
| Production ChatOllama logprobs | **available**，採用下面說明的 native format invocation |
| Context | 配置並在 `/api/ps` 驗證 loaded model 為 8192；model capacity 顯示 40960，兩者不同 |
| Model VRAM | `/api/ps` 的 qwen3 allocation：6,295,440,588 bytes（約 5.86 GiB），並非 entire pipeline peak |
| GPU | RTX 4070，12,282 MiB；probe snapshot total used 8,694～10,371 MiB，後者另有其他 loaded model；未卸載其他模型 |

完整原始證據在 `outputs/compatibility/qwen3_8b_production_probe_20261010.json`，另有其中 `production_qa_schema_and_logging.trajectory_path` 指向的 JSON。這些是 component compatibility probes，並非完整 MA-RAG end-to-end。

## Structured Output 與 logprobs 的實際差異

Pinned ChatOllama 的 `with_structured_output(..., include_raw=True)` 建立 RunnableMap；傳給外層 invoke 的 `stream=False, logprobs=True` 未送到 inner LLM。該 helper probe 有 JSON output，但沒有 logprobs，因此該次 `logprobs_status="unsupported"`，沒有補 NLL。

Production Ollama path 使用官方 `ChatOllama.invoke(messages, format=original_schema, stream=False, logprobs=True)`。之後用 `json.loads` 和原始 Pydantic schema 驗證。Extractor 保持純文字。沒有改 prompt、加第二次 generation、剝除 thinking text 或修復 malformed JSON。Malformed JSON／unknown success 會保留 raw output 並失敗。

API 與 SDK 的 token records 已分別處理；SDK 的 records 是 Pydantic objects，需要轉成可保存的 dict，數值未更換。Token count／rating／thinking 不用來推估 logprobs。

**Coverage 限制**：production QA probe 回傳 49 筆 logprobs，但 eval_count=50；49 個 token 字串連接後等於完整 JSON content。NLL 是 API 回傳的 generated-token records 的統計，包含 JSON syntax，未補 EOS／control tokens。保存 `reported_output_token_count` 與 `token_count_matches_usage=false`，不宣稱對所有內部 generated tokens 完整覆蓋。Default-thinking probe 的 records 也可能包含 reasoning，不能和 thinking-disabled baseline 混用。

尚未驗證 API logprobs 在 sampler／grammar constraint 前後的精確定義；不同 output modes 的 NLL 不應無條件直接比較。Field-specific NLL 尚未實作。

API 若未回傳 logprobs，保存 `unsupported` 與 null NLL；回傳未知形狀時保存 `returned_unverified`，不解析成假 NLL。已實測支援的版本不會被硬編碼成 unsupported。

## Hardware 與 teacher-forced alternative

本機短 context smoke 能載入 Qwen3 Q4_K_M；尚未共同載入原 GTE model、DPR corpus 與 GPU FAISS，因此不能保證 full pipeline 或長 context peak memory。32GB RAM 為使用者提供規格，未重新量測。初始 num_ctx=8192 是明示 baseline 設定；真實 prompt 長度及 Ollama truncation 行為仍須在單題測試確認，不宣稱 40960-context 配置已驗證。

未實作／載入 Hugging Face scoring backend。若未來需要 teacher-forced sequence NLL：8.2B BF16 weights 約 16.4GB，超過 12GB VRAM，尚未含 activations／KV／logits；8192×約15萬 vocab 的 float32 logits 本身約5GB。低位元量化、分段 scoring／CPU offload 也需要另行評估記憶體與時間。

HF BF16 或 bitsandbytes quantization 不等於目前 GGUF Q4_K_M。Weights、quantization、tokenizer、chat template、special tokens、prompt/context 必須逐一核對，現在沒有證據可宣稱相同。Teacher-forced scores 是重新對保存序列評分，不是原 generation 的 token logprobs。若要實作必須另行取得使用者同意；目前 native logprobs 已可收集。

## Windows setup／驗證

在專案根目錄使用 PowerShell；已存在的 `.venv311` 是此次驗證環境。新建環境可用 `uv venv --python 3.11 .venv311`，不要覆蓋正在使用的環境。此 repo 的 dependencies 仍以 requirements files 為準。

```powershell
uv pip install --python .venv311\Scripts\python.exe -r requirements-dev.txt
ollama --version
ollama list
Invoke-RestMethod http://localhost:11434/api/version
```

Ollama 需先以 Windows 應用程式啟動服務；若沒有服務，可在另一個終端執行 `ollama serve`。本機 model 已安裝，不需要下載。若另台機器沒有 exact tag，probe 會報錯；下載需由使用者自行決定，程式不自動 pull。

只跑離線測試：

```powershell
.\.venv311\Scripts\python.exe -m pytest -q
.\.venv311\Scripts\python.exe scripts\audit_contract.py
```

以下僅 small compatibility calls，會使用 GPU，不執行 dataset pipeline；output 路徑須是新的：

```powershell
.\.venv311\Scripts\python.exe scripts\probe_ollama.py --output outputs\compatibility\my_new_probe.json
```

OpenAI alternative 可設 `LLM_BACKEND=openai`、`MODEL_NAME`、key；本次只測 factory 建立，未驗證 paid backend 的 Structured Output 或成績。

## 單題 end-to-end readiness

Backend 與 instrumented graphs 的 synthetic execution 已可用；**原 main 完整 pipeline 尚未就緒**。需先提供原 index shards／確認其 normalization、Windows dataset path、DPR corpus store，並處理原始 GPU FAISS path 與 CPU-only FAISS package 不相容。這次未改 embedding model、重建 index、下載 corpus 或改成其他 retrieval backend。

這些前置條件完成且使用者明確同意後，才可執行一題；main 的 end_index 為 inclusive，需將 start_index=end_index。Legacy `--model` 僅為 label，LLM 仍由上述 env 決定。不在目前環境提供可直接誤觸 batch 的執行指令。

參考 primary API 定義：[Ollama request/response types](https://github.com/ollama/ollama/blob/main/api/types.go)、[ChatOllama reference](https://reference.langchain.com/python/langchain-ollama/chat_models/ChatOllama)、[Structured Output API](https://reference.langchain.com/python/langchain-ollama/chat_models/ChatOllama/with_structured_output)。本文件的支援結論以本機 probe 為準。

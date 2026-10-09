# Phase 0 + Phase 1 validation report

完成日期：2026-10-10（Asia/Taipei）。原始本地 commit：`daa94d2471ce83065eb7e969d242040556034ba6`；此次修改未 commit／push。

| 檢查 | 結果 | 證據 |
|---|---|---|
| Offline tests | **133 passed in 6.08s**；最終一輪無 warnings | `outputs/validation/offline_tests.xml` |
| Syntax compile | 通過 | `python -m compileall -q agents src scripts corpus main.py` |
| Diff whitespace | 通過；僅 Git CRLF informational warnings | `git diff --check` |
| Original prompt content | 相同 | `scripts/audit_contract.py` |
| Four Pydantic output schemas | Class AST 相同 | 同上 |
| Outer／executor／RAG topology | Node/edge declarations 相同 | 同上 |
| Ollama connectivity／exact installed tag | 通過；0.40.0／qwen3:8b | `outputs/compatibility/qwen3_8b_production_probe_20261010.json` |
| Thinking disabled | API `think:false` 與 ChatOllama `reasoning=False`；有 default-thinking control probe | 同上 |
| Structured Output | Native schema、ChatOllama helper、production QA schema 均成功 parse | 同上 |
| Generated-token logprobs | Native API 與 production ChatOllama path：available | 同上 |
| Logprob coverage | QA 49 records／eval_count 50；decoded records 覆蓋實際 JSON content，未補缺少 token | production QA trajectory |
| Real QA success serialization | Raw string True → yes；raw 保留 | production QA trajectory |
| Synthetic cosine／ranking | CPU FAISS IndexFlatIP normalized／non-normalized vectors 通過 | `tests/test_retriever.py` |
| Actual corpus cosine | **尚未驗證**：本機沒有原 shards | Phase 0 audit |
| Logging disabled equivalence | Output、model-call arguments、call count 相同 | `tests/test_workflow.py` |
| Complete real MA-RAG question | **未執行**，且目前有 dataset/index/GPU FAISS blockers | Phase 0 audit |

實機 API generated-token logprobs 與離線 numerical tests 分開驗證；未把 synthetic logprobs 當成 model output。Synthetic example 在 `docs/examples/synthetic_trajectory.json`，有 8 個 invocation records、2 個 steps、1 次 retrieval，metadata 明確標記 mocked。

目前可以進行 local backend/component smoke。完整一題 MA-RAG 需先準備原 index／dataset/corpus、解決 CPU-only FAISS 與原 GPU path 不相容，再由使用者明確同意執行。尚未證明 full pipeline peak memory、原始 OpenAI reproduction 或跨模式 NLL 可比性。

未執行 HotpotQA batch、teacher-forced scoring、failure prediction training、paid API calls。沒有更換 embedding model、重建 index、調整 top-k、rerank 或依 uncertainty 改路由。

詳細說明與 changed-file list：

- [Architecture／Retriever audit](phase0_audit.md)
- [Qwen3 migration／Windows setup](qwen3_migration.md)
- [Success serialization／fail-closed policy](success_serialization_audit.md)
- [Trajectory instrumentation／changed files](phase1_instrumentation.md)

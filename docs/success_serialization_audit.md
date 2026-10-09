# Success serialization audit

## 原始路徑與意義

`QAAnswerFormat.success` 是 string，description 指示 Yes／No。QA 與 Aggregate prompts 都透過這個 schema 輸出。兩處原本先 `.model_dump()` 成 `QAAnswerState`；RAG 的 `final_raw_answer` 再由 executor 加進 `PlanExecState.step_output`。主程式最終存成 JSON，之後的 `task_define` 使用最後一個 success 的 `.lower() == "no"` 控制停止。

`success` 是 agent 是否能回答該 subtask 的內部訊號，**不是 final answer correctness／gold label**。此次沒有實作 correctness labeling 或 failure prediction。

## 集中化規則

`src/success.py` 提供 `normalize_success()`、`validate_qa_output()` 與 `should_summarize()`。

| 接受的 raw value | Normalized |
|---|---|
| 字串 yes／true／1（忽略大小寫，去除前後 whitespace） | yes |
| Python／JSON boolean true、integer 1 | yes |
| 字串 no／false／0（相同 whitespace／case 規則） | no |
| Python／JSON boolean false、integer 0 | no |

型別刻意區分；float 1.0／0.0、其他數字、null、missing、空字串、maybe、probably yes、not sure、list／object 均不接受。沒有從 analysis／answer 推斷 success。

`canonical` 表示原值已是精確字串 yes／no；其他明確等價值為 `mapped`；無欄位為 `missing`；其他為 `invalid`。

## Wire schema 與 raw preservation

四個 Pydantic output classes 未改寫，API 的 success wire schema 仍是 string。Ollama transport 先接收原始 JSON object，再集中正規化：原始 `QAAnswerFormat` 暫時用 canonical success 驗證其餘欄位，之後將儲存的 success 還原為 raw 值。

這讓非標準 boolean／integer success 在 schema 驗證前可被觀察，不必將原始 wire schema 擴大成 union。僅 TypedDict state 的 success annotation 與附加 audit fields 擴充，並非新的 model output contract。

每個成功 QA／Aggregate 的 state 與 trajectory 都保存：

```json
{
  "success": "True",
  "success_raw": "True",
  "success_normalized": "yes",
  "success_normalization_status": "mapped"
}
```

Boolean raw 會保持 boolean，integer raw 保持 integer；不把它們全部轉成字串。完整 raw AIMessage／JSON content 另外保存。

## Routing 與 fail-closed policy

`should_summarize()` 以同一集中化 normalizer 讀最後一步 raw success，避免信任與 raw 不一致的額外 metadata。明確 no → summary；yes 且還有 steps → 繼續；所有 steps 完成 → summary。沒有依 confidence、NLL 或 retrieval score 改路由。

不合法 success 在 QA／Aggregate output boundary 立即拋 `SuccessValidationError`，不加入合法完成 step、不做下一步、不自動 summary、不 regenerate。Trace 保存 raw output、null normalized、missing／invalid status、structured error；executor 與 outer graph 的 exception propagation 保持原本方式，main 原本的 catch/print 也保留。Logging 關閉時仍會拋 error。

就算最後一個 step 已完成，未知 success 仍不能繞過 validation。這是明示 compatibility correction；原始已完成 step 的短路條件可能跳過檢查，或 bool 值在非最後一步引發 AttributeError。

## 已測與限制

離線 tests 對 QA 與 Aggregate 的所有列出 serialization equivalents、whitespace、大小寫、原型別保存、正確 routing、missing／ambiguous values、no retry 均通過。現有 yes／no behavior 有回歸測試。

實際 Qwen3 QA component probe 回傳 raw `"True"`，已保存並映射為 yes。完整真實 multi-hop run 仍未執行；OpenAI optional backend 未呼叫，所以沒有其新 transport 的實機驗證。

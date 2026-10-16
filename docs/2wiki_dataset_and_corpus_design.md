# 2Wiki Dataset Adapter：Phase 1 設計與實測

更新：2026-10-10。本次僅執行 Phase 1；Phase 2–6 未開始。

後續資料品質調查已新增獨立 qa_valid/evidence_valid 與明確 `evidence_policy='retain'`，詳見 [資料品質調查](2wiki_data_quality_investigation.md)。下文是 Phase 1 原始實測紀錄；strict 預設保留，annotation-only 問題現在可明確保留於 QA 評估，但不得用於 annotation-dependent evidence metrics。Phase 2 仍未開始。

## 來源、授權與原始資料

採用官方作者 [Repository](https://github.com/Alab-NII/2wikimultihop) 的 2021-04-07 sentence segmentation 修正版。官方 README 該日期段落直接連結 [data_ids_april7.zip](https://www.dropbox.com/s/ms2m13252h6xubs/data_ids_april7.zip?dl=1)，本次下載 HTTP 200，實際 bytes 與 Content-Length 均為 **258,968,175**。

- 下載完成時間：2026-10-09 16:48:22 UTC，即 2026-10-10 00:48:22 Asia/Taipei。
- SHA256：`95df2bf56fdabe034e27aebc580e02264232203cf52552f9efe8a919e5529eef`。
- 原檔位置：`data/2wiki/official_20210407/data_ids_april7.zip`。
- 原檔設為 Windows read-only；直接讀 ZIP，不解壓後覆寫、不重新序列化來源。
- `.gitignore` 新增 `data/2wiki/`，`git check-ignore` 已確認。
- [來源紀錄](2wiki_source_provenance.json) 保存 URL、版本、下載日期、原始檔名、大小、hash 與授權註記。SHA256 為本地實測，並非作者發布的 checksum。

官方 [LICENSE](https://github.com/Alab-NII/2wikimultihop/blob/main/LICENSE) 是 Apache-2.0，Copyright 2020 Xanh Ho；再散布需遵守其授權／著作權通知與修改標示條件。README 要求引用 Ho et al. (2020) 的 COLING 論文。archive 僅含四個 JSON，沒有另附 dataset LICENSE 或 NOTICE；因此記錄 repository 的授權，**不推論它重新授權所有第三方 Wikipedia 文字**。本階段不重新散布 archive，也未下載超連結 corpus、DPR、模型或 embeddings。

|ZIP member|未壓縮 bytes|用途|
|---|---:|---|
|train.json|707,810,660|有標註訓練資料|
|dev.json|57,614,142|有標註 validation 資料|
|test.json|53,838,398|官方未標註 test|
|id_aliases.json|17,501,406|保留原始 alias 資產；本階段不實作 evaluator|

## 真實 schema

已檢查 archive 內 **192,606** 筆，而非只依 README。三個 split 每筆均有以下相同 10 個 keys，沒有其他 keys；完整報告保存每個 split 的 field set/count 與 member SHA256。

|Raw field|實際型別與處理|
|---|---|
|`_id`|非空 str；原樣保留為 question_id|
|`question`|非空 str；保留原文字，不補問號|
|`answer`|str；train/dev 非空；test 全部為空字串|
|`type`|comparison、inference、compositional、bridge_comparison|
|`context`|list，元素為 `[title: str, sentences: list[str]]`；全部題目均為 10 entries|
|`supporting_facts`|list，元素為 `[title: str, sentence_index: int]`；index 從 0 起；test 為空 list|
|`evidences`|list of `[subject, relation, object]` 字串 triples；僅保留 annotation，不當 retriever 文件|
|`entity_ids`|str；train/dev 非空，test 全空；原樣保留，不拆成文件|
|`answer_id`|可選 alias annotation，實際 str 或 null；保留在 raw_example，null 不代表缺 gold answer|
|`evidences_id`|list of triples，部分有標註題目為空；保留在 raw_example|

**官方 test 是空佔位，不是實際刪除欄位**：answer/answer_id/entity_ids 全為 `""`，supporting_facts/evidences/evidences_id 全為 `[]`。normalizer 同時允許 test 省略這些標註 keys，但要求明確 split=test；不會從答案空白或檔名猜 split。test 的 gold_answer=None、correctness_evaluation_available=false、answerability=unknown；不能拿來建立 correctness/failure labels。若 test 帶非空標註，明確拒絕，避免誤選 split。

## Adapter 與 registry

`src/datasets/two_wiki.py` 提供：

- `normalize_example(row, split=..., provenance=..., row_index=...)`：嚴格驗證並建立獨立副本。
- `read_source(path, split=..., provenance=...)`：讀 raw records 供完整 audit；ZIP 以明確 split 選 member，亦支援本地 JSON/JSONL/Arrow reader。
- `iter_two_wiki(...)` / `load_two_wiki(...)`：驗證每筆並檢查 split 內重複 IDs。任何異常丟 DatasetValidationError，包含 row index、question ID 與 issue codes；不跳過、不修復。

統一輸出 question_id、question、gold_answer、question_type、source_split、context_entries（title/sentences）、supporting_facts（保留原 index）、evidence_annotations、entity_ids、raw_example、source_provenance；另外輸出 correctness_evaluation_available 與 answerability。

provenance 包含來源檔名／絕對路徑、SHA256、大小、split、row index，以及 ZIP member/hash/size。呼叫者可提供來源紀錄；若 archive SHA256 與本次驗證官方檔案完全相符，補上已驗證的 source URL/version；其他本地 fixture 不假稱為官方檔案。`validation` 僅明確映射到原 `dev` split，不改 question ID。

迭代介面目前會先載入一個 split 的 raw JSON array；不是 bounded-memory streaming parser。validation 逐 split 處理，避免三個 split 同時留存；需要最小記憶體 corpus builder 時須另外設計，不在本階段實作。

`src/datasets/__init__.py` 的 registry 接受明確名稱 `2wikimultihopqa`、`hotpotqa`、`kilt` 與既有 dataset names；HotpotQA/KILT 仍回傳原始 rows。既有 `2wiki` 名稱保留為 legacy raw loader，不自動轉成新 adapter。`src/utils.py` 的 active `load_dataset` 改為委派 registry，其餘既有 Phase 1.5 修改保留。

```python
import json
from src.datasets import load_dataset, agent_input

source = json.load(open('docs/2wiki_source_provenance.json', encoding='utf-8'))
# 整個 dev 因真實索引越界而明確報錯，不會自動回傳「過濾後」資料。
examples = load_dataset(
    '2wikimultihopqa', 'data/2wiki/official_20210407/data_ids_april7.zip',
    split='dev', provenance=source,
)
# 單一已驗證 example 的 graph input 僅含 original_question。
graph_input = agent_input(examples[0], dataset_format='2wikimultihopqa')
```

`agent_input` 使用 allowlist，只回傳 `{'original_question': question}`，不加入 gold answer、supporting facts、evidences、entity_ids、context 或 raw_example。保留 raw_example 是為了來源追溯，**不能直接把 normalized dict 傳給 graph**。本階段未把新 backend 接上 `main.py` 的完整 agent runner；原 CLI 與 graph 不變，整合 runner 屬後續 Phase 4。

## 全量 validation 結果

可審查的 [JSON Schema Validation Report](2wiki_schema_validation.json) 包含來源／版本／SHA256、每個 split 的統計、異常 code/count 及前 20 個異常題案例；全數資料均檢查，案例上限不影響計數。

|Split|樣本數／唯一 IDs|Context entries|Supporting facts|匹配 facts|匹配率|有索引問題的題目|
|---|---:|---:|---:|---:|---:|---:|
|train|167,454／167,454|每題 10|404,884|402,877|99.5043%|1,990|
|dev|12,576／12,576|每題 10|30,687|30,617|99.7719%|70|
|test|12,576／12,576|每題 10|0|0|N/A|0|

三個 split 內與跨 split 的 Question ID 重複數均為 **0**。沒有缺少任何上述 10 欄位；沒有空 context／空句子等其他 schema errors。合法 null：train answer_id=88,647，dev=1,973。合法 empty evidences_id：train=30,434，dev=4,512。test 的六個空標註欄位各有 12,576 筆，屬預期 unlabeled schema。

|Question type|train|dev|test|
|---|---:|---:|---:|
|comparison|51,963|3,040|2,986|
|inference|4,379|1,549|1,550|
|compositional|76,481|5,236|5,262|
|bridge_comparison|34,631|2,751|2,778|

Supporting-fact count distributions：train `{2:132512, 3:297, 4:34256, 5:389}`；dev `{2:9805, 3:20, 4:2739, 5:11, 6:1}`；test `{0:12576}`。

### 未解決的資料品質問題

官方修正版仍有 **2,007 train facts、70 dev facts** 的 sentence index 越界；分別涉及 1,990 與 70 題。所有不匹配均是 invalid_sentence_index，沒有缺 title 或 ambiguous-title errors。例：dev row_index=1123、question_id=`695689d20baf11ebab90acde48001122`，Supporting Fact `["Aditya Chopra",19]`，對應 context 只有 15 句，有效 index 為 0–14。

沒有調整 index、重切句子、刪題或過濾 facts；strict loader 載入完整 train/dev 會明確失敗。validation command 會完整統計並以 exit code **1** 回報資料不通過，而不是把它當成程式執行成功的 clean dataset。合法列雖然可獨立 normalize，但未建立任何篩選後研究 split。

「匹配成功」只驗證 title 唯一對應、index 型別與範圍；不代表人工確認該句在語意上支持答案。未進行語意校正、unanswerability 重標註或 aliases correctness 評估。

## 驗證命令與測試

從 repo root 使用 `.venv311\Scripts\python.exe`：

```powershell
.\.venv311\Scripts\python.exe scripts/validate_two_wiki.py --archive data/2wiki/official_20210407/data_ids_april7.zip --provenance docs/2wiki_source_provenance.json --output outputs/2wiki_validation_new.json
```

腳本拒絕覆寫已有 report；重跑請使用新 output 路徑。`scripts/inspect_two_wiki_source.py` 可另外輸出 member／欄位型別與真實首筆樣本，預設不修改來源；本次 inspection 在 ignored data 目錄。

新增 `tests/test_two_wiki.py` 的 37 tests，涵蓋四種 question types、正常 schema、raw 保留／副本隔離、缺欄位／answer、空 context、malformed facts、bool/負數/越界 index、title missing/ambiguous、重複 IDs、test placeholders/缺標註、split 誤用、gold leakage、ZIP/member SHA、missing-file、不得 silent skip、HotpotQA/KILT regression 與 validation 統計。

完整離線 suite 初次 **170 passed in 5.86s**；最終 provenance／validator robustness 調整後重跑為 **170 passed in 5.83s**，JUnit 位於 `outputs/phase1_two_wiki_final_tests_20261010.xml`。Windows sandbox 暫存目錄限制沿用先前已知問題，取得 sandbox 外執行許可後測試；沒有呼叫 Qwen3、OpenAI 或完整 agent graph。

`scripts/audit_contract.py` 五項 preserved=true：原 prompts、四個 Pydantic schemas、graph 宣告不變。HotpotQA/KILT loader 保留原 rows。此次只增加資料介面與檢查，沒有改 agent 語意、success normalization、uncertainty 計算或 trajectory 行為。

## 檔案清單與 Phase 2 前置條件

本次新增：`src/datasets/__init__.py`、`src/datasets/two_wiki.py`、`scripts/inspect_two_wiki_source.py`、`scripts/validate_two_wiki.py`、`tests/test_two_wiki.py`、本文件、`docs/2wiki_source_provenance.json`、`docs/2wiki_schema_validation.json`。修改 `.gitignore` 與 `src/utils.py` active loader。其餘原有未提交 Phase 1.5 檔案保留，不代表本次重新實作。

Phase 2 開始前需要使用者批准，且先決定：

1. 如何處理 2,077 筆越界 facts：保留所有題與 context，但將有問題 annotation 標示不可評估；或其他經批准的研究政策。不能默默改 index 或刪題。若採保留政策，需另設明確 annotation-validity 狀態，不能放寬本階段 strict validation 而無紀錄。
2. corpus policy 選 `train_only` 或 `train_plus_validation`；本階段未選定／凍結。若加入 dev contexts，研究須標明 pooled-context benchmark。
3. passage title/text 拼接、sentence boundaries、normalized-content 去重與 stable doc ID 規則；不得以 gold/support labels 選文件。
4. corpus builder 第一個工作僅 dry-run statistics 與 RAM/disk 估算；尚未建立 pooled corpus、embeddings 或 FAISS index。

Phase 1 程式與資料 audit 完成；資料不是全數 clean，品質問題如實保留。停止於此，等待 Phase 2 授權。不 commit、不 push。

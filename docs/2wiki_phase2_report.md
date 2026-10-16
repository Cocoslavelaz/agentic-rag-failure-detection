# Phase 2 完成報告：Fixed Pooled Corpus

日期：2026-10-10。已完成 Phase 2，未開始 Phase 3–6。

## 主要結果

主要政策 `train_plus_validation`，納入官方 2021-04-07 修正版 train/dev 的每個 context entry，未使用 test。研究定位為 pooled-context adaptation，非 DPR-W100 open-corpus benchmark。

|統計|數量|
|---|---:|
|train 題目／context entries|167,454／1,674,540|
|dev 題目／context entries|12,576／125,760|
|總 context entries／來源映射|1,800,300|
|最終 unique passages|384,857|
|Duplicate occurrences|1,415,443|
|原始表示 variants|384,857|
|train unique passages|369,378|
|dev unique passages|54,957|
|train/dev shared unique passages|39,478|
|只出現在 train／只出現在 dev|329,900／15,479|
|落在跨 split 共用文件的原 entry occurrences|1,128,558|

Duplicate occurrences 定義為 total entries − unique passages；shared 定義為 canonical document 同時出現在兩個 splits，不是重複 question IDs。train 內 duplicate occurrences=1,305,162，dev 內=70,803，另有 39,478 個跨 split 文件重疊，合計 1,415,443。

所有 2,060 題 evidence-invalid 樣本的 **20,600 個 context entries** 納入（train 19,900、dev 700），沒有因 invalid supporting facts 選擇／排除任何段落；flags 與原始來源映射仍保留。沒有依 gold answer、supporting facts 或 correctness 選文，也沒有計算 evidence metrics。

## 正式資產與 manifest

Corpus checksum：

`2543427d7c42ac82a88b6e8b4b2ff0063495a17dfefa72013e3f6cdacaf38b72`

正式目錄：`artifacts/2wiki/2543427d7c42ac82a88b6e8b4b2ff0063495a17dfefa72013e3f6cdacaf38b72/`。

- [Manifest 快照](2wiki_corpus_manifest.json)：與正式 manifest bytes 相同，可在 Git 中追蹤小型 metadata。
- [完整格式／算法規格](2wiki_corpus_manifest_schema.md)：canonicalization、document IDs、source mappings 與 corpus checksum 定義。
- [Dry-run、build、verification 與資源 JSON report](2wiki_corpus_build_report.json)。

|檔案|實際 bytes|
|---|---:|
|documents.jsonl|612,744,577|
|source_variants.jsonl|575,413,448|
|source_mappings.jsonl|784,436,330|
|provenance.json|903|
|manifest.json|2,789|
|合計|1,972,598,047（約 1.84 GiB）|

來源 ZIP SHA256：`95df2bf56fdabe034e27aebc580e02264232203cf52552f9efe8a919e5529eef`，原始檔仍唯讀，建置／驗證前後一致。資料檔各自 SHA256 已存 manifest；沒有 embeddings 或 index 檔案。

## Dry-run 與資源

先以 `ijson` 串流讀取 ZIP，在 SQLite disk state 精確去重，不保存多份全量 JSON/list。Dry-run 預先計算輸出 bytes，不是抽樣估計 unique 數；在向使用者回報以下資源後才匯出正式 corpus。

|項目|Dry-run／預估|實際結果|
|---|---|---|
|來源 entries／unique|1,800,300／384,857|一致|
|資料檔 bytes（不含 manifest）|1,972,595,258|完全一致|
|SQLite 暫存|3,429,130,240 bytes（3.19 GiB）|保留於 ignored outputs 供追溯|
|Dry-run runtime|—|117.79 秒|
|Export runtime|約 94 秒，假設 20 MiB/s，另加排序開銷|16.13 秒|
|Dry-run peak RAM|實測 103,157,760 bytes；保守預估 256 MiB|約 98 MiB|
|Export peak RAM|低於 256 MiB 預估|68,263,936 bytes，約 65 MiB|
|完整追溯 verification|—|91.04 秒；peak 67,301,376 bytes，約 64 MiB|
|建置前可用磁碟|461,741,916,160 bytes，約 430 GiB|驗證後約 425 GiB 可用|

正式 corpus + prepared SQLite 約 5.03 GiB，另有原 archive 約 247 MiB 與少量 reports/tests。既有輸出與暫存未刪除。沒有 GPU 工作，沒有新套件／資料／模型下載；ijson==3.5.1 與 psutil==7.2.2 為本機既有套件，新增 requirements pins。

## ID 與 provenance tracing

`document_id = '2wiki-v1:' + SHA256(canonical_json([NFC/whitespace-normalized title, NFC/whitespace-normalized full joined text]))`。相同 title 不同內容保留不同 doc。hash 命中比較完整 canonical content；不同即停止，forced collision test 已通過。

文件的 source references 分開存 relational JSONL，而非巨大內嵌 list：

`documents.document_id → source_mappings(document_id, split, question_id, context_position) → source_variants.variant_id`。

每個 mapping 亦保存 row index、source_provenance_id、qa_valid/evidence_valid 與排除理由；provenance 檔包含 archive/member SHA256、版本、URL。variant 保存原 title、原 sentence list、original_text 與 embedding_text，確保去重時不丟失原句界或空白差異。指定 embedding text 為原 `title + LF + " ".join(sentences)`，不另切 chunks。

同一政策以固定 train/dev、原 row/context order 選代表版本；資料輸出按 ID／來源鍵排序，UTF-8 canonical JSON + LF。manifest 不含執行時間或環境路徑。

## 驗證與測試

- 新增 16 個 corpus tests，完整 suite **198 passed in 6.46s**；JUnit：`outputs/phase2_final_tests_20261010.xml`。
- 涵蓋排序、stable IDs、完整內容去重、同 title 異文、Unicode/whitespace normalization、原句界 variants、多 provenance、train/dev overlap、train_only、invalid facts 納入、QA-invalid 不選文、forced hash collision、缺檔、來源 hash 不符、duplicate IDs、malformed contexts、重建一致性、gold leakage prevention、HotpotQA regression、完整性 verifier 與 tampering 拒絕。
- 小型 fixture 從相同 inputs **獨立 prepare/export 兩次**，所有四個資料檔與 manifest bytes、corpus checksum 完全一致；同政策 ID 排序一致，跨 policy 共用文件 ID 穩定。
- 正式資料完整 read-back：所有資料檔 hashes 相符，按排序逐列比對 prepared records；重新串流原 ZIP，逐一驗證 **1,800,300 個 context entries** 的原 title/sentences 可由 mapping/variant 還原，且 document ID 與 canonical content 相符。
- 正式全量 corpus 本次建置一次，未額外重做第二份全量 prepare/export；重建一致性以獨立 fixture builds 驗證，不把 read-back 宣稱為第二次建置。
- 靜態 contract audit 五項 preserved=true；原 prompts、四個 Pydantic schemas、graph 宣告不變。

## 本次檔案變更

新增 `corpus/build_pooled_corpus.py`、`scripts/verify_pooled_corpus.py`、`tests/test_pooled_corpus.py`、`docs/2wiki_corpus_manifest_schema.md`、`docs/2wiki_corpus_manifest.json`、`docs/2wiki_corpus_build_report.json`、本文件。修改 `.gitignore`（artifacts/2wiki）與 `requirements.txt`（已安裝套件 pins）。

原有 Phase 1/1.5 及資料品質調查未提交修改全部保留；未改 Agent prompts、schemas、graph、uncertainty 計算或 HotpotQA 路徑。原始資料與既有 corpus/outputs 未覆寫。

## 剩餘問題與停止點

原始 2,077 筆越界 supporting facts 與歷史句子切分成因仍未修復／確認；其 context 全數納入，annotation-dependent evidence evaluation 未來依 evidence_valid 排除，不能改小分母或自動修正。對於 `train_only`，dev evidence 是否在 corpus 需未來獨立評估 availability，不能把 unavailable 與 not-retrieved 混為一談。

尚未驗證 GTE 模型、tokenizer、cosine 或 FAISS；這些屬 Phase 3，未執行。未呼叫 Ollama/OpenAI、未執行 graph、未 commit/push。Phase 2 完成後停止，等待 Phase 3 授權。

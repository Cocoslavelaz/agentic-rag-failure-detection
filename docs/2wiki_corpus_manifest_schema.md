# Phase 2 固定 pooled corpus：格式與重現規格

Builder version：`2wiki-pooled-v1`，manifest schema_version：`1`。此規格只涵蓋 corpus；沒有 embeddings、FAISS index 或 retrieval 執行。

## Corpus policy 與來源

- `train_plus_validation`：主要政策，依固定順序讀官方 `train.json`、`dev.json`，不讀 `test.json`。
- `train_only`：可選比較政策，只讀 train；已以小型 fixture 驗證，不額外建立完整比較 corpus。
- 納入指定 splits 的每個 context entry。不以 gold answer、supporting facts、QA correctness、qa_valid 或 evidence_valid 選文件。
- flags 僅保存在來源映射供後續資料品質／評估分母使用，不是 retrieval features。
- 遇到無法追溯的缺 ID、重複 split 內 ID、malformed context、空句子列表或 hash collision，停止並報錯，不默默跳過。

主要政策明確包含 dev contexts，是 pooled-context adaptation，不是 unseen-document 或 DPR-W100 benchmark。

## Passage、canonicalization 與 IDs

一個原 context entry 對應一個 passage，不另切 chunks。保留原 title 和原順序 sentence list；定義：

```python
original_text = ' '.join(sentences)
embedding_text = title + '\n' + original_text  # 真正 LF，非兩字元反斜線+n
canonical_title = ' '.join(unicodedata.normalize('NFC', title).split())
canonical_text = ' '.join(unicodedata.normalize('NFC', original_text).split())
canonical_payload = canonical_json([canonical_title, canonical_text])
content_hash = sha256(canonical_payload.encode('utf-8')).hexdigest()
document_id = '2wiki-v1:' + content_hash
```

canonical JSON 使用 UTF-8、ensure_ascii=false、sort_keys=true、compact separators、allow_nan=false。JSON array 分隔 title/text，避免字串串接歧義。不 lowercase、不移除標點、不只依 title 去重。相同 title 不同 canonical text 保留不同文件；同 canonical content 多次出現只有一個 document，但每個來源 entry 都保留 mapping。

document hash 命中後，比較完整 canonical payload；不同則丟 `HashCollisionError`，不合併、不覆寫、不任意追加不穩定 suffix。raw variant hash 也比對完整 raw payload。

`variant_id = 'variant-v1:' + SHA256(canonical_json([original_title, original_sentences]))`。這個 hash 不作 NFC/whitespace 正規化，因此保留 canonical 等價但原標題、空白或句界不同的來源版本。文件 representative 為固定 split/row/context order 的首次出現版本；其餘版本仍由 variant 與 mapping 保存。相同輸入和配置重建穩定；ID 本身不依政策、row number、路徑或時間。

原始 entry 沒有獨立 paragraph text 欄位；`original_text` 是依指定規則串接，無法宣稱恢復上游 Wikipedia 未提供的空白。原 sentence 字串完整保存，不 trim、不重切句。

## 生成檔案

資產目錄：`artifacts/2wiki/<corpus_checksum>/`；已加入 Git ignore。

|檔案|內容與順序|
|---|---|
|`documents.jsonl`|每個 canonical document 一列；document_id、content_hash、title、sentences、original_text、embedding_text、representative_variant_id；依 document_id 排序|
|`source_variants.jsonl`|每個未改寫 raw variant 一列；variant_id、document_id、原 title/sentences/text/embedding_text；依 variant_id 排序|
|`source_mappings.jsonl`|每個原 context entry 一列；document_id、variant_id、source_split、source_question_id、context_position、source_row_index、source_provenance_id、qa_valid/evidence_valid 與排除理由；依 doc_id/split/question_id/position 排序|
|`provenance.json`|以 split 為 key，保存 dataset URL/revision/SHA256、原檔名、archive member/hash/size；不加入環境絕對路徑或執行時間|
|`manifest.json`|以下 manifest schema 與 corpus checksum|

每列使用上述 canonical JSON + 單一 LF，沒有 BOM。JSON 內的原始換行以 JSON escaping 保存。來源集合是 relational representation：document_id 關聯多筆 mappings，再由 variant_id 取回每個來源的原始表示；不是把大量 references 塞入一個巨大 document list。

以 `(source_split, source_question_id, context_position)` 唯一定位原 context；row index 為零起算，提供原 ZIP JSON 的直接定位。由 source_provenance_id 取得 source/member SHA256。即使句界正規化後相同，也必須沿 variant_id 取回原始 sentence indices，而非套用 representative 的句界。

文件資料不加入 answer、question、supporting_facts、evidences 或 entity_ids。context 自然出現答案文字不作刪除。mapping/provenance 是離線追溯資料，後續 Retriever 應只用 document text；不得將 flags、source question ID 或 annotation 排除理由餵給 agent／predictor 作 uncertainty features。

## Manifest 欄位

|欄位|型別／規則|
|---|---|
|schema_version|int，現在為 1|
|builder_version|str，現在為 2wiki-pooled-v1|
|dataset / dataset_source_url / dataset_revision / dataset_sha256|來源名稱、作者下載 URL、版本描述、原 ZIP SHA256|
|included_splits / corpus_policy|有序 list[str] 與政策名稱，必須一致|
|benchmark|pooled-context adaptation 的研究定位|
|passage_granularity / embedding_text_rule|context-entry 粒度與原文拼接規則|
|canonicalization / document_id_algorithm / raw_variant_algorithm|正規化、hash 與 collision handling 規則|
|representative_rule / serialization|來源 representative、檔案列排序、JSON 與 LF 規則|
|data_quality_policy|所有 contexts 納入，flags 不作選文條件|
|total_context_entries|int，來源 mappings 的列數|
|unique_passages / unique_raw_variants|int，documents 與 variants 的列數|
|duplicate_count|total_context_entries − unique_passages；重複出現次數，不是重複群組數|
|split_statistics|每個 included split 的 questions、context entries、qa_valid_questions、evidence_invalid_questions、evidence_invalid_context_entries_included|
|cross_split_overlap|每 split unique 數、shared_unique_passages、source_entries_on_shared_passages 與定義|
|output_file_checksums / output_file_sizes_bytes|四個資料檔的 SHA256 與精確 bytes，不包含 manifest 本身以避免循環|
|corpus_checksum|SHA256(canonical_json(除 corpus_checksum 外的完整 manifest))|

不在 manifest 放時間、runtime、RAM、staging path 或 SQLite 的物理 hash；這些存在 build report，避免污染重建 checksum。manifest.json 自身 checksum 可在外部驗證報告另記。

## Bounded-memory dry-run 與安全發布

`ijson` 串流讀取 ZIP array，每次只留一題；SQLite 保存完整去重鍵、raw variants、source mappings。SQLite cache 設 32MiB、temp_store=FILE、禁止 mmap；每 2,000 題 commit。沒有多份全量 dataset 或 GPU allocation。

Dry-run 建立 ignored `outputs/phase2/.../prepared.sqlite`，精確計算 unique/counts 與三個 JSONL 輸出 bytes，**不是用抽樣推估 unique 數**。先檢查可用磁碟與 RAM；正式 build 僅從已審查且 SHA256 符合的 prepared DB 匯出。來源 archive hash、policy 不符就拒絕。匯出前另確認磁碟安全餘裕。

先寫 staging、計算檔案 hashes 與 corpus checksum，完整後才 rename 到 checksum 目錄。已有正式目錄時逐檔驗證相同，不覆寫；若相同，保留此次 staging 並回報位置。碰到既有 staging 不清除，請使用新 work/output root 重建。中斷或碰撞留下的暫存不會當成正式 corpus，也不自動刪除。

正式 corpus 不保存 SQLite；它是 dry-run／verification state，留在 outputs 供追溯。可依小型 fixture 測試從相同 archive 各自 prepare/export 兩次，比對所有資料檔與 manifest bytes。

## Windows 執行方式

使用既有 `.venv311\Scripts\python.exe`。ijson==3.5.1、psutil==7.2.2 已在本機且加入 requirements pins；本次未安裝／下載新套件。

```powershell
.\.venv311\Scripts\python.exe corpus/build_pooled_corpus.py --mode dry-run --archive data/2wiki/official_20210407/data_ids_april7.zip --provenance docs/2wiki_source_provenance.json --policy train_plus_validation --work-dir outputs/phase2/new_dryrun --report outputs/phase2/new_dryrun.json
# 審查 dry-run 資源後才執行：
.\.venv311\Scripts\python.exe corpus/build_pooled_corpus.py --mode build --archive data/2wiki/official_20210407/data_ids_april7.zip --provenance docs/2wiki_source_provenance.json --policy train_plus_validation --dry-run-report outputs/phase2/new_dryrun.json --report outputs/phase2/new_build.json
```

報告與 work-dir 不覆寫既有資料。`train_only` 使用新 work-dir 並顯式改 policy；test 不提供可選 corpus policy。

## 後續 evidence evaluation 的界線

本階段不計算 gold evidence metrics。未來離線 evaluator 必須先檢查 evidence_valid，再用原 split/question/position → document_id mapping 判斷 corpus availability：

1. annotation invalid：整題不進 annotation-dependent metrics，記錄原因與排除數。
2. annotation valid 但來源文件不在所選 corpus：evidence unavailable in corpus。
3. annotation valid、文件在 corpus 但未於實際 retrieval 返回：present but not retrieved。

不能把以上三種情況都當成一次 retrieval failure，亦不能修改現有 uncertainty 計算。Phase 2 沒有檢索分數、cosine、EM/F1 或模型呼叫。

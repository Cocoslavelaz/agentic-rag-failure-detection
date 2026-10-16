# Phase 3A：GTE Embeddings 與 CPU FAISS 工程驗證

日期：2026-10-10。Phase 3A 已完成；未執行 Phase 3B、完整 Agent Graph、Qwen3、OpenAI API 或正式 retrieval-quality evaluation。

## 已驗證結果

固定抽取 Phase 2 corpus 的 128 筆真實文件，以本地 GTE 編碼並建立、儲存、重新載入 CPU `IndexFlatIP`。使用官方原始 dev 的前五個問題查詢；五組 Top-10 排序均與 NumPy float64 cosine 的獨立計算一致。最大絕對誤差 **9.768775510732297e-8**，低於 `1e-5`。

離線測試 **225 passed**：原有 198 項、新增 27 項。原有 HotpotQA、資料品質政策、corpus 與 uncertainty 回歸測試保持通過。

本次 smoke 是工程驗證。128 筆隨機文件並不保證包含查詢答案或正確證據，因此不能用這些結果判定完整 corpus 的 retrieval 品質。

可追溯數據：

- [精簡驗證 JSON](2wiki_phase3a_validation.json)：設定、所有五組 Top-10、cosine 比對、資源估算及測試摘要。
- 原始 integration report：`outputs/phase3a/real_gte_integration.json`，含每筆 token 長度、batch 時間與完整 token histogram。
- 原始 report SHA256：`05d6f8d1a019d3e58c1a15c1598bf974e466ea7b6e5d4112dcdee1f3743f3b76`。

## Phase 2 輸入確認

沒有重建或修改 Phase 2 corpus。使用 `artifacts/2wiki/<corpus_checksum>/` 中的既有 `manifest.json` 與 `documents.jsonl`，重新驗證 manifest version、checksum、documents SHA256、384,857 筆計數、ID 唯一性與排序，以及 embedding text 規則。

- Corpus policy：`train_plus_validation`；未使用 test。
- Corpus checksum：`2543427d7c42ac82a88b6e8b4b2ff0063495a17dfefa72013e3f6cdacaf38b72`。
- Documents SHA256：`ecc144615441cc7bbb14e2a0b2cad16984d515bcd41ec536c50d6becfd84ffac`。
- ID：保留既有 `2wiki-v1:<content_hash>`。
- Text：原始 `title + '\n' + ' '.join(sentences)`。
- Evidence-invalid 問題的 context 仍在固定 corpus 中，沒有因 gold annotations 排除文件。
- 原有 `hf/gte-multilingual-base` 不存在，無法證明其與本次官方模型完全等價；本次為明確記錄設定的 2Wiki adaptation。

## 模型來源、程式碼審閱與設定

|項目|本次值|
|---|---|
|Model ID|`Alibaba-NLP/gte-multilingual-base`|
|Model / tokenizer revision|`9bbca17d9273fd0d03d5725c7a4b0f6b45142062`|
|Remote code repository|`Alibaba-NLP/new-impl`|
|Code revision|`40ced75c3017eb27626c9d4ea981bde21a2662f4`|
|授權|官方 model card 與 code headers：Apache-2.0|
|Pooling|`last_hidden_state[:, 0, :]`，768 維 CLS|
|Document / query token limit|512 / 512，包含 special tokens|
|輸入前綴|無；query 與 document 共用相同模型、tokenizer、pooling 與 normalization|
|Normalization|L2；float64 獨立檢查 norm，容許誤差 `1e-5`|
|Model / embedding dtype|float32 / float32|
|Attention|本地審閱程式碼的 padded SDPA；未啟用 unpadding 或 xformers|
|Batch / threads / device|8 / 4 / CPU|
|隨機 seed|43|
|Encoder config hash|`69ba38882ab9878a0d7e3ac3e62a177670c61a18423dff3f5346fddbd0f2ae86`|

官方來源：[固定版本 model card](https://huggingface.co/Alibaba-NLP/gte-multilingual-base/blob/9bbca17d9273fd0d03d5725c7a4b0f6b45142062/README.md)、[model config](https://huggingface.co/Alibaba-NLP/gte-multilingual-base/blob/9bbca17d9273fd0d03d5725c7a4b0f6b45142062/config.json)、[固定 code revision](https://huggingface.co/Alibaba-NLP/new-impl/tree/40ced75c3017eb27626c9d4ea981bde21a2662f4)。官方 dense encoding 範例使用 CLS 與 normalization；model card 範例的展示分數乘 100 沒有帶入本次索引。

模型 config 的 position limit 為 8192，tokenizer metadata 為 32768；本次明確固定 512，沒有以 tokenizer metadata 自動擴張。原 MA-RAG doc 512 / query 8192 及舊 slicing 行為保持原狀，新的共用 encoder 只供 adaptation 使用。

在執行模型前已讀完整固定 revision 的 `configuration.py`（7,127 bytes）與 `modeling.py`（59,023 bytes），檢查 imports 與所有類別定義。其模型路徑使用 Torch/Transformers tensor operations，未發現下載、subprocess、eval/exec 或 inference 中的檔案執行；optional xformers 路徑沒有啟用。官方 AutoModel 配置需要 remote code；本次直接載入經 SHA256 驗證的本地 reviewed code，不呼叫可變動的遠端程式碼載入。Tokenizer 使用內建 fast tokenizer，`trust_remote_code=False`、`local_files_only=True`。

下載日期 2026-10-10；下載前核對官方來源、commit、授權、大小及可用磁碟約 425 GiB。只下載必要 config、tokenizer、safetensors 與兩個 reviewed code 檔，共 **627,905,786 bytes**；其中權重 **610,753,338 bytes**。每個檔案的固定 URL、大小及 SHA256 記於 [gte_model_lock.json](gte_model_lock.json)，下載後及模型載入前再次核對。模型保存在 `models/gte-multilingual-base/<model_revision>/`，已排除 Git。

模型實際載入沒有 missing/mismatched keys 或 errors。只有 `classifier.weight`、`classifier.bias` 未使用；dense `NewModel` 不使用分類 head。文件與查詢 forward 均成功。

## Windows 相容性與索引契約

目前環境的 Torch 與 FAISS 啟動計算時會發生 OpenMP Error #15：`libomp140.x86_64.dll` 與 `libiomp5md.dll` 同時初始化。沙箱內外都能重現，調整 import 順序無法解決。

新 adaptation 使用程序隔離：GTE 程序只執行 Torch；獨立 FAISS worker 只載入 FAISS/NumPy，使用套件 metadata 記錄 Torch/Transformers 版本。沒有設定 `KMP_DUPLICATE_LIB_OK`、替換 DLL 或變更現有環境。Windows pipe 通訊明確使用 UTF-8；索引 I/O 使用 Python bytes 與 Windows extended paths，已測試中文及超過 MAX_PATH 的路徑。

FAISS 為 CPU `IndexFlatIP`。Builder 只驗證已正規化的原向量，再原樣加入；沒有重新 normalize、乘 100、clip 或重排 scores。載入時核對 corpus checksum/manifest hash、完整 encoder config/hash、完成標記、各 asset hash、index type/dimension/ntotal、mapping count/row order/ID 唯一性，以及 persisted embeddings 與 FAISS reconstructed vectors 完全一致。

`-1` row 明確記為 `faiss_missing_row`，不放入 hits，不製造 score=0。Rank 保留 FAISS 原始位置；未檢索的 aggregate branch 保留 mask=false 與 None 分數。

正式產物與 smoke 分開；本次完整目錄為：

```text
artifacts/2wiki/
  2543427d7c42ac82a88b6e8b4b2ff0063495a17dfefa72013e3f6cdacaf38b72/
    69ba38882ab9878a0d7e3ac3e62a177670c61a18423dff3f5346fddbd0f2ae86/
      smoke/
        ff05fe65276cb06d13374af1763068141ec7d4379f8b0a107866dcb04cb07f19/
```

`index_manifest.json` 固定 `scope=smoke`、`research_retrieval_index=false`、`build_status=complete`；含要求的 corpus、model/code/tokenizer、pooling、dtype、L2、limits、seed、batch、device、versions、asset hashes。`build_complete.json` 綁定 manifest SHA256；任何未完成或版本不符的 index 都拒絕載入。Phase 3A builder 強制最多 1,000 documents。

|Asset|實際 bytes|
|---|---:|
|`embeddings.npy`|393,344|
|`index.faiss`|393,261|
|`document_mapping.jsonl`|13,330|
|`selected_documents.jsonl`|183,383|

Index SHA256：`a5f262ce3ee8d62d4680d7d2e4acfa14c0c9ae46ff75ae0f487fdceaf5ee75f9`。
Mapping SHA256：`c760edca4633abbf72dc4fa3b10eb801c7eee6314a4a7004e331436fce507395`。
早期失敗的測試目錄保持原狀；報告只採上述 seed=43、有完成標記且通過重新載入驗證的產物。

## 真實小樣本與 cosine 驗證

抽樣規則：對每個 ID 計算 SHA256(`43:<document_id>`)，取 hash 最小的 128 筆，再按 document ID 排列，完整記錄選取 ID 與 selection hash。抽樣不讀取答案或 supporting facts。

|測量|結果|
|---|---:|
|Document encoding|128 筆 / 38.77735 秒；3.30090 docs/sec|
|Document norm 範圍|0.9999998901–1.0000001006|
|Document 被截斷數|1 / 128|
|Query encoding|5 筆 / 0.24551 秒；零截斷|
|Query norm 範圍|0.9999999751–1.0000000640|
|Index build/persistence|0.04246 秒，不含程序啟動與模型編碼|
|獨立 cosine 比對|5 queries × 10 hits = 50 scores|
|最大絕對誤差|9.768775510732297e-8|
|Top-10 exact row order match|5 / 5|

獨立驗證用重新載入 index 後 reconstruct 的真實向量，轉 NumPy float64 計算 `dot(q,d)/(norm(q)*norm(d))`，並對全部 128 documents 計算 cosine 排序，再與 FAISS Top-10 比較。沒有為通過門檻而修正分數或排序。Tie policy 允許同分文件順序不同，但本次五組均 exact match。

第一個 dev 問題：`8813f87c0bdd11eba7f7acde48001122`，原文 `Who is the mother of the director of film Polish-Russian War (Film)?`。其完整 Top-10 如下；document ID 可由同列 row ID 對應固定 mapping，所有完整 ID 另載於 JSON report。

|Rank|Row ID|Title|Raw FAISS score|
|---:|---:|---|---:|
|1|59|The Pianist (2002 film)|0.6217153668|
|2|21|Paul Weitz (filmmaker)|0.5378793478|
|3|55|Marya Chéliga-Loevy|0.5362129211|
|4|126|Cléo from 5 to 7|0.5316736698|
|5|123|The Sino-Japanese War at Sea 1894|0.5259764194|
|6|78|Dissolved and Effused|0.5143499374|
|7|13|Clio Barnard|0.5097658038|
|8|45|Jean-Jacques Beineix|0.5057801008|
|9|26|Jag Mundhra|0.5010714531|
|10|5|The Convict of Cayenne|0.4977865517|

五個原始問題只將 ID 與 question text 帶入 integration inputs，沒有 gold answer、supporting facts、evidence labels 或 correctness。各 retrieval event 記錄 config、原始分數、ID、rank、actual K、missingness/mask，以及已驗證的 cosine。

既有 `src/uncertainty.py` 的公式沒有更動：`top1_uncertainty=1-top1`、`topk_mean_uncertainty=1-mean(topk)`、`score_margin=top1-top2`。只在 cosine 驗證成功後呼叫既有計算；這些量不是 calibrated failure probability。新的 standalone retriever 沒有接入 Agent Graph。

## 資源實測與全量估算

實測環境：Python 3.11，Torch distribution metadata 2.5.1（runtime 為 CPU build）、Transformers 4.50.3、faiss-cpu 1.8.0、NumPy 1.26.4。平台 API 回報 NT kernel `Windows-10-10.0.26200-SP0`。CPU Intel Family 6 Model 183，使用 4 threads。

主編碼程序 peak working set **2,264,973,312 bytes = 2.109 GiB**。此值不含 FAISS 子程序，也不是整台機器或全量任務的 peak RAM。GPU 驅動顯示 RTX 4070、12,282 MiB、driver 591.86；當時全系統 GPU 使用 2,560 MiB，這是背景使用量，不能當作 GTE VRAM。Torch `cuda.is_available()` 為 false，故 GTE GPU 時間與 peak allocated/reserved VRAM 均為 null。

另外只做完整 corpus **tokenization**，沒有全量 model forward：384,857 documents，40,956,714 tokens；平均 106.42，p50=72、p90=221、p95=317、p99=570、max=4,708。512 limit 截斷 **5,492 筆（1.427%）**；token audit 耗時 24.27 秒。這是 embedding 輸入截斷，原始 corpus 完全保留。

|全量項目|估算；未執行|
|---|---|
|FP32 vector payload|1,182,280,704 bytes，約 1.101 GiB|
|單一 embeddings.npy|1,182,280,832 bytes|
|單一 FlatIP index|1,182,280,749 bytes，按本次相同序列化格式|
|JSONL row mapping|41,453,446 bytes，約 39.53 MiB，按完整 ID 與實際 row 編號格式計算|
|額外一份 vector checkpoint|約 1.101 GiB|
|vectors + index + 一份 checkpoint + mapping|約 3.34 GiB；另計 manifest、臨時檔、既有 corpus/model 與可能的文件文字副本|
|CPU count-scaled reference|116,591.67 秒，約 32.39 小時|
|CPU token-linear scaling|129,796.34 秒 inference，約 36.05 小時；另加 tokenization/I/O|
|CPU token-quadratic scaling|150,027.40 秒 inference，約 41.67 小時；另加 tokenization/I/O|
|GPU encoding time / peak VRAM|未測量，不能由 CPU 速度推導|

Token scaling 使用實際完整 corpus 依固定 ID 順序、batch=8、512 limit 的 padded token work，與 smoke 相同規則比較。Linear 與 quadratic 是兩個示意成本模型，並非上下信賴界限；CPU 峰值負載、散熱及真實 GPU throughput 仍會影響結果。

全量 RAM 尚未測量。若未來採 streaming/mmap，將本次主程序 2.109 GiB 加上一份完整 vectors 與 CPU FlatIP，可得到約 **4.31 GiB 的粗略基準**，未含所有文字、Python 物件、載入副本與轉換暫存；建議先保留 **10–16 GiB RAM 預算**並在 Phase 3B 實測。這不表示目前 smoke 的 JSON subprocess 介面可以直接擴到 384,857 筆：全量必須另做串流/檔案傳輸及 checkpoint 設計，避免巨大的 JSON float lists。

GPU 時間需要經批准的 CUDA 環境小樣本量測，才能使用 `384857 / measured_GPU_docs_per_second` 並依 token 分布修正。未測前不能保證 batch=8 的 VRAM，也不宣稱 Qwen3 與 GTE 可在 12GB 同時常駐。後續可優先考慮 **Qwen3 使用 GPU、GTE query encoder 使用 CPU、FAISS 使用 CPU**；本次 CPU 五筆 query batch 約 0.246 秒，但未量測 Qwen3 並行負載。

## 離線測試與重現

新增測試採 fake tokenizer/model 或 synthetic vectors，不依賴下載的 GTE，不呼叫 LLM。涵蓋 CLS/shape/L2/finite、query/doc 一致性、CPU fallback、empty/long text、CUDA OOM 明確回報、索引 round trip、mapping、checksum/model revision/ntotal mismatch、duplicate/missing IDs、`-1`、獨立 cosine、固定抽樣、telemetry、aggregate missingness、中文長路徑。

為避免此環境的原生 OpenMP runtime 在同一 pytest 程序相互影響，原有回歸與新增 native encoder 測試以兩個獨立程序執行。Windows sandbox 的暫存目錄 ACL 問題使 pytest 在沙箱外執行，使用專屬 basetemp，不覆寫既有 outputs。

```powershell
.venv311\Scripts\python.exe -m pytest tests --ignore=tests/test_embedding_encoder.py --ignore=tests/test_local_faiss_index.py -q --basetemp=outputs/pytest_phase3a_regression_20261010 -p no:cacheprovider
# 198 passed
.venv311\Scripts\python.exe -m pytest tests/test_embedding_encoder.py tests/test_local_faiss_index.py -q --basetemp=outputs/pytest_phase3a_verified_20261010 -p no:cacheprovider
# 27 passed
```

再次執行需指定新的 basetemp；integration 也會拒絕覆寫既有 report/index。真實 integration 使用：

```powershell
.venv311\Scripts\python.exe scripts/phase3a_gte_integration.py --run-real-integration --corpus-dir artifacts/2wiki/2543427d7c42ac82a88b6e8b4b2ff0063495a17dfefa72013e3f6cdacaf38b72 --model-dir models/gte-multilingual-base/9bbca17d9273fd0d03d5725c7a4b0f6b45142062 --model-lock docs/gte_model_lock.json --dataset-archive data/2wiki/official_20210407/data_ids_april7.zip --limit 128 --seed 43 --device cpu --audit-full-token-lengths --report outputs/phase3a/real_gte_integration.json
```

## 新增與修改檔案

本階段新增：

- `src/embedding_encoder.py`：shared local GTE document/query encoder。
- `src/embedding_contract.py`：不載入 Torch/FAISS 的設定與向量驗證契約。
- `src/corpus_selection.py`：corpus 驗證與固定 smoke selection。
- `corpus/build_local_index.py`：CPU FlatIP、manifest、mapping、reload、cosine verification。
- `src/faiss_process.py`、`scripts/faiss_index_worker.py`：原生 runtime 程序隔離與 UTF-8 IPC。
- `src/local_retriever.py`：獨立檢索工具與既有 uncertainty 相容事件。
- `scripts/inspect_gte_model.py`、`scripts/download_gte_assets.py`：metadata/code inspection 與 hash-locked 必要資產下載。
- `scripts/phase3a_gte_integration.py`：有明確開關的 100–1,000 文件 real integration 與 token audit。
- `tests/test_embedding_encoder.py`、`tests/test_local_faiss_index.py`。
- `docs/gte_model_lock.json`、`docs/2wiki_phase3a_validation.json`、本報告。

本階段修改：`.gitignore` 新增 `models/gte-multilingual-base/`；既有 artifacts/data/outputs 排除規則保持有效。沒有變更既有 Agent prompts、Pydantic schemas、graph topology、routing、HotpotQA/DPR encoder、success normalization 或 uncertainty 公式。Phase 1.5 與 Phase 1/2 原有未提交修改保留；沒有 commit/push/reset/clean 或刪除資料。

## 尚未解決與 Phase 3B 建議

1. **GPU 環境**：現有 Torch 為 CPU build。需另行批准 CUDA 相容環境與固定依賴版本的小樣本測試，不能承諾 GPU 時間/VRAM。沒有自行安裝或替換 Torch。
2. **全量能力**：本次 builder 故意限制 smoke≤1,000。Phase 3B 需獨立 `full` scope、串流 embeddings/checkpoints、resume、批次 OOM 明確處理、最終 count/hash 與 mapping 驗證；不能直接解除上限後使用巨量 JSON 傳輸。
3. **Token policy**：1.427% 文件超過 512 tokens。維持本次固定設定不代表答案一定在截斷區內或外；如需改 limit/chunking，須另批准新的 encoder/corpus policy 並產生不同 hash，不能默默修正原 corpus。
4. **Research validity**：原始 GTE 的確切等價性仍無法證明；pooled train+dev corpus 與 DPR-W100 是不同 retrieval 設定。小樣本沒有產生 retrieval recall、答案 EM/F1 或 failure-prediction 成果。
5. **資料品質**：先前 2,077 筆異常 supporting facts、2,060 個 evidence-invalid 問題保持原樣；未修復索引或改動 evidence 評估排除政策。
6. **下一步批准範圍**：先選擇 CPU full 或經隔離環境驗證後的 GPU full，確認 512-token policy、執行時間、checkpoint/RAM/disk 預算及固定 full output location，再批准 Phase 3B。Phase 4 與 Agent Graph 不在此範圍。

Phase 3A 到此停止，等待 Phase 3B 批准。

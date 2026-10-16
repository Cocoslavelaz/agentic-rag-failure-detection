# Phase 3B-0：CUDA Readiness、Checkpoint/Resume 與 GPU Benchmark

日期：2026-10-10。**Phase 3B-0 已完成**：CUDA GTE、CPU/GPU 數值一致性、三種 batch benchmark 與 checkpoint/resume 均通過。**未執行全量 embedding**。離線測試 **244 passed**。

建議批准獨立的 Phase 3B Full Embedding，使用 batch=8、512/512、float32、固定順序與 shard checkpoints；純編碼推估約 37 分鐘，執行預算建議 45–60 分鐘。這是工程建議，不是已執行或對完成時間的保證。

## 環境與 CUDA 修正

使用專案既有 `.venv311\Scripts\python.exe`，Python 3.11.16、Windows x64。安裝前 Torch runtime 為 `2.5.1+cpu`、CUDA 不可用；Transformers 4.50.3、NumPy 1.26.4、faiss-cpu 1.8.0、accelerate 1.6.0。NVIDIA 驅動 591.86；RTX 4070，總 VRAM 12,282 MiB。安裝前 `nvidia-smi` 使用約 2,373 MiB，約 9,909 MiB 可用；這是系統快照，並非 GTE 的 VRAM。

`nvidia-smi` 的 CUDA 13.1 表示驅動支援能力，不能當作 Torch 使用的 runtime。選擇官方 Windows CPython 3.11 `torch 2.5.1+cu124`，維持既有 Torch 主版本。官方來源：[PyTorch previous versions](https://pytorch.org/get-started/previous-versions/)、[CUDA 12.4 wheel index](https://download.pytorch.org/whl/cu124/torch/)、[NVIDIA CUDA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/index.html)。

安裝前保存 150 個套件版本與 requires、interpreter、Torch/CUDA、GPU 及磁碟空間至 `outputs/phase3b0/environment_before.json`；`uv pip check` 顯示所有套件相容。可用磁碟約 424 GiB。沒有 pip 套件，因此使用既有 uv 的 `pip` 子命令，沒有執行 `uv add` 或重建環境。

先 dry-run 確認只更換 Torch，再使用：

```powershell
uv --cache-dir .uv-cache pip install --python .venv311\Scripts\python.exe --no-deps --reinstall-package torch --index-url https://download.pytorch.org/whl/cu124 torch==2.5.1+cu124
```

wheel 約 2.3 GiB，cache 留在專案 `.uv-cache`。只修改本專案 `.venv311`；沒有修改其他 Conda/uv 環境、NVIDIA driver 或安裝完整 CUDA toolkit。`requirements.txt` 的 `torch==2.5.1` 接受此 local-version wheel，CUDA 安裝則須使用上方明確指令。

安裝後 Torch 為 **2.5.1+cu124**、CUDA runtime **12.4**、`cuda.is_available()=True`；GPU capability **8.9**。CUDA `arange(16)` 平方和等於 1240，64×64 CUDA 矩陣乘法結果通過檢查，GTE 隨後實際成功在 CUDA 執行。

前後 package snapshot 比對：只有 `torch: 2.5.1 → 2.5.1+cu124`，其餘 149 套件未改變；安裝後 `uv pip check` 再次顯示 150 套件全部相容。更新記錄保存於 `outputs/phase3b0/environment_after.json`。未使用 force/ignore-conflicts；未安裝 torchvision/torchaudio。

CUDA probe 的 `mem_get_info()` 回報 free **11,628,707,840 bytes**、total **12,878,086,144 bytes**。Windows WDDM 下 CUDA memory API、`nvidia-smi` 與 allocator allocated/reserved 的統計口徑不同，這些數值各自保留，不將背景 GPU 用量當作 GTE peak。

## 固定模型與截斷政策

Model ID：`Alibaba-NLP/gte-multilingual-base`。
Model/tokenizer revision：`9bbca17d9273fd0d03d5725c7a4b0f6b45142062`。
Reviewed code revision：`40ced75c3017eb27626c9d4ea981bde21a2662f4`。
全部本地模型資產沿用 `docs/gte_model_lock.json` 並在載入前核對 SHA256。

保持 CLS `last_hidden_state[:,0,:]`、768 維、float32 model/embeddings、L2 normalization、padded SDPA、無 query/document prefixes。Document/query token limit 固定 **512/512**，包含 special tokens。完整原始 passage text 保留於 frozen corpus，只在 tokenizer/embedding input 截斷，不刪除長文件。

Corpus checksum：`2543427d7c42ac82a88b6e8b4b2ff0063495a17dfefa72013e3f6cdacaf38b72`。重新核對 documents SHA256 與 manifest；未修改 corpus。沿用 Phase 3A 對全部 384,857 篇的 tokenizer-only 統計：5,492 篇超過 512 tokens，**1.427%**；平均 106.42，p50=72、p90=221、p95=317、p99=570、max=4,708。沒有為本次 benchmark 執行全量 model forward。

## GPU Benchmark 與 CPU/GPU 一致性

固定 seed `20261010`，按 SHA256(`seed:document_id`) 最小值選取 1,000 篇，再依原 corpus document ID 排列。選取所有 ID 及 selection hash 保存於 benchmark 目錄。必須同時覆蓋 `≤64`、`65–128`、`129–256`、`257–512`、`>512` 五種 token 長度。

先按 token 長度分位位置選出 128 篇（保留文件 ID 順序），進行 CPU/GPU float32 embedding 對照，再以同一批 1,000 篇比較 batch 8、16、32。預先固定最大元素差 `≤1e-4`、每篇 vector cosine `≥0.99999`；關閉 TF32，未使用 AMP/FP16/BF16。各 benchmark batch 的 128 篇子集合也與 CPU 再次比對。

每次 GPU benchmark 前 synchronize、reset peak memory，結束後 synchronize，記錄 elapsed、throughput、peak allocated/reserved VRAM、process RAM、token lengths/truncation。RAM 是程序累積 peak working set，不是可重設的單一 batch peak，且包含前置 CPU 模型驗證；CUDA allocator VRAM 不含其他 apps 或所有 driver/context overhead。

OOM 會記錄失敗、清除 cache，依 32→16→8→4→2→1 回退，重新編碼完全相同的輸入，不接受部分 rows；所有 batch 失敗就明確停止。OOM 回退是 benchmark policy；checkpoint 的固定 encoder config 不會默默改 batch/hash。

實測成功，沒有真實 OOM。樣本長度分布：≤64=428、65–128=320、129–256=170、257–512=70、>512=12。平均 **107.734** tokens，p50=73、p90=221.4、p95=321.1、p99=521.4、max=872；**12/1,000（1.2%）** 被截斷，所有文件均完成編碼。樣本 mean 與全 corpus 的 106.42 接近，但不是完整長度分布的保證。

128 篇 CPU reference 耗時 **16.186 秒**。首次 GPU/CPU 比對：最大元素差 **7.050111889839172e-7**、平均差 **5.5755e-8**、最低逐篇 cosine **0.9999999999877145**、最大向量 L2 差 **4.9569e-6**，通過事先固定門檻。三個 benchmark batch 的同一 128 篇子集也全部通過，最大元素差介於 **4.16e-7–4.91e-7**。

|Batch|GPU elapsed（含 tokenization/CPU copy）|Throughput docs/sec|Peak allocated VRAM GiB|Peak reserved VRAM GiB|程序累積 peak RAM GiB|全量 count-scaled 分鐘|
|---:|---:|---:|---:|---:|---:|---:|
|8|5.7704 s|173.30|1.426|1.621|2.867|37.01|
|16|7.3536 s|135.99|1.674|2.090|2.867|47.17|
|32|8.8127 s|113.47|2.170|2.496|2.867|56.53|

batch 8 是這次固定 corpus ID 順序下最快、VRAM 最低的設定。三組實際有效 token 總量相同，都是 105,714；padding 後分別為 271,360、334,800、405,552 tokens，而 padded token throughput 約 45,500–47,000 tokens/sec。較大的 batch 更慢與 padding 增加的觀察一致；這是單輪測試的解釋，不宣稱已排除 clocks/溫度等其他影響。沒有為加速而重排原 corpus、按長度 bucketing 或改 dtype。

以 batch 8 推估全量：`384857 / 173.29749 = 2220.79 秒 ≈ 37.01 分鐘`。再用 Phase 3A 完整 corpus 的 batch-8 padded token work 修正，linear 約 **37.47 分鐘**；quadratic inference-work 模型約 **36.33 分鐘**，需另計 tokenizer/I/O。這兩種 scaling 是成本模型，不是統計信賴區間。

建議配置 **45–60 分鐘執行預算**，包含 model startup、checkpoint/fsync、最終驗證與背景負載餘裕；並非保證上限。每個 batch 僅量測一輪，未做長時間熱穩態、全量 checkpoint I/O 或 Qwen3 並行測試。全量須保留固定 ID 順序與相同 batch/policy，否則 padding 成本會改變。Phase 3A 的舊 CPU-only wheel 耗時估算不當作本次 CUDA 計畫依據。

完整 raw report：`outputs/phase3b0/gpu_benchmark_1000/report.json`。
SHA256：`caf49dca01bffadf590ba85217aa0aecdc32da493a33f26a76ed8b4fe8daf94b`。
可追蹤摘要：[2wiki_phase3b0_validation.json](2wiki_phase3b0_validation.json)，含 19 個環境/benchmark/checkpoint 檔案的 hashes。
Selection hash：`d2e8a292e8a642988941686f52fab9bd06ba3ec6fb49e77be150f275d04d4b4a`。
batch-8 encoder config hash：`713b50b09bb405d06bd89a351b79bd9a28600aec7bd78d43ec270cf884f0dccf`。

本次只有 **1,000 個 unique passages**；CPU128 + GPU128 + 三組 GPU1000 + checkpoint128 合計 3,384 個成功編碼的 document rows，重複測量有明確用途。沒有以 384,857 篇執行模型。

## Checkpoint / Resume 實作

`src/embedding_checkpoint.py` 接受固定 corpus 順序的 iterator，保留 shard-size 的向量 buffer，將每個 shard 寫成：

```text
run.json
shard-000000/
  vectors.npy
  rows.jsonl
  metadata.json
shard-000001/...
interrupted/   # 僅明確復原時保留未提交的舊 shard
```

每個 payload 使用同目錄 UUID `.tmp`、flush/fsync、`os.replace`。完成 payload 後才 atomic commit `metadata.json`，最後 atomic 更新 `run.json`。Metadata 包含 corpus checksum/manifest hash、model ID/revision、完整 encoder config hash、shard ID、global row start/count、document ID 起迄/ordered-ID hash、shape、dimension、float32 dtype、payload checksums、completion status 及 encoding statistics。

Resume 先核對 root contract（包含 tokenizer/code revision、512/512、dtype、device、batch、seed、selection 與 Torch/Transformers/NumPy 版本），再驗證所有已提交 shard 的 checksum、vector norms、shape、連續 row IDs、唯一且排序的 document IDs；完成且有效的 shard 不重編碼。

宣告完成但缺失/損壞的 shard 一律報錯。沒有 metadata 的未提交 shard 預設也報錯；只有明確 `recover_incomplete=True`，才將整個舊目錄 atomic rename 到 `interrupted/`，保留原檔、記錄原因後重編碼。若中斷發生在 shard metadata 完成、root state 更新前，Resume 可經完整驗證後承認該 shard，避免重編碼。

最終重新讀取全部 shard mapping/vectors，比對每個 Embedding Row ID 所對應的 document ID 與輸入 corpus 順序完全一致，再提交 complete。缺失輸入不會提交不完整的最後 shard。程式本階段仍強制 **≤5,000 documents**、`scope=benchmark`，沒有全量執行入口。

離線 checkpoint 測試 **16 項通過**，涵蓋：已提交 shard 後中斷、metadata commit 與 root state update 之間中斷、未完成 shard 預設停止／明確隔離復原、corpus/model/batch mismatch、checksum mismatch、缺失 shard、重新計算 checksum 後仍能發現重複 Row ID、resume input ID 對不齊、missing/duplicate/unsorted 文件、atomic write 失敗與 5,000/512 guard。

真實 CUDA checkpoint 使用 128 篇、shard_size=64。完成第一個 shard 後模擬中斷；Resume 只編碼第二個 shard（resumed=1、new=1），最終 128 rows／2 shards 全部驗證通過；再 Resume 為 resumed=2、new=0。所有原已完成檔案保持有效，不重新編碼。此為控制式中斷，沒有對實體電源中斷做驗證。

## 離線測試與修改檔案

安裝後完整離線集合以兩個程序執行，沿用 Phase 3A 的 OpenMP 隔離策略：

```powershell
.venv311\Scripts\python.exe -m pytest tests --ignore=tests/test_embedding_encoder.py --ignore=tests/test_local_faiss_index.py --ignore=tests/test_embedding_checkpoint.py --ignore=tests/test_embedding_benchmark.py -q --basetemp=outputs/pytest_phase3b0_regression_20261010 -p no:cacheprovider
# 198 passed in 10.72s
.venv311\Scripts\python.exe -m pytest tests/test_embedding_encoder.py tests/test_local_faiss_index.py tests/test_embedding_checkpoint.py tests/test_embedding_benchmark.py -q --basetemp=outputs/pytest_phase3b0_native_20261010 -p no:cacheprovider
# 46 passed in 8.51s
```

合計 **244 passed、0 failed**，含原 225 項與新增 checkpoint 16 項、benchmark/OOM policy 3 項。既有 HotpotQA、FAISS scores/ranking、uncertainty 與資料品質政策的回歸測試保持通過。測試使用 fake models/synthetic vectors，不下載或執行 GTE/Qwen3/OpenAI；真實 GTE benchmark 是另外執行的 integration。

GPU benchmark 重現命令如下。會拒絕覆寫現有 output directory；重跑須指定新目錄，不能將此命令擴為全量。

```powershell
.venv311\Scripts\python.exe scripts/phase3b0_gpu_benchmark.py --run-gpu-benchmark --corpus-dir artifacts/2wiki/2543427d7c42ac82a88b6e8b4b2ff0063495a17dfefa72013e3f6cdacaf38b72 --model-dir models/gte-multilingual-base/9bbca17d9273fd0d03d5725c7a4b0f6b45142062 --model-lock docs/gte_model_lock.json --output-dir outputs/phase3b0/gpu_benchmark_1000 --count 1000
```

新增：`src/embedding_checkpoint.py`、`src/embedding_benchmark.py`、`scripts/phase3b0_gpu_benchmark.py`、`tests/test_embedding_checkpoint.py`、`tests/test_embedding_benchmark.py`、`docs/2wiki_phase3b0_report.md`、`docs/2wiki_phase3b0_validation.json`。

修改：`src/embedding_encoder.py` 將 input transfer 納入 CUDA OOM 捕捉範圍，OOM 時釋放 batch/output 的 GPU references，方便 caller 重試原輸入。正常 encoding 的模型、dtype、pooling、normalization 與分數語意沒有變更。

沿用 Phase 3A 的 FAISS 程序隔離；未設定 `KMP_DUPLICATE_LIB_OK`。原有未提交修改全部保留。未修改 Agent prompts、schemas、graph 或 uncertainty formulas；未執行 Qwen3、OpenAI API、Agent Graph、commit、push、reset、clean 或刪除既有資料。

## 尚未解決與 Full Embedding 建議

**建議批准下一階段 Full Embedding，優先 batch=8、shard_size=256、CUDA float32、512/512、固定 document ID 順序。** GPU、數值一致性與中斷恢復已具備實測支持；沒有環境相依性衝突。

仍須在批准後才新增獨立 `full` 執行入口、解除該入口的數量限制，並將同一 checkpoint contract 用於完整 corpus。現有 benchmark 檔案與 Phase 3A smoke index 均不能當作全量產物。全量 embedding payload 約 1.10 GiB，另計 shard headers、mapping/metadata、暫存空間；本次不是完整 FAISS index build 或 Agent Graph 測試。

剩餘限制：全量/長時間 throughput、實際斷電恢復及 Qwen3 同時常駐 GPU 尚未實測；真實 OOM 未發生，回退政策經離線注入驗證。512-token truncation 依授權保留，仍有 5,492 篇輸入截斷；沒有評估截斷對答案證據的影響。先前 2Wiki supporting-fact 資料品質問題與評估排除政策均未改動。

Full 執行應維持單一 GTE job，先固定 corpus/model/config/runtime versions、output location 與 checkpoint 參數，再開跑；不在同一 checkpoint 混用不同 batch/device/runtime，也不自行加入 Qwen3 或下一階段 Agent 實驗。

本階段結束後停止，等待使用者批准 Phase 3B Full Embedding。

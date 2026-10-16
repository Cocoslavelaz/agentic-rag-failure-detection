# 2Wiki Supporting Facts 資料品質調查

日期：2026-10-10。範圍：Phase 1 後資料品質調查；沒有開始 Phase 2。

## 結論與成因分類

直接以 `zipfile + json.load` 讀官方修正版 ZIP，獨立於 adapter 逐筆判斷 title 與 index，重現 **2,060 題、2,077 筆 invalid supporting facts**。受影響題目的原始 JSON、supporting_facts、context 與 gold answer，再與 `normalize_example(..., evidence_policy='retain')` 比對，**2,060 題全部一致**；原 archive 前後 SHA256 相同，仍為唯讀。不是 parsing／normalization 引入的問題。

|分類|train facts|dev facts|判定|
|---|---:|---:|---|
|Missing context title|0|0|受影響 facts 的 title 均存在且唯一|
|Sentence index out of range|2,007|70|原始 JSON 已越界，為確定的直接原因|
|Empty sentence list|0|0|對應 context 句子列表非空|
|Sentence segmentation mismatch|未確定|未確定|可能的歷史成因，不能由越界直接證明|
|Dataset parsing／normalization error|0|0|原始與 normalized annotations/context 全部一致|
|Other observed structural causes|0|0|沒有另外觀察到 malformed fact 或 ambiguous title|

所有越界索引均超過上界，沒有負數。當 `index == len(sentences)`，已超過最後有效 index 一格：train 1,672、dev 55，共 **1,727** 筆。其餘 350 筆超出更多；`index-len(sentences)` 分布為 `{0:1727, 1:256, 2:27, 3:17, 4:46, 6:4}`。這個模式不能證明作者採用 one-based indexing，也不能據此將 index 減一。

## Split 與 question type

表格每格為「受影響題數／invalid facts 數」。

|Type|train|dev|合計|
|---|---:|---:|---:|
|compositional|1,776／1,791|41／41|1,817／1,832|
|inference|110／111|27／27|137／138|
|comparison|26／26|1／1|27／27|
|bridge_comparison|78／79|1／1|79／80|
|全部|1,990／2,007|70／70|2,060／2,077|

2,060 題均保留非空 gold answer 與 10 個 context entries。字面、忽略大小寫的 gold substring 在 train 1,965／1,990 題、dev 59／70 題 context 中出現；其餘 36 題不能據此判為 unanswerable（例如比較題答案、別名或跨句推理），不因此剔除 QA 評估。`qa_valid` 表示有可用題目／標籤及結構，不代表答案已被證明正確或 context 語意充分。

## 小樣本與官方版本對照

重新核對 [官方 README 的 April 7, 2021 更新](https://github.com/Alab-NII/2wikimultihop#new-update-april-7-2021)：其直接連結 `data_ids_april7.zip`；Dataset Contents 指定 supporting-fact index 從 0 起。本地 archive SHA256：`95df2bf56fdabe034e27aebc580e02264232203cf52552f9efe8a919e5529eef`，與 Phase 1 下載紀錄相符。因此下列案例是在官方宣稱修正句子切分的版本中仍存在的範圍不一致。

|Split／question ID|Raw annotation|實際句數／有效 index|Gold answer|
|---|---|---|---|
|train `19e8111a0bda11eba7f7acde48001122`|`["Leslie Goodwins",5]`|5／0–4|Hollywood|
|train `1a5936300bdd11eba7f7acde48001122`|`["Heroes of Fort Worth",1]`|1／0|Rome|
|dev `695689d20baf11ebab90acde48001122`|`["Aditya Chopra",19]`|15／0–14|Yash Chopra|
|dev `26af280a0bdc11eba7f7acde48001122`|`["Georges Lautner",3]`|3／0–2|Renée Saint-Cyr|

JSON report 另含每個 split/type 最多兩題，實際 14 題代表樣本，保存 question/gold、原 facts、句數、末句與 adapter equality。完整 2,077 筆清單附 split、row index、question ID、原 annotation 及句數，可回溯原 ZIP。

另外掃描同一 archive 三個 splits 中涉及的 titles，以「相同 title + 串接文字空白正規化 SHA256」比較句數，未找到這些 invalid facts 有相同文本但不同句數的替代 context（0 筆）。這只是同版本診斷，不驗證早期版本，也不能排除歷史切句／段落截斷／標註生成等成因。沒有下載舊版、超連結 corpus 或重新切句。**歷史 root cause 尚未確認**，需要作者說明或額外版本比對，不能報成已證實 segmentation mismatch。

## 獨立 flags 與評估政策

`validation_flags(row, split=...)` 輸出：

- `qa_valid`：題目、ID、type、context 與 labeled gold answer 結構可用；annotation-only 問題不使 QA 失效。official test 始終 false。
- `evidence_valid`：context、ID 與 annotation 結構／句子對應可用；有任一 evidence annotation 問題，整題排除 annotation-dependent evidence metrics。official test 始終 false。
- `qa_exclusion_reasons`、`evidence_exclusion_reasons`、完整 `validation_issues`；均不改 raw annotations。

`evaluation_eligibility(example)` 將 qa_valid 對應 EM/F1 與 failure-prediction eligibility，將 evidence_valid 對應 annotation-dependent evidence metrics eligibility；只提供政策，不執行 scorer 或訓練。缺 QA 標籤與缺 evidence 是獨立理由。evidence-invalid 題目不能只保留有效 facts 而偷偷縮小分母；應整題排除並記錄 reason。Flags、gold、排除理由均不能當 failure predictor 的 uncertainty features。

|Split|QA eligible|QA excluded|Evidence eligible|Evidence excluded／原因|
|---|---:|---:|---:|---|
|train|167,454|0|165,464|1,990／invalid_sentence_index|
|dev|12,576|0|12,506|70／invalid_sentence_index|
|test|0|12,576|0|12,576／unlabeled_official_test|

排除數以題計算；同題兩個 invalid facts 不重複計兩次。JSON 同時保留 fact-level 與 question-level 計數。相同 QA split 在 QA/evidence 指標使用不同 eligibility masks，後續報表須明列各自分母。

為保留原 Phase 1 strict 行為，loader／normalizer 預設 `evidence_policy='strict'`，仍對 annotation errors 報錯；新增明確 `evidence_policy='retain'` 讓 QA-valid、evidence-invalid 題目保留原樣並附 flags。不接受 `repair` 等隱含策略；QA 結構破損仍報錯，duplicate IDs 檢查不取消。registry 與 `src.utils.load_dataset` 均可傳此選項。官方未標註 test 仍可載入，但兩種 eligibility 都 false。

```python
from src.datasets import load_dataset, agent_input
from src.datasets.two_wiki import evaluation_eligibility

rows = load_dataset('2wikimultihopqa', archive_path, split='dev', evidence_policy='retain')
policy = evaluation_eligibility(rows[0])
inputs = agent_input(rows[0], dataset_format='2wikimultihopqa')
# inputs 只有 original_question，不包含 flags、gold、context 或 evidence。
```

## 重現、測試與變更

從 repo root 執行；輸出已存在時拒絕覆寫，重跑請使用新路徑：

```powershell
.\.venv311\Scripts\python.exe scripts/investigate_two_wiki_quality.py --archive data/2wiki/official_20210407/data_ids_april7.zip --output outputs/2wiki_quality_new.json
.\.venv311\Scripts\python.exe scripts/validate_two_wiki.py --archive data/2wiki/official_20210407/data_ids_april7.zip --provenance docs/2wiki_source_provenance.json --output outputs/2wiki_validation_quality_new.json
```

第二個命令仍以 exit code 1 表示 strict evidence validation 未全通過；不代表 QA eligibility 全部失效。保留原 Phase 1 report，新增 [品質調查 JSON](2wiki_data_quality_investigation.json) 與 [含 flags/exclusions 的 validation report](2wiki_schema_validation_quality.json)。

新增 12 個離線 policy/classification tests，完整 suite **182 passed in 6.00s**。JUnit：`outputs/two_wiki_quality_tests_20261010.xml`。驗證 retention 不改 raw/gold/annotations、不洩漏 gold、QA-invalid 仍拒絕、test 不評估、排除按題計數、四種 raw causes 分類與 strict 回歸。靜態契約比對五項 preserved=true。

修改 `src/datasets/two_wiki.py`、`src/datasets/__init__.py`、`src/utils.py` loader wrapper、`scripts/validate_two_wiki.py`、`tests/test_two_wiki.py`；新增 investigation script、兩個 JSON reports 與本文件。既有未提交修改與原 report 保留。Agent prompts、Pydantic schemas、graph 與 uncertainty 計算均未改。

沒有執行模型、完整 graph、embeddings、FAISS 或 Phase 2 工作，沒有 commit/push。調查完成；停止並等待 Phase 2 批准。歷史 segmentation 成因仍未確認，所有 annotation 原樣保留。

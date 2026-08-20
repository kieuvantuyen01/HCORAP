# Kế hoạch hoàn thiện và nộp bài ICIIT 2027

Cập nhật ngày 20/08/2026. Ma trận publication hiện hành là compact campaign
1.270 measured runs trong
`experiments/configs/reduced_campaign_manifest.json`. Bản phân tích đầy đủ về
cấu hình, kết quả cũ và lý do loại nhánh nằm tại
[`docs/COMPACT_EXPERIMENT_MATRIX_20260820.md`](docs/COMPACT_EXPERIMENT_MATRIX_20260820.md).

Không chạy measured phase cho tới khi code/config sau rà soát được commit,
worktree sạch, publication tag mới được tạo và `HCORAP_EXPECTED_COMMIT` trỏ
đúng commit/tag đó.

## 1. Thông điệp khoa học

Policy chính được khóa là LEX-COS:

\[
\min CONT\;\rightarrow\;\min OT\;\rightarrow\;\max SIM.
\]

LEX-OCS đổi hai ưu tiên đầu và chỉ là order-sensitivity analysis:

\[
\min OT\;\rightarrow\;\min CONT\;\rightarrow\;\max SIM.
\]

Thông điệp bài báo giữ ở mức:

> LEX-COS làm rõ ưu tiên vốn có thể mơ hồ dưới weighted optimum; Totalizer làm
> thay đổi cấu trúc encoding; tác động của implied constraints và exact symmetry
> breaking phải được đánh giá bằng paired factorial ablation.

Không claim rằng mọi treatment đều nhanh hơn. Hiệu ứng âm, trung tính hoặc phụ
thuộc lớp instance đều là kết quả hợp lệ.

## 2. Câu hỏi nghiên cứu và bằng chứng

### RQ1 — Objective policy

Weighted B0 và LEX-COS khác nhau thế nào về CONT, OT, SIM, proved count và
PAR-2? LEX-OCS có làm thay đổi vector objective trên subset paired hay không?

### RQ2 — Totalizer

Totalizer ảnh hưởng thế nào đến số biến/clauses, peak RSS, proved count, PAR-2
và paired time-to-proof so với sorting network?

### RQ3 — Constraint strengthening

Implied constraints và exact slot-service symmetry breaking có hiệu ứng nào,
và hiệu ứng đó tương tác ra sao với cardinality encoding?

### Validation

Policy effect có còn quan sát được trên corrected-v2 critical benchmark không?
EvalMaxSAT có khớp exact objective với Gurobi MIP và CPLEX MIP trên commercial
subset đã khai báo trước không?

RQ2--RQ3 chỉ phát biểu cho EvalMaxSAT binary đã khóa và protocol đã khóa.
Gurobi/CPLEX là
exact-objective validation backends, không phải căn cứ cho MaxSAT runtime claim.

## 3. Cấu hình và benchmark split đã khóa

```text
B = sorting-network / implied none / symmetry none
R = totalizer / implied both / symmetry slot-service
```

- original factorial: 16 lớp, seeds 1--5, 80 instances;
- original LEX-COS: 14 lớp, seeds 1--10, 140 instances, chỉ R;
- LEX-OCS sensitivity: cùng 14 lớp, seeds 1--5, 70 instances, chỉ R;
- corrected-v2: 16 critical strata, evaluation seeds 1001--1005, 80 instances;
- commercial subset: hai lớp `30_15_4`, `40_25_5`, seeds 1--10, 20 instances.

Hai commercial-development classes bị loại khỏi original LEX sets. Chúng vẫn
được dùng trong commercial validation vì vai trò này được khai báo riêng. B--R
weighted comparison tái sử dụng hai cell của factorial. RQ1 chạy lại weighted R
cùng LEX-COS; commercial validation cũng chạy cả hai MaxSAT policies. Mọi
measured top-level run dùng timeout 300 s.

## 4. Ma trận measured compact

| Campaign | Thiết kế | Runs | Timeout | Worst-case core-hour |
|---|---|---:|---:|---:|
| original factorial ablation | 80 × 8 configs × weighted | 640 | 300 s | 53,33 |
| original policy comparison | 140 × R × weighted/LEX-COS | 280 | 300 s | 23,33 |
| LEX-OCS sensitivity | 70 × R × LEX-OCS | 70 | 300 s | 5,83 |
| corrected-v2 validation | 80 × R × weighted/LEX-COS | 160 | 300 s | 13,33 |
| Gurobi/CPLEX validation | 20 × 2 backends × weighted/LEX-COS | 80 | 300 s | 6,67 |
| EvalMaxSAT commercial validation | 20 × R × weighted/LEX-COS | 40 | 300 s | 3,33 |
| **Tổng measured** |  | **1.270** |  | **105,83** |

Ngoài measured matrix có 4 EvalMaxSAT LEX-COS scalability-calibration runs ở
timeout 300 s và 18 commercial correctness-smoke runs ở timeout 30 s. Chúng
không được dùng trong runtime tables. Calibration phải đạt ít nhất 2/4 optimum
trước khi measured campaign bắt đầu. Worst case measured tuần tự là 4,41
ngày. So với kế hoạch 4.996 runs/335,27 core-hour trước đó, thiết kế mới giảm
74,58% số run và 68,43% worst-case compute. Diagnostic lịch sử cho thấy cả 12
direct-factor contrasts giữ cùng hướng khi giảm từ 10 xuống 5 seeds/lớp; các
effect size lịch sử không được dùng làm publication evidence.

## 5. Phần hoãn hoặc loại khỏi bài

| Nhánh | Quyết định | Cách trình bày |
|---|---|---|
| Pareto/epsilon confirmation | hoãn | không đưa exploratory pilot vào main results |
| weight sensitivity confirmation | hoãn | không đưa claim sensitivity |
| corrected relaxed/saturated stress | hoãn | limitation/future study |
| availability uncertainty | hoãn | không claim robust optimization |
| routing | loại | ngoài implemented decision model |
| CPLEX CP và Open-WBO | loại | CP không cần cho exact MIP validation; giữ một MaxSAT solver chính nhất quán với nghiên cứu gốc |
| extra IC/SB modes, `both-plus` | loại | pilot-only; không cần cho factorial claim |

Không mở lại các nhánh này trước submission trừ khi một claim cốt lõi bị loại
và nhóm tác giả chủ động định nghĩa lại scope trước khi xem primary results.

## 6. Protocol GCP

Publication machine: non-Spot GCP `c4-highcpu-8`, 8 vCPU, 16 GB RAM, Ubuntu
24.04 LTS. Mỗi measured run dùng một solver process bị giới hạn vào một pinned
vCPU; publication default là `WORKERS=1`. Gurobi và CPLEX còn được khóa một
native solver thread.

Khóa các yếu tố sau:

- EvalMaxSAT Linux x86-64 SHA-256 `97614c996e1173ca0672ec46da153656046db1d84b9362a8561161ee750779f7`;
- compile flags `-O3 -DNDEBUG -std=c++11`;
- cùng VM image, source/binary/solver hash;
- timeout 300 s cho mọi measured top-level run; 30 s chỉ cho smoke;
- cumulative timeout cho toàn bộ stages của lexicographic policy;
- 10 warm-up instances không thuộc measured sample;
- instance-major blocked randomized order với seed lưu trong config;
- Gurobi/CPLEX: one thread, seed 0, relative/absolute gap 0;
- raw JSON, native/stderr logs, peak RSS, verifier outcome và all hashes.

## 7. Cách chạy

Sau khi đã đặt đúng EvalMaxSAT binary, Gurobi và CPLEX trên VM:

```bash
export EVALMAXSAT_BIN=/opt/evalmaxsat/EvalMaxSAT_bin
export GUROBI_HOME=/absolute/path/to/gurobi/platform
export CPLEX_STUDIO_DIR=/absolute/path/to/CPLEX_Studio
export HCORAP_CPU_CORE=0
export HCORAP_EXPECTED_COMMIT=iciit2027-exp-v2
export HCORAP_BACKUP_DIR=/mnt/hcorap-backup
export CONFIRM_PUBLICATION_CAMPAIGN=YES

bash experiments/run_all_remaining_publication.sh
```

Thứ tự pipeline:

```text
build/test/benchmark checks
-> commercial license + 18-run correctness smoke
-> 640-run factorial hard gate
-> 140 weighted + 140 LEX-COS + 70 LEX-OCS
-> 160 corrected-v2
-> 80 commercial MIP + 40 MaxSAT commercial weighted/LEX
-> analysis -> package -> manuscript freeze
```

Chi tiết resume và checkpoint nằm trong
[`docs/GCP_EXPERIMENT_RUNBOOK.md`](docs/GCP_EXPERIMENT_RUNBOOK.md).

## 8. Gates bắt buộc

### G1 — Preflight

- all tests và C++ builds pass;
- solver source/binary đúng pinned commit;
- corrected-v2 instances qua witness/hash/matrix verification;
- manifest đúng 1.270 measured + 18 non-measured runs;
- mọi MaxSAT config resolve đúng instance/task count;
- commercial preflight và 18/18 smoke runs qua verifier;
- ba smoke backends agreement trên 6 instance-policy groups.

### G2 — Factorial hard gate

Dừng campaign nếu C1 thiếu row, có technical/validation error, có unverified
solver-reported optimum, paired weighted-objective mismatch hoặc peak RSS vượt
12 GB. `reference_composite` chỉ là evidence label; gate không giả định R phải
nhanh hơn B. Không còn branch gate hậu nghiệm cho epsilon, weight hoặc LEX.

### G3 — Data freeze

- chính xác 1.270 measured rows, không duplicate/unexpected run ID;
- mọi `OPTIMUM` qua independent verifier;
- all analyzers trả `valid=true` với expected pair/group counts;
- MaxSAT/Gurobi/CPLEX objective agreement được tính chỉ trên groups cả ba backend
  chứng minh optimum;
- generated LaTeX/prose/provenance được tạo từ frozen artifacts;
- archive và SHA-256 được tạo từ clean publication commit;
- manuscript build chỉ mở sau freeze marker hợp lệ.

## 9. Ánh xạ kết quả vào bản thảo

| Claim/bảng | Nguồn duy nhất được phép dùng |
|---|---|
| factorial RQ2/RQ3 và B--R 80 pairs | `gcp_primary_analysis/factorial_*`, `weighted_composite_*` |
| weighted--LEX-COS R, 140 pairs | `gcp_primary_analysis/lex_confirmatory_*` |
| LEX-COS--LEX-OCS R, 70 pairs | `lex_policy_sensitivity_{pairs,summary}.csv` |
| corrected-v2, 80 instances/policy | `gcp_corrected_analysis/corrected_*` |
| three-backend agreement, 20 groups/policy | `gcp_cross_paradigm_analysis/*` |
| reproducibility | resolved configs, environment, manifests, hashes, validators |

Không dùng timing lịch sử trong `results/` hoặc `results_addition/` trong bảng
chính. Không gộp development/pilot rows với publication rows. Timeouts phải ở
trong solved counts và PAR-2; objective deltas chỉ dùng jointly-optimum,
verifier-passing pairs.

## 10. Trình bày trong bản 5 trang

Giữ đúng hai visual kết quả full-width:

1. bảng factorial: tám cells, bốn direct contrasts tiêu biểu và B--R trên cùng
   80 instances; đủ 12 direct contrasts nằm trong artifact;
2. bảng policy/validation: weighted--LEX-COS, LEX-OCS sensitivity,
   corrected-v2 và 20-instance three-backend agreement.

Không thêm Pareto, weight-sensitivity, uncertainty hoặc routing figure. Nếu còn
chỗ, ưu tiên threats to validity và exact sample/protocol description hơn một
exploratory plot.

## 11. Definition of done

- [ ] Publication commit/tag sạch và `HCORAP_EXPECTED_COMMIT` đã khóa.
- [ ] G1, G2, G3 pass; exact count là 1.270 measured rows.
- [ ] Tất cả bảng và prose định lượng sinh từ frozen raw data.
- [ ] Mỗi con số trong Abstract/Conclusion truy về generated evidence.
- [ ] Không dùng historical/development/pilot runtime làm publication evidence.
- [ ] Artifact có source, configs, instances, raw logs, environment và SHA-256.
- [ ] Clean-room reproduction chạy được trên fresh checkout.
- [ ] PDF đúng template/page limit, font embedded và không warning nghiêm trọng.
- [ ] Author metadata và submission metadata đã được đồng tác giả xác nhận.
- [ ] Kiểm tra lại deadline và yêu cầu ICIIT trên trang chính thức trước upload.

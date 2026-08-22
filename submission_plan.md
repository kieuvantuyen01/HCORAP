# Kế hoạch hoàn thiện và nộp bài ICIIT 2027

Cập nhật ngày 22/08/2026. Ma trận publication hiện hành là compact campaign
924 measured runs trong
`experiments/configs/reduced_campaign_manifest.json`. Bản phân tích đầy đủ về
cấu hình, kết quả cũ và lý do loại nhánh nằm tại
[`docs/COMPACT_EXPERIMENT_MATRIX_20260820.md`](docs/COMPACT_EXPERIMENT_MATRIX_20260820.md).
Trạng thái 732 rows đã có và đúng 192 rows còn thiếu được khóa tại
[`docs/EXPERIMENT_SUPPLEMENT_MATRIX_20260822.md`](docs/EXPERIMENT_SUPPLEMENT_MATRIX_20260822.md).

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

Weighted và LEX-COS khác nhau thế nào về CONT, OT, SIM, proved count và PAR-2?
Trên corrected benchmark, LEX-OCS có làm thay đổi vector objective hay không?

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

- original factorial: 16 lớp, seeds 1--3, 48 instances;
- original weighted/LEX-COS: 14 lớp, seeds 1--3, 42 instances, chỉ R;
- corrected-v2: 16 critical strata, evaluation seeds 1001--1003, 48 instances,
  chạy weighted/LEX-COS/LEX-OCS dưới R;
- commercial subset: hai lớp `30_15_4`, `40_25_5`, seeds 1--10, 20 instances.

Hai commercial-development classes bị loại khỏi original LEX sets. Chúng vẫn
được dùng trong commercial validation vì vai trò này được khai báo riêng. B--R
weighted comparison tái sử dụng hai cell của factorial. RQ1 chạy lại weighted R
cùng LEX-COS; commercial validation cũng chạy cả hai MaxSAT policies. Mọi
measured top-level run dùng timeout 300 s.

## 4. Ma trận measured compact

| Campaign | Thiết kế | Runs | Timeout | Worst-case core-hour |
|---|---|---:|---:|---:|
| original factorial ablation | 48 × 8 configs × weighted | 384 | 300 s | 32,00 |
| original policy comparison | 42 × R × weighted/LEX-COS | 84 | 300 s | 7,00 |
| corrected-v2 EvalMaxSAT scalability | 48 × R × weighted/LEX-COS/LEX-OCS | 144 | 300 s | 12,00 |
| Gurobi/CPLEX validation | 20 × 2 backends × weighted/LEX-COS | 80 | 300 s | 6,67 |
| EvalMaxSAT commercial validation | 20 × R × weighted/LEX-COS | 40 | 300 s | 3,33 |
| corrected-v2 exact policy primary | 48 × Gurobi × weighted/LEX-COS/LEX-OCS | 144 | 300 s | 12,00 |
| corrected-v2 CPLEX stratum audit | 16 × CPLEX × weighted/LEX-COS/LEX-OCS | 48 | 300 s | 4,00 |
| **Tổng measured** |  | **924** |  | **77,00** |

Ngoài measured matrix có 4 EvalMaxSAT LEX-COS scalability-calibration runs,
48 corrected-v2 exact-solver calibration runs ở timeout 300 s và 18 commercial
correctness-smoke runs ở timeout 30 s. Chúng
không được dùng trong runtime tables. Calibration phải đạt ít nhất 2/4 optimum
trước khi measured campaign bắt đầu. Worst case measured tuần tự là 3,21 ngày;
với dữ liệu hiện có chỉ còn 192 measured rows, tối đa 16 core-hours.
So với campaign 1.270 runs trước khi compact, thiết kế hiện tại giảm 27,24%
số run và worst-case compute. Mỗi factorial contrast vẫn có 48 paired
blocks trên đủ 16 lớp; kết quả theo từng lớp chỉ được trình bày mô tả vì mỗi lớp
có ba seeds.

Một kiểm tra chuyển giao được tách khỏi ma trận 924 runs: chạy LEX-COS trên
corrected-v2 với `Totalizer-only` và cấu hình R. Pilot dùng seed 1002 của 16
strata, tức 32 runs và tối đa 2,67 core-hour. Chỉ chạy confirmation 96 runs
(48 instances × 2 configs, tối đa 8 core-hour) nếu pilot đạt ít nhất một điều
kiện: thêm ròng 2 optimum, tiến thêm một criterion trên ít nhất 4/16 cặp, hoặc
giảm PAR-2 ít nhất 10%. Kết quả chỉ nhập vào bài sau khi đủ 96 rows; pilot không
được gộp vào estimate cuối.

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
export HCORAP_EXPECTED_COMMIT=iciit2027-exp-v3
export HCORAP_BACKUP_DIR=/mnt/hcorap-backup
export CONFIRM_PUBLICATION_CAMPAIGN=YES

bash experiments/run_all_remaining_publication.sh
```

Thứ tự pipeline:

```text
build/test/benchmark checks
-> commercial license + 18-run correctness smoke
-> 384-run factorial hard gate
-> 42 weighted + 42 LEX-COS trên original
-> 48 weighted + 48 LEX-COS + 48 LEX-OCS trên corrected-v2
-> 80 commercial MIP + 40 MaxSAT commercial weighted/LEX
-> 48-run corrected exact calibration
-> 144 Gurobi corrected primary + 48 CPLEX stratum audit
-> analysis -> package -> manuscript freeze
```

Chi tiết resume và checkpoint nằm trong
[`docs/GCP_EXPERIMENT_RUNBOOK.md`](docs/GCP_EXPERIMENT_RUNBOOK.md).

## 8. Gates bắt buộc

### G1 — Preflight

- all tests và C++ builds pass;
- solver source/binary đúng pinned commit;
- corrected-v2 instances qua witness/hash/matrix verification;
- manifest đúng 924 measured + 52 calibration + 18 smoke runs;
- mọi MaxSAT config resolve đúng instance/task count;
- commercial preflight và 18/18 smoke runs qua verifier;
- ba smoke backends agreement trên 6 instance-policy groups.

### G2 — Factorial hard gate

Dừng campaign nếu C1 thiếu row, có technical/validation error, có unverified
solver-reported optimum, paired weighted-objective mismatch hoặc peak RSS vượt
12 GB. `reference_composite` chỉ là evidence label; gate không giả định R phải
nhanh hơn B. Không còn branch gate hậu nghiệm cho epsilon, weight hoặc LEX.

### G3 — Data freeze

- chính xác 924 measured rows, không duplicate/unexpected run ID;
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
| factorial RQ2/RQ3 và B--R 48 pairs | `gcp_primary_analysis/factorial_*`, `weighted_composite_*` |
| original weighted--LEX-COS R, 42 pairs | `gcp_primary_analysis/lex_confirmatory_*` |
| corrected EvalMaxSAT scalability | `gcp_corrected_analysis/corrected_*` |
| corrected weighted--LEX-COS và LEX-COS--LEX-OCS exact pairs | `gcp_corrected_exact_analysis/corrected_pairwise_*` |
| corrected-v2 exact policy, 48 instances/policy | `gcp_corrected_exact_analysis/corrected_policy_summary.csv` |
| three-backend agreement, 20 groups/policy | `gcp_cross_paradigm_analysis/*` |
| reproducibility | resolved configs, environment, manifests, hashes, validators |

Không dùng timing lịch sử trong `results/` hoặc `results_addition/` trong bảng
chính. Không gộp development/pilot rows với publication rows. Timeouts phải ở
trong solved counts và PAR-2; objective deltas chỉ dùng jointly-optimum,
verifier-passing pairs.

## 10. Trình bày trong bản 5 trang

Giữ đúng ba visual kết quả, mỗi visual có một nhiệm vụ riêng:

1. figure full-width hai panel: weighted--LEX-COS policy deltas trên
   corrected-v2 và bốn sorting-network/Totalizer confidence intervals;
2. bảng single-column đủ tám factorial cells, báo PAR-2, peak RSS và số biến;
3. bảng single-column đối chiếu signal của original/corrected-v2 và cho biết
   EvalMaxSAT dừng ở criterion nào trong các lexicographic runs.

Đủ 12 direct contrasts ở mức hàng và toàn bộ summary columns nằm trong
artifact; bản chính chỉ giữ range/direction cần cho claims. Cấu trúc Results đi
theo claim (policy, encoding, validation) thay vì lặp lại nhãn RQ cơ học.

Không thêm Pareto, weight-sensitivity, uncertainty hoặc routing figure. Nếu còn
chỗ, ưu tiên threats to validity và exact sample/protocol description hơn một
exploratory plot.

## 11. Definition of done

- [ ] Publication commit/tag sạch và `HCORAP_EXPECTED_COMMIT` đã khóa.
- [ ] G1, G2, G3 pass; exact count là 924 measured rows.
- [ ] Tất cả bảng và prose định lượng sinh từ frozen raw data.
- [ ] Pilot Totalizer-only/LEX-COS đã có quyết định GO/STOP; nếu GO thì đủ 96
      confirmation rows trước khi đưa claim chuyển giao vào bài.
- [ ] Mỗi con số trong Abstract/Conclusion truy về generated evidence.
- [ ] Không dùng historical/development/pilot runtime làm publication evidence.
- [ ] Artifact có source, configs, instances, raw logs, environment và SHA-256.
- [ ] Clean-room reproduction chạy được trên fresh checkout.
- [ ] PDF đúng template/page limit, font embedded và không warning nghiêm trọng.
- [ ] Author metadata và submission metadata đã được đồng tác giả xác nhận.
- [ ] Kiểm tra lại deadline và yêu cầu ICIIT trên trang chính thức trước upload.

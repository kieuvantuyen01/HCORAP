# Kế hoạch hoàn thiện và nộp bài ICIIT 2027

Cập nhật ngày 04/09/2026. Kế hoạch này thay thế ma trận 924 runs và bốn RQ
trước đây. Bản thảo chính chỉ còn hai thiết kế thực nghiệm. Validation,
sensitivity và ablation là bằng chứng hỗ trợ, không được trình bày như các
nghiên cứu ngang hàng.

Runbook chi tiết cho campaign còn thiếu:
[`docs/COMPACT_RESULTS_RUNBOOK.md`](docs/COMPACT_RESULTS_RUNBOOK.md).

Không chạy measured phase cho tới khi code và configs được review, commit,
push, worktree sạch và `HCORAP_EXPECTED_COMMIT` trỏ đúng frozen commit.

## 1. Định vị nghiên cứu

Nghiên cứu không tuyên bố MaxSAT nhanh hơn các commercial solvers. Đóng góp
được định vị như sau:

1. xây dựng một chính sách HCORAP theo thứ tự ưu tiên tường minh:
   `continuity → overtime → compatibility`;
2. xây dựng Corrected-v2 để các mục tiêu continuity và overtime thực sự thay
   đổi trong nghiệm được chọn;
3. đánh giá một phương pháp MaxSAT chính xác với hai cardinality encodings,
   đồng thời kiểm tra encoding effect dưới cả weighted và LEX-COS;
4. kiểm chứng status, assignment và objective vector bằng independent verifier
   và exact MIP references.

Kết luận về chính sách tối ưu được tách khỏi kết luận về hiệu năng MaxSAT.
Gurobi tạo objective-quality evidence; EvalMaxSAT tạo encoding evidence.

## 2. Hai câu hỏi nghiên cứu

### RQ1: Ảnh hưởng của objective policy

So với weighted objective, LEX-COS thay đổi continuity, overtime và
compatibility như thế nào trên các instances có tải đủ cao để ba tiêu chí cùng
có ý nghĩa?

### RQ2: Ảnh hưởng của cardinality encoding

Khi mọi yếu tố khác được giữ cố định, Totalizer thay đổi completion, PAR-2,
paired runtime và formula size như thế nào so với sorting network dưới weighted
và LEX-COS?

Implied constraints, symmetry breaking, solver agreement và LEX-OT không có RQ
riêng. Chúng lần lượt là configuration-selection ablation, correctness
validation và priority-order sensitivity.

## 3. Thiết kế A: policy study đã hoàn tất

- Dataset: 48 Corrected-v2 critical instances.
- Cấu trúc: 16 size categories × 3 evaluation seeds.
- Main policies: weighted và LEX-COS.
- Primary solver: Gurobi MIP-E, 300 s, một thread.
- Main runs: 96/96 `OPTIMUM`.
- CPLEX audit: 16 categories × 2 policies = 32/32 `OPTIMUM`, objective values
  khớp Gurobi.
- LEX-OT: 48 Gurobi + 16 CPLEX runs đã có, chỉ dùng sensitivity.

Không chạy thêm Thiết kế A.

Nguồn được phép dùng:

- `results_v2/gcp_corrected_exact_analysis/corrected_pairwise_summary.csv`;
- `results_v2/gcp_corrected_exact_analysis/corrected_pairwise_pairs.csv`;
- `results_v2/gcp_corrected_exact_analysis/corrected_policy_summary.csv`;
- `results_v2/gcp_corrected_exact_analysis/corrected_exact_validation.json`.

## 4. Thiết kế B: policy-by-encoding cần chạy

### MaxSAT matrix

| Policy | Encoding | IC | SB | Instances | Timeout | Runs |
|---|---|---|---|---:|---:|---:|
| weighted | sorting network | none | none | 48 | 3.600 s | 48 |
| weighted | Totalizer | none | none | 48 | 3.600 s | 48 |
| LEX-COS | sorting network | none | none | 48 | 3.600 s | 48 |
| LEX-COS | Totalizer | none | none | 48 | 3.600 s | 48 |
| **Tổng MaxSAT** |  |  |  |  |  | **192** |

Dataset là toàn bộ Original suite với seeds 1--3. Không loại hai classes từng
bị loại trong campaign 42 instances. MaxSAT dùng một cumulative budget cho cả
ba LEX-COS stages. Bốn tasks của cùng instance tạo thành một randomized block.

### Exact reference

Gurobi MIP-E chạy hai policies trên cùng 48 instances:

| Backend | Policies | Instances | Timeout | Runs |
|---|---:|---:|---:|---:|
| Gurobi MIP-E | 2 | 48 | 3.600 s | 96 |

Gurobi reference không phải runtime baseline. Nó xác nhận mọi MaxSAT result đã
quyết định và objective vector của mọi MaxSAT `OPTIMUM`.

### Tổng khối lượng mới

- 192 EvalMaxSAT rows;
- 96 Gurobi reference rows;
- tổng 288 records;
- chỉ bốn MaxSAT configurations, không phải 288 thiết kế.

MaxSAT worst case là 192 core-hour. Dữ liệu 300 s hiện có cho thấy ước lượng
thực tế khoảng 9--12 giờ tuần tự, nhưng con số này chỉ dùng để vận hành VM.

## 5. Vai trò của dữ liệu cũ

| Nguồn | Vai trò mới |
|---|---|
| 8-cell weighted factorial | giải thích vì sao IC và SB bị tắt |
| weighted SN/TOT 300 s | evidence sơ bộ, sẽ bị thay bởi matrix 3.600 s |
| original 42-instance weighted/LEX-COS | diagnostic về objective activity |
| corrected-v2 EvalMaxSAT 300 s | historical scalability diagnostic, không dùng ở main results |
| three-solver 20-instance subset | correctness check bổ sung |
| Totalizer-transfer pilot | STOP, artifact only |
| epsilon, weight, uncertainty, routing | ngoài scope |

Không cộng dữ liệu 300 s và 3.600 s trong cùng PAR-2 hoặc runtime distribution.
Không trộn rows từ khác binary hash hoặc source commit trong một direct
comparison.

## 6. Cách chạy trên GCP

```bash
export EVALMAXSAT_BIN=/opt/evalmaxsat/EvalMaxSAT_bin
export GUROBI_HOME=/absolute/path/to/gurobi/platform
export HCORAP_CPU_CORE=0
export HCORAP_EXPECTED_COMMIT=$(git rev-parse HEAD)
export HCORAP_BACKUP_DIR=/mnt/hcorap-backup
```

Preflight:

```bash
./experiments/run_compact_policy_encoding.sh preflight
```

Measured campaign:

```bash
export CONFIRM_COMPACT_POLICY_ENCODING=YES
./experiments/run_compact_policy_encoding.sh all
```

Runner hỗ trợ resume theo phase:

```bash
./experiments/run_compact_policy_encoding.sh reference
./experiments/run_compact_policy_encoding.sh maxsat
./experiments/run_compact_policy_encoding.sh analyze
```

Sau khi đồng bộ kết quả về máy chứa analysis Corrected-v2, sinh bảng và macro
LaTeX bằng evidence gate:

```bash
export HCORAP_POLICY_ANALYSIS=results_v2/gcp_corrected_exact_analysis
export HCORAP_MANUSCRIPT_RESULTS=LaTeX-Templates/paper/generated_compact
./experiments/run_compact_policy_encoding.sh manuscript
```

Generator chỉ chạy khi cả `manuscript_eligible=true` của Thiết kế A và
`evidence_valid=true` của Thiết kế B. File provenance ghi SHA-256 của mọi input
và output; không sao chép số liệu thủ công từ CSV vào bảng cuối.

## 7. Gates trước khi đưa vào bản thảo

### G1: source và execution

- frozen commit tồn tại trên remote;
- worktree sạch;
- EvalMaxSAT đúng SHA-256;
- build và toàn bộ tests pass;
- dry-run resolve đúng 192 MaxSAT + 96 Gurobi tasks;
- VM, CPU affinity, memory và disk checks pass.

### G2: completeness và correctness

- đủ 288 rows, không thiếu hoặc duplicate key;
- đủ bốn MaxSAT cells trên đúng 48 instance hashes;
- mọi row đúng timeout, method, encoding, IC và SB;
- Gurobi chứng minh optimum hoặc infeasibility cho đủ 96 references;
- timeout không bị báo thành infeasible;
- không có status contradiction với Gurobi;
- mọi MaxSAT `OPTIMUM` khớp objective vector Gurobi;
- mọi assignment được independent verifier chấp nhận.

### G3: claim gate cho từng policy

Chỉ phát biểu “Totalizer nhanh hơn” nếu:

- proved count không giảm;
- PAR-2 thấp hơn;
- median SN/TOT speedup lớn hơn 1;
- bootstrap 95% CI nằm hoàn toàn trên 1;
- không có status hoặc objective mismatch.

Nếu gate không đạt, kết quả vẫn được báo trung thực là neutral hoặc
policy-dependent. Không thay dataset, timeout hoặc encoding sau khi xem kết
quả.

## 8. Outline bản thảo

### 1. Introduction

- quyết định HCORAP và ba quality measures;
- hạn chế của weighted exchange rates;
- nhu cầu về objective priorities;
- hai câu hỏi nghiên cứu;
- bốn đóng góp, trong đó policy evidence và encoding evidence được tách rõ.

### 2. Related Work

- home-care allocation/scheduling;
- lexicographic and multi-objective optimization;
- MaxSAT và cardinality encodings;
- khoảng trống: chưa có đánh giá HCORAP kết nối objective order với encoding
  choice trên cùng fixed matrix.

### 3. Problem Formulation and Lexicographic Policy

- sets, assignment variables và feasibility;
- định nghĩa CONT, OT, SIM;
- weighted objective;
- LEX-COS và staged exact solution procedure;
- LEX-OT chỉ được giới thiệu như sensitivity policy.

### 4. Boolean Encoding

- phần model được encode;
- sorting network;
- Totalizer;
- IC và SB được mô tả ngắn vì chúng chỉ thuộc ablation.

### 5. Experimental Methodology

- benchmark và hardware;
- Design A: policy study;
- Design B: fixed 2×2 policy-by-encoding matrix;
- Gurobi/CPLEX và verifier là validation;
- metrics, pairing, bootstrap CI và PAR-2;
- không có Table đếm tổng số runs trong main paper.

### 6. Experimental Results

#### 6.1 Effect of the lexicographic policy

- Table: 48 weighted/LEX-COS objective deltas;
- Figure: joint reductions của CONT và OT;
- paragraph: compatibility trade-off;
- một câu về LEX-OT sensitivity.

#### 6.2 Effect of the cardinality encoding

- Table bốn rows: Weighted-SN, Weighted-TOT, LEX-COS-SN, LEX-COS-TOT;
- report completion, PAR-2, median runtime, RSS, variables và clauses;
- Figure gồm hai paired speedup estimates với 95% CI, một estimate mỗi policy;
- một paragraph ngắn về full runtime distribution nếu còn chỗ;
- một paragraph configuration selection từ ablation cũ.

#### 6.3 Independent validation

- CPLEX khớp Gurobi trên Corrected-v2 audit;
- MaxSAT optimum khớp Gurobi trên full Original matrix;
- independent verifier;
- không dùng bảng solver-completion 300 s cũ.

### 7. Discussion

- khi nào dùng LEX-COS, khi nào weighted hợp lý;
- ý nghĩa của compatibility trade-off;
- encoding effect có transfer giữa policies hay không;
- vai trò khác nhau của Original và Corrected-v2.

### 8. Limitations

- synthetic instances và ba seeds mỗi stratum;
- hai dataset có mục đích khác nhau;
- solver/hardware dependence;
- không có routing/uncertainty/operational data;
- proof files chưa có nếu trạng thái này vẫn đúng khi freeze.

### 9. Conclusion

- một kết luận về objective policy;
- một kết luận về encoding effect, có điều kiện theo claim gate;
- một câu về validation;
- không đưa exploratory hoặc failed pilot vào conclusion.

## 9. Bảng và hình cuối cùng

Giữ tối đa bốn visual chính:

1. policy-effect table;
2. CONT/OT delta scatter plot;
3. four-cell encoding table;
4. two-policy paired speedup plot hoặc cactus plot.

Không dùng experiment-map table, eight-cell main table hoặc corrected-v2
EvalMaxSAT completion table. Full factorial, per-instance records và provenance
được chuyển sang artifact.

## 10. Những nhánh không chạy

- epsilon/Pareto;
- weight sensitivity;
- routing;
- uncertainty;
- relaxed/saturated load profiles;
- Open-WBO;
- full 8-cell factorial ở 3.600 s;
- full CPLEX trên 48 Original instances;
- lặp lại deterministic solver nhiều lần trên cùng instance.

Campaign Corrected-v2 MaxSAT 3.600 s chỉ được mở nếu nhóm tác giả thay đổi scope
để claim high-load MaxSAT scalability. Nó không thuộc campaign mặc định này.

## 11. Definition of done

- [ ] Source/config review hoàn tất; frozen commit đã push.
- [ ] `preflight` pass trên GCP.
- [ ] Đủ 192 MaxSAT + 96 Gurobi rows.
- [ ] `policy_encoding_validation.json` có `evidence_valid=true`.
- [ ] Claim gate được đọc riêng cho weighted và LEX-COS.
- [ ] Table/figure được sinh từ CSV đã freeze, không chép số bằng tay.
- [ ] Xóa đoạn thông báo “matrix has not yet been collected” khỏi bản thảo sau
      khi dữ liệu pass gate.
- [ ] Abstract, Results, Discussion và Conclusion thống nhất cùng số liệu.
- [ ] Artifact giữ configs, resolved matrix, raw JSON, logs, environment và
      checksums.
- [ ] Khôi phục hai Git objects cũ nếu vẫn phân phối historical artifact.
- [ ] Build PDF, kiểm tra overflow, font embedding và page count.
- [ ] Xác minh metadata, deadline và yêu cầu ICIIT 2027 trước khi upload.

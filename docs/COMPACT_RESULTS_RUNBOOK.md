# Kế hoạch kết quả compact cho bản thảo SOICT 2026

Cập nhật trạng thái ngày 07/09/2026. Tài liệu này thay thế cách tổ chức kết quả theo bốn
RQ và bảng đếm 924 runs trong bản thảo chính. Dữ liệu cũ không bị xóa; chúng
được giữ trong artifact để kiểm toán và để giải thích việc chọn cấu hình.

## 1. Hai thiết kế chính

### Thiết kế A: ảnh hưởng của chính sách tối ưu

- Dataset: 48 HCORAP-LC instances, lưu trong các thư mục phân tích
  `corrected_v2`, đủ 16 lớp và ba seeds mỗi lớp.
- So sánh chính: weighted với LEX-COS.
- Solver tạo bằng chứng chính: Gurobi MIP-E, timeout 300 s, một thread.
- Kết quả đã có: 96/96 main-policy runs đạt `OPTIMUM`.
- Kiểm chứng: CPLEX MIP-E trên một seed của mỗi lớp, 32/32 main-policy runs
  khớp Gurobi.
- LEX-OT: chỉ là sensitivity check, không phải policy thứ ba của thiết kế chính.

Không cần chạy thêm Thiết kế A. Các file nguồn là:

- `results_v2/gcp_corrected_exact_analysis/corrected_pairwise_summary.csv`;
- `results_v2/gcp_corrected_exact_analysis/corrected_pairwise_pairs.csv`;
- `results_v2/gcp_corrected_exact_analysis/corrected_policy_summary.csv`;
- `results_v2/gcp_corrected_exact_analysis/corrected_exact_validation.json`.

### Thiết kế B: ảnh hưởng của cardinality encoding

Ma trận cuối cùng có đúng hai yếu tố và bốn cells:

| Policy | Encoding | Implied constraints | Symmetry breaking |
|---|---|---|---|
| weighted | sorting network | none | none |
| weighted | Totalizer | none | none |
| LEX-COS | sorting network | none | none |
| LEX-COS | Totalizer | none | none |

- Dataset: toàn bộ 48 Original instances, seeds 1--3, không loại lớp sau khi
  xem kết quả.
- Solver đo runtime: EvalMaxSAT đã khóa SHA-256.
- Timeout: 3.600 s tích lũy cho toàn policy, giống nghiên cứu gốc.
- Thứ tự: block theo instance, xáo trộn bốn cells trong block bằng seed đã khóa.
- Worker: một process, một vCPU được pin.
- Số runs: `48 × 2 × 2 = 192`.

Gurobi MIP-E chạy cùng 48 instances và hai policies để tạo 96 exact-reference
rows. Để hoàn tất đối chứng commercial, CPLEX MIP-E được chạy trên cùng ma trận
48 instances, hai policies, timeout 3.600 s và một thread. So sánh encoding vẫn
chỉ dùng EvalMaxSAT; bảng so sánh phương pháp sẽ báo riêng EvalMaxSAT-Totalizer,
Gurobi MIP-E và CPLEX MIP-E.

## 2. Vai trò của ablation cũ

Ma trận tám cấu hình ở `results_v2/gcp_original_ablation` không còn là thiết kế
chính. Nó chỉ hỗ trợ quyết định tắt implied constraints và symmetry breaking:

- implied constraints làm PAR-2 tăng trong cả bốn conditional comparisons;
- symmetry breaking không có lợi ích runtime ổn định;
- Totalizer/none/none có PAR-2 thấp nhất trong tám cells ở dữ liệu 300 s.

Không chạy lại tám cells ở timeout 3.600 s. Bản thảo chỉ trình bày một đoạn
configuration-selection ngắn; bảng đầy đủ nằm trong artifact.

## 3. Các file triển khai

- MaxSAT config:
  `experiments/configs/gcp_original_policy_encoding_3600.json`;
- Gurobi reference config:
  `experiments/configs/gcp_original_policy_reference_3600.json`;
- CPLEX baseline config:
  `experiments/configs/gcp_original_cplex_reference_3600.json`;
- runner một lệnh:
  `experiments/run_compact_policy_encoding.sh`;
- runner CPLEX có resume:
  `experiments/run_full_cplex_baseline.sh`;
- analyzer và evidence gate:
  `experiments/analyze_policy_encoding_matrix.py`;
- analyzer ba phương pháp chính xác:
  `experiments/analyze_full_commercial_baseline.py`;
- generator LaTeX có evidence gate:
  `experiments/generate_compact_manuscript_results.py`.

Analyzer tạo bốn bảng và một validation report:

- `policy_encoding_summary.csv`: bốn cells;
- `policy_encoding_pairs.csv`: SN/TOT pairs cho từng policy;
- `policy_encoding_contrasts.csv`: hai direct encoding contrasts;
- `policy_encoding_reference_agreement.csv`: MaxSAT so với Gurobi;
- `policy_encoding_validation.json`: structural, correctness và claim gates.

Sau khi cả Thiết kế A và B qua gate, generator tạo bốn artifact, không
chèn số trực tiếp vào câu văn:

- `compact_result_macros.tex`: các đại lượng dùng trong prose;
- `compact_policy_table.tex`: bảng policy effect;
- `compact_encoding_table.tex`: bảng bốn cells;
- `compact_result_provenance.json`: hash của mọi input và output.

## 4. Chuẩn bị trên GCP

VM publication là Linux x86-64, `c4-highcpu-8`, 16 GB RAM, không dùng Spot.
Không bắt đầu measured phase từ dirty worktree. Tạo một commit sạch sau khi
code, config, tests và outline được review.

```bash
export EVALMAXSAT_BIN=/opt/evalmaxsat/EvalMaxSAT_bin
export GUROBI_HOME=/absolute/path/to/gurobi/platform
export HCORAP_CPU_CORE=0
export HCORAP_EXPECTED_COMMIT=$(git rev-parse HEAD)
export HCORAP_BACKUP_DIR=/mnt/hcorap-backup
```

Preflight build cả hai binaries, chạy tests, kiểm hash solver và resolve chính
xác 192 + 96 tasks:

```bash
./experiments/run_compact_policy_encoding.sh preflight
```

Sau khi preflight pass:

```bash
export CONFIRM_COMPACT_POLICY_ENCODING=YES
./experiments/run_compact_policy_encoding.sh all
```

Có thể chạy và resume từng phase:

```bash
./experiments/run_compact_policy_encoding.sh reference
./experiments/run_compact_policy_encoding.sh maxsat
./experiments/run_compact_policy_encoding.sh analyze
```

Hoàn tất CPLEX baseline sau khi đã commit và push source mới:

```bash
export CPLEX_STUDIO_DIR=/absolute/path/to/CPLEX_Studio
export HCORAP_EXPECTED_COMMIT=$(git rev-parse HEAD)
export HCORAP_MAXSAT_RESULTS=experiments/results/gcp_original_policy_encoding_3600
export HCORAP_GUROBI_RESULTS=experiments/results/gcp_original_policy_reference_3600
./experiments/run_full_cplex_baseline.sh preflight
export CONFIRM_FULL_CPLEX_BASELINE=YES
./experiments/run_full_cplex_baseline.sh all
```

Không ghép 40 CPLEX runs cũ trên 20 instances vào campaign này vì chúng dùng
timeout 300 s. Lệnh `all` chạy hoặc resume đúng 96 CPLEX runs ở timeout 3.600 s,
thu thập kết quả và đối chiếu với cả EvalMaxSAT-Totalizer lẫn Gurobi.

Sau khi đồng bộ thư mục phân tích mới về cùng máy đang chứa kết quả
Corrected-v2 đã kiểm định, sinh các fragment cho bản thảo:

```bash
export HCORAP_POLICY_ANALYSIS=results_v2/gcp_corrected_exact_analysis
export HCORAP_ENCODING_ANALYSIS=\
hcorap_compact_policy_encoding_3600_20260907_082501/analysis
export HCORAP_MANUSCRIPT_RESULTS=LaTeX-Templates/paper/generated_compact
./experiments/run_compact_policy_encoding.sh manuscript
```

Lệnh này dừng nếu một trong hai validation report không qua gate. Sau khi sinh,
kiểm tra `compact_result_provenance.json`. `main_soict.tex` đọc trực tiếp
`compact_policy_table.tex`, `compact_encoding_table.tex` và các macro dùng cho
hai figure cùng phần diễn giải. Không tạo hoặc sửa thủ công các fragment được
sinh.

Runner luôn dùng `--resume`. Không xóa result directory giữa các lần chạy.

## 5. Ngân sách

- MaxSAT worst case: `192 × 3.600 s = 192 core-hour`.
- Gurobi worst case theo timeout: 96 core-hour, nhưng các lớp hiện có thường
  hoàn tất trong vài giây.
- CPLEX worst case theo timeout: 96 core-hour; runner dùng resume và một worker.
- Ước lượng MaxSAT thực tế từ dữ liệu 300 s là khoảng 9--12 giờ tuần tự vì phần
  lớn instances đã kết thúc trước 300 s. Đây chỉ là ước lượng vận hành, không
  phải giới hạn khoa học hoặc kết quả được phép trích dẫn.

Không tăng workers để rút ngắn thời gian nếu việc đó làm các solver processes
cạnh tranh CPU hoặc RAM. Publication timing dùng một worker.

## 6. Evidence gates

`evidence_valid=true` chỉ khi:

1. đủ 192 MaxSAT và 96 Gurobi rows trên đúng 48 instance hashes;
2. mỗi MaxSAT block có đủ hai policies và hai encodings;
3. mọi cell dùng timeout 3.600 s, IC=none, SB=none và đúng solver hash;
4. mọi Gurobi reference dùng một thread, seed 0 và hai optimality gaps bằng 0;
5. không có validation error hoặc hard timeout do wrapper;
6. Gurobi chứng minh optimum hoặc infeasibility cho đủ 96 rows;
7. mọi MaxSAT result đã quyết định có status phù hợp với Gurobi;
8. mọi MaxSAT `OPTIMUM` có objective vector khớp Gurobi.

CPLEX baseline chỉ qua gate khi có đủ 96 rows, CPLEX và Gurobi đều chứng minh
toàn bộ 96 trường hợp, status và objective vector khớp trên mọi cặp, và mọi
MaxSAT result đã quyết định khớp hai commercial references. Analyzer còn kiểm
tra cùng instance hashes, timeout, số thread, CPU affinity và phần source mô
hình giữa các commit.

Timeout không bị ánh xạ thành infeasible. `TIMEOUT_FEASIBLE` chỉ được giữ khi
assignment qua verifier; nó vẫn nhận penalty `2T` trong PAR-2.

Với từng policy, chỉ claim Totalizer nhanh hơn khi đồng thời:

- Totalizer không giảm proved count;
- không có status hoặc objective mismatch;
- bootstrap 95% CI của median SN/TOT runtime ratio nằm hoàn toàn trên 1;
- PAR-2 của Totalizer thấp hơn sorting network.

Nếu gate về tốc độ không đạt, dữ liệu vẫn hợp lệ. Khi đó bản thảo phải báo hiệu
ứng trung tính hoặc phụ thuộc policy; không đổi dataset, timeout hoặc cấu hình
sau khi xem kết quả.

## 7. Ánh xạ vào bản thảo

Phần Experimental Methodology chỉ mô tả hai studies. Không dùng bảng đếm tổng
số runs trong main paper.

Phần Results có cấu trúc:

1. `Effect of the lexicographic policy`: Table policy delta và một scatter plot
   trên HCORAP-LC;
2. `Effect of the cardinality encoding`: một bảng bốn rows grouped theo policy,
   cùng một paired speedup plot hoặc cactus plot;
3. `Independent validation`: một paragraph về CPLEX audit, Gurobi reference và
   verifier;
4. một bảng compact so sánh EvalMaxSAT-Totalizer, Gurobi MIP-E và CPLEX MIP-E
   sau khi full CPLEX gate hợp lệ. Bảng này là đối chứng phương pháp, không trộn
   vào kết luận về SN so với Totalizer.

Validation report đã pass ngày 07/09/2026. Các số
trong Abstract, Results và Conclusion phải lấy từ các macro/table được sinh sau
khi freeze artifact. Bản thảo hiện không còn nhánh kết quả `preliminary`.

## 8. Nhánh không chạy mặc định

Không chạy thêm Pareto/epsilon, weight sensitivity, routing, uncertainty,
Open-WBO, relaxed/saturated profiles hoặc full IC/SB factorial.

Campaign `run_maxsat_lex_3600.sh` trên Corrected-v2 trả lời một câu hỏi khác về
high-load scalability. Chỉ chạy nếu nhóm tác giả quyết định claim MaxSAT giải
được Corrected-v2. Nếu pilot không chọn được candidate tốt hơn baseline thì
dừng, không chạy baseline-only confirmation và không đưa pilot vào main paper.

## 9. Trạng thái provenance cũ

Hai commit ghi trong một số campaign cũ, `0a264adc...` và `a4d810b...`, hiện
không resolve trong local object database. Có thể khôi phục chúng để bảo toàn
artifact lịch sử, nhưng không dùng runtime của các campaign đó để ghép vào ma
trận 3.600 s mới. Ma trận mới phải được chạy hoàn toàn từ một frozen commit.

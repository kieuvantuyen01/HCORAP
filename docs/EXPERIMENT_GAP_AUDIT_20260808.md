# Rà soát khoảng trống thực nghiệm HCORAP — 08/08/2026

## Kết luận ngắn

Bộ kết quả hiện tại đủ để chứng minh Totalizer bảo toàn weighted optimum và có
lợi thế kích thước encoding, nhưng **chưa đủ để khóa bài nộp**. Ba thiếu hụt
nghiêm trọng nhất là:

1. commercial baseline có 29/100 instance bị loại sai bởi parser C++;
2. kết quả weighted lịch sử thiếu 54 run-record và không còn raw JSON/manifest;
3. benchmark cũ có overtime rất hiếm, nên chưa đánh giá được đầy đủ lexicographic,
   epsilon/Pareto và weight sensitivity.

Đã sửa parser, thêm chính sách `LEX-COS = CONT -> OT -> SIM`, thêm corrected-v2
có feasible witness và dựng runner tái lập. Các pilot mới xác nhận mục tiêu mới
tạo ra trade-off thực, nhưng chưa thay thế campaign chính trên GCP. Sau khi rà
soát chi phí, campaign mặc định đã được rút từ 16.040 xuống 4.896 measured runs;
full Pareto, full weight confirmation, uncertainty, load stress và CP baseline
được hoãn, không còn nằm trong lệnh `all`.

## 1. Audit kết quả đang có

Nguồn máy đọc được: `results/audit_20260808.json`.

### 1.1 Weighted MaxSAT trên 800 instance gốc

| Cấu hình | Run record | OPTIMUM | UNSAT | TIMEOUT | Thiếu |
|---|---:|---:|---:|---:|---:|
| ORIGINAL | 800 | 616 | 159 | 25 | 0 |
| SN-none-ss | 800 | 615 | 159 | 26 | 0 |
| SN-both-none | 797 | 614 | 159 | 24 | 3 |
| SN-both-ss | 799 | 614 | 159 | 26 | 1 |
| TOT-none-none | 750 | 571 | 159 | 20 | 50 |
| TOT-none-ss | 800 | 619 | 159 | 22 | 0 |
| TOT-both-none | 800 | 619 | 159 | 22 | 0 |
| TOT-both-ss | 800 | 619 | 159 | 22 | 0 |

Tổng cộng thiếu 54 run-record. Trên các cặp cùng đạt `OPTIMUM`, audit không tìm
thấy weighted-score mismatch giữa baseline và các encoding, nên bằng chứng bảo
toàn optimum là tốt. Tuy nhiên `results/summary_by_config.csv` cũ không nên dùng
cho paper vì số đếm tích lũy theo lớp instance; cần tái sinh summary từ raw.

Điểm yếu provenance: tám CSV đầy đủ không có raw JSON, solver log, instance
manifest hoặc environment snapshot đi kèm. Chỉ chạy bù 54 record sẽ sửa tính
đầy đủ, **không sửa được tính tái lập của 6.346 record còn lại**. Nếu dùng các
số runtime này trong bảng chính, nên chạy lại toàn bộ hai cấu hình được chọn
trên Linux/GCP bằng runner mới; tám cấu hình chỉ nên là screening/ablation phụ.

### 1.2 Commercial baseline

Kết quả cũ có 3.200 dòng: 100 instance, bốn formulation/backend, weighted, hai
lex policy cũ và năm mức epsilon. Với mỗi Gurobi MIP hoặc CPLEX MIP và mỗi
method/delta: 68 `OPTIMUM`, 3 `INFEASIBLE`, 29 `PARSE_ERROR`. Hai CP formulation
có 68 `TIMEOUT_FEASIBLE`, 3 `INFEASIBLE`, 29 `PARSE_ERROR`.

Nguyên nhân đã xác định: benchmark gốc bỏ các user không có service khỏi `#SU`,
trong khi commercial validator yêu cầu số dòng `SU == U`. Semantics đúng là
`SU.size() <= U` và toàn bộ service phải tạo thành đúng một partition. Bản sửa
đã được kiểm tra trên 100/100 instance bằng parser preflight: 0 parse error.

Kết quả commercial cũ không có raw output nên cần chạy lại **đủ 100 instance**,
không chỉ 29 instance lỗi. Campaign commercial tối thiểu đã khóa cho paper:

- Gurobi MIP và CPLEX MIP: `weighted`, `lex-cos`; 100 instance; 1 thread;
  seed 0; gap tuyệt đối/tương đối bằng 0; timeout 300 s;
- tổng 400 measured runs; chỉ so hai exact MIP backend trên cùng hai policy;
- `lex-continuity`, commercial epsilon, corrected-v2 commercial và CP-T/CP-I
  được loại khỏi campaign mặc định vì không trực tiếp củng cố claim chính;
- correctness preflight còn 36 non-measured rows trên ba tiny instances bằng
  Gurobi MIP, CPLEX MIP và reference enumerator.

Máy hiện tại chỉ build được `reference-enumerator`; Gurobi/CPLEX SDK và license
chưa có. Vì vậy chưa thể tạo commercial result mới tại local.

## 2. Chính sách lexicographic

Đã cài đặt đồng nhất trong C++ MaxSAT, commercial C++, Python MaxSAT và CP-SAT:

```text
lex-cos: minimize CONT -> minimize OT -> maximize SIM
```

Policy cũ `lex-continuity` thực chất là `CONT -> SIM -> OT`; vì vậy nó có thể
chấp nhận thêm overtime để đổi lấy rất ít similarity. Trên 68 instance
commercial đã solve, `lex-continuity` và `lex-overtime` cho cùng vector ở 64
instance; bốn instance còn lại cho thấy đúng trade-off OT/SIM này. `lex-cos`
là policy chính hợp lý hơn về mặt vận hành. `lex-overtime` được giữ làm
LEX-OCS sensitivity trên subset nhỏ; `lex-continuity` chỉ còn là kết quả lịch sử.

## 3. Corrected benchmark v2

Generator hiện bảo đảm và ghi lại:

- đúng số service/user và domain ngôn ngữ;
- nested projection theo A/V;
- qualification redundancy trong tập agent nhỏ nhất;
- mỗi service có candidate;
- một full-coverage feasible witness không xung đột agent-slot/user-slot;
- capacity của từng agent không thấp hơn workload witness;
- ba mức tải: relaxed `rho=0.55`, critical `rho=0.85`, saturated `rho=0.98`;
- tách seed calibration/evaluation, TXT + JSON sidecar, SHA-256, diagnostics CSV;
- đọc lại file đã serialize và verify witness độc lập.

Kiểm tra sinh thử đã verify 800 nested saturated projection liên tiếp. Tập pilot
hiện tại có 24 instance quy mô U30 và 32 instance functional quy mô U8. Các
pilot sinh trước khi có witness được chuyển nguyên trạng vào
`experiments/archive/pre_witness_20260808/`; không được dùng làm kết quả paper.

### Campaign corrected-v2 và screening đã khóa

Suite rút gọn có 160 calibration-critical instance (seed 1–10) và 160
evaluation-critical instance (seed 1001–1010). Giai đoạn screen:

| Phase | Instance | Phương pháp/cấu hình | Run | Timeout |
|---|---:|---|---:|---:|
| factorial ablation | 160 original | weighted × 8 cấu hình | 1.280 | 120 s |
| multiobjective screen | 32 critical (seed 1–2) | LEX-COS + delta `0,.05,.10` | 128 | 60 s |
| weight screen | 32 critical (seed 1–2) | bốn cặp `(1,1),(1,4),(4,1),(8,8)` | 128 | 60 s |
| lex scalability gate | 80 original (5 seed/lớp) | weighted + LEX-COS × 2 config | 320 | 300 s |
| **Tổng** |  |  | **1.856** | tối đa 73,6 core-hour |

Publication runner dùng đúng một worker được pin vào một vCPU. Sau screening
phải **đóng băng** cấu hình và timeout rồi mới chạy evaluation seed; không chọn
tham số trên evaluation set. Cổng tự động yêu cầu
LEX-COS hoàn tất ít nhất 60% B0-optimal instances trên ít nhất một cấu hình,
peak RSS không quá 12 GB và không có lỗi/mismatch.

Corrected-v2 confirmatory chỉ chạy 160 evaluation-critical instances bằng
proposed configuration, mỗi instance có weighted và LEX-COS, tổng 320 runs.
Relaxed/saturated load stress đã hoãn.

## 4. Pilot mới đã chạy

Trên corrected-v2 critical `U8/A4/V2`, seed 102, Totalizer + slot-service:

| Policy | SIM | CONT | OT |
|---|---:|---:|---:|
| weighted `(1,1)` | 48 | 1 | 2 |
| `CONT -> SIM -> OT` | 46 | 0 | 2 |
| `LEX-COS` | 38 | 0 | 0 |
| `OT -> CONT -> SIM` | 38 | 0 | 0 |

Sáu mức epsilon tạo ba điểm duy nhất, đều nondominated:

| Delta | SIM | CONT | OT |
|---|---:|---:|---:|
| 0, 0.01, 0.025 | 48 | 1 | 2 |
| 0.05, 0.10 | 46 | 0 | 2 |
| 0.20 | 43 | 0 | 1 |

Lưới trọng số 4×4 tạo sáu vector khác nhau:

```text
(48,1,2), (46,1,1), (42,1,0),
(46,0,2), (43,0,1), (38,0,0)
```

Đây là functional pilot, không phải mẫu đủ lớn để suy luận thống kê. Nó chứng
minh pipeline và thiết kế tải mới đã tạo được tín hiệu mà benchmark cũ thiếu.

Pilot hợp lệ ở quy mô `U30/A10/V5`, `rho≈0.847`, timeout 15 s cho 10/10
`TIMEOUT`. Weighted và similarity-reference của epsilon chưa hoàn tất stage đầu;
LEX-COS hoàn tất `CONT` và `OT` rồi timeout ở `SIM`. Peak RSS lớn nhất quan sát
được là khoảng 344 MB. Kết quả này ủng hộ screening 60 s trên GCP, nhưng chưa đủ
để tăng worker trước khi đo lớp `U40/A25/V5`.

## 5. Epsilon/Pareto còn thiếu gì

Đã có exact decimal ceiling, tuần tự `SIM* -> CONT -> OT -> SIM tie-break`, raw
stage metrics, gộp delta trùng vector và kiểm tra dominance. Với ngân sách rút
gọn, chỉ chạy exploratory screen trên 32 calibration-critical instances với
delta `0,.05,.10`. Còn thiếu nếu muốn đưa Pareto thành claim xác nhận:

1. evaluation trên seed chưa từng dùng để chọn delta;
2. đánh giá tỷ lệ delta trùng điểm, số điểm nondominated/instance và hypervolume
   chỉ khi đã khai báo reference point;
3. so sánh thời gian với LEX-COS và weighted trên cùng instance/config;
4. nếu bước `SIM*` timeout nhiều, không mở rộng campaign; cần cache/reuse
   similarity reference giữa các delta hoặc dùng incremental/native API trước;
5. tăng số delta chỉ sau khi reuse/caching similarity reference được kiểm thử.

Không nên chạy thiết kế cũ `5 delta × 8 config × 800 = 32.000` top-level run.
Nó vừa tốn kém vừa trộn câu hỏi encoding với câu hỏi Pareto.

## 6. Weight sensitivity còn thiếu gì

Code và lưới `{1,2,4,8}²` đã chạy functional pilot. Campaign rút gọn chỉ giữ
bốn cặp đại diện trên 32 calibration-critical instances. Nếu muốn biến weight
sensitivity thành claim xác nhận thì còn cần:

- evaluation trên seed chưa từng dùng để chọn trọng số;
- báo cáo số vector duy nhất, stability của assignment/vector, và transition
  point theo tỷ lệ `wc:wo`;
- khóa trước bốn cặp đại diện để confirm trên evaluation set;
- luôn ghi `P` vì objective dùng `wo * |P| * OT`; không so sánh `wo` mà bỏ qua P;
- không dùng runtime của 16 trọng số như 16 independent datasets trong kiểm định.

## 7. Routing và uncertainty

### Routing

Hiện **không có implementation routing**. Toạ độ chỉ tồn tại trong metadata của
generator để tạo similarity; TXT/model không có depot, travel-time matrix,
service duration, route arc hoặc travel feasibility constraint. Vì vậy không có
“routing experiment” hợp lệ để chạy bằng một flag hiện tại.

Muốn thêm routing phải thay bài toán: định nghĩa depot/location, duration,
travel-time, transition constraints, travel objective và commercial VRP/MIP
baseline. Đây là một contribution mới, có rủi ro làm loãng bài ICIIT. Khuyến
nghị thẳng: để routing ở limitation/future work hoặc một paper riêng.

### Uncertainty

Đã triển khai một **availability-disruption stress test**, không gọi là robust
optimization. Semantics được khóa ở agent-day absence với xác suất 5%, 10%, 20%,
5 scenario seed và common random numbers; absence sets được verify nested giữa
ba mức xác suất. Tám boundary-class base instances tạo 120 scenario, chạy
weighted và LEX-COS với soft coverage.

Analysis báo đồng thời hai chính sách: lịch nominal cố định sau khi loại các
assignment bị disruption (`no recourse`) và nghiệm full re-optimization. Lịch
nominal bắt buộc phải `OPTIMUM`, có assignment và qua verifier; nếu không,
analysis ghi exclusion và trả lỗi. Không được tự suy diễn travel/service-time
uncertainty vì model hiện không chứa hai đại lượng này.

Nhánh này đã bị hoãn khỏi campaign mặc định: nó tạo 256 runs nhưng chưa hỗ trợ
claim robust optimization, trong khi bài hiện tập trung vào encoding và
lexicographic objective. Chỉ chạy lại nếu bản thảo dành một RQ riêng và giới hạn
claim đúng ở availability disruption.

## 8. Reproducibility checklist

Runner mới đã có: instance/binary/solver SHA-256, git commit + dirty diff hash,
resolved task list, seeded randomized order, exact run id, resume, raw JSON,
stderr log, hard timeout, peak RSS (psutil hoặc `ps` fallback), validation của
requested configuration, completeness report, summary theo lớp U/A/V và Pareto
deduplication.

Trước campaign chính còn phải:

- commit/tag một release sạch; không dùng worktree dirty làm artifact cuối;
- build Open-WBO Linux tại commit đã khóa và lưu binary hash;
- lưu compiler/version/flags và image/OS ID;
- dùng cùng timeout, 1 solver thread/run, cùng VM family;
- không chạy workload khác đồng thời;
- lưu toàn bộ result directory, config, source archive/diff và instance sidecars;
- chạy collector và kiểm tra `validation.json: complete=true` trước phân tích;
- chạy commercial bằng raw JSON/native logs mới, không ghép CSV cũ vào campaign.

## 9. GCP C4 execution

Runbook đầy đủ nằm ở `docs/GCP_EXPERIMENT_RUNBOOK.md`. Ma trận rút gọn được khóa
trong `experiments/configs/reduced_campaign_manifest.json`: 4.896 measured runs,
tối đa 326,93 core-hour hay 13,62 ngày tuần tự, giảm khoảng 69,5% so với ma trận
cũ. Script một lệnh là `experiments/run_iciit2027_reduced_campaign.sh`.

Thứ tự bắt buộc vẫn là `preflight -> screen -> GO/NO-GO -> publication phases
-> package`. Các script đã có exact expected counts, blocked-instance order,
resume, CPU affinity, one-thread publication mode, hard-error rejection,
factorial/lex/screening analysis và artifact checksum. Commercial campaign còn
cần Gurobi/CPLEX SDK + license thật trên VM; preflight giải tiny instances bằng
Gurobi MIP, CPLEX MIP và reference enumerator trước khi chạy 400 measured rows.

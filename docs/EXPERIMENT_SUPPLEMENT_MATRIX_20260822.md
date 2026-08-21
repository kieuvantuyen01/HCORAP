# Ma trận bằng chứng và phần thực nghiệm còn thiếu — 22/08/2026

Tài liệu này thay thế kết luận về trạng thái dữ liệu trong
`COMPACT_EXPERIMENT_MATRIX_20260820.md`. Nguồn máy-đọc chuẩn là
`experiments/configs/reduced_campaign_manifest.json`; báo cáo kiểm tra tự động
là `results_reduced/publication_evidence_audit.json`.

## 1. Kết luận rà soát dữ liệu hiện có

`results_reduced` có đủ **732 measured rows** của năm campaign đã chạy. Không có
run ID trùng, collector validation đều complete, config hash khớp config hiện
tại, không có validation error hoặc optimum không qua verifier. Sau khi sửa
phân loại infeasible, cross-paradigm validation đạt 40/40 groups: 36 groups cả
ba solver chứng minh optimum và cùng objective, 4 groups cùng chứng minh
infeasible, không có disagreement.

Phần corrected-v2 EvalMaxSAT có đủ cấu trúc 144/144 rows nhưng không đủ bằng
chứng để ước lượng policy effect: weighted--LEX-COS chỉ có 1 jointly-optimum
pair và LEX-COS--LEX-OCS chỉ có 3. Có 137/144 timeouts. Vì vậy campaign này chỉ
được dùng để báo giới hạn khả năng mở rộng của formulation MaxSAT ở timeout
300 s; không được dùng các median policy delta làm kết luận.

Một provenance blocker độc lập vẫn tồn tại: cả 732 rows ghi source commit
`0a264adc59f2bc8e01bd00aeca3b3078b0faa04b`, nhưng commit này chưa resolve trong
clone hiện tại. Cần push commit đó từ clone GCP cũ và fetch vào publication
clone trước khi chạy phần bổ sung. Không được sửa tay trường provenance.

## 2. Ma trận publication sau rà soát

| ID | Vai trò | Dataset / solver / policy | Runs | Trạng thái |
|---|---|---|---:|---|
| C1 | RQ2--RQ3 | original 48 × 8 factorial cells × weighted, EvalMaxSAT | 384 | đã có, dùng được sau khi giải quyết provenance |
| C2 | RQ1 gốc | original 42 × weighted/LEX-COS, EvalMaxSAT | 84 | đã có; 33 jointly-optimum pairs |
| C3 | scalability | corrected critical 48 × weighted/LEX-COS/LEX-OCS, EvalMaxSAT | 144 | đã có; không đủ policy-effect evidence |
| C4 | cross-solver | original subset 20 × Gurobi/CPLEX × weighted/LEX-COS | 80 | đã có |
| C5 | cross-solver | cùng subset 20 × EvalMaxSAT × weighted/LEX-COS | 40 | đã có |
| C6 | corrected policy primary | corrected critical 48 × Gurobi × 3 policies | 144 | **cần chạy** |
| C7 | independent audit | corrected seed 1002, 16 strata × CPLEX × 3 policies | 48 | **cần chạy** |
|  | **Tổng measured** |  | **924** | **732 có, 192 còn thiếu** |

Measured timeout được giữ ở 300 s. Tổng worst-case measured là 277.200 s =
77 core-hours; phần còn thiếu là 57.600 s = 16 core-hours. Không mở lại Pareto,
weight sensitivity, uncertainty, routing, Open-WBO hoặc các IC/SB mode phụ.

## 3. Calibration không tính vào kết quả bài báo

Trước C6--C7 chạy 8 corrected calibration instances × 3 policies × 2 exact
solvers = 48 rows. Chỉ tiếp tục nếu:

- đủ đúng 48 rows, không duplicate, technical error hoặc unverified optimum;
- mỗi backend chứng minh cả ba policies trên ít nhất 6/8 instances;
- Gurobi và CPLEX không có status disagreement;
- mọi jointly-optimum weighted score hoặc lexicographic vector đều khớp.

Calibration không được nhập vào runtime table hay policy-effect estimate.

## 4. Evidence gates cho C6--C7

Analyzer `analyze_corrected_exact_evidence.py` chỉ mở corrected policy table khi
đồng thời đạt:

- C6 đủ 144 rows, 48 instances, 16 strata và đúng ba policies;
- ít nhất 36/48 instances có cả ba Gurobi runs đạt `OPTIMUM`;
- ít nhất 12/16 strata có từ hai all-policy-optimum seeds;
- C7 đủ 48/48 CPLEX optimum groups trên seed 1002;
- Gurobi--CPLEX không có status/objective disagreement trên audit subset;
- không có technical error hoặc unverified solver-reported optimum.

Objective delta chỉ tính trên jointly-optimum pairs. Nếu gate thất bại, mã sinh
bản thảo dừng; không thay threshold, seed hoặc sample sau khi xem kết quả.

## 5. Cấu hình và outputs mới

| Thành phần | File |
|---|---|
| calibration matrix | `gcp_commercial_corrected_calibration.json` |
| Gurobi primary | `gcp_commercial_corrected_primary.json` |
| CPLEX stratum audit | `gcp_commercial_corrected_audit.json` |
| calibration gates | `corrected_commercial_calibration_gates.json` |
| evidence gates | `corrected_exact_evidence_gates.json` |
| exact analyzer | `analyze_corrected_exact_evidence.py` |
| full evidence audit | `audit_publication_evidence.py` |

Expected analysis outputs là `corrected_policy_summary.csv`,
`corrected_pairwise_pairs.csv`, `corrected_pairwise_summary.csv`,
`corrected_solver_agreement.csv` và `corrected_exact_validation.json` trong
`gcp_corrected_exact_analysis`.

## 6. Lệnh chạy đúng phần còn thiếu

Trên clean GCP publication commit, sau khi đặt các biến môi trường trong
runbook:

```bash
bash experiments/run_remaining_corrected_evidence.sh --check-only
export CONFIRM_PUBLICATION_CAMPAIGN=YES
bash experiments/run_remaining_corrected_evidence.sh
```

Script xác nhận 732 rows cũ trước khi chạy, thực hiện calibration, chạy/resume
đúng 192 measured rows còn lại, phân tích, checkpoint và audit. Nếu cần resume,
chạy lại cùng lệnh; authoritative run ID ngăn chạy lại completed rows.

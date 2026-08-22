# Runbook thực nghiệm compact HCORAP trên GCP — ICIIT 2027

Cập nhật ngày 22/08/2026. Manifest chuẩn là
`experiments/configs/reduced_campaign_manifest.json`; giải thích khoa học và
audit dữ liệu cũ nằm tại
[`COMPACT_EXPERIMENT_MATRIX_20260820.md`](COMPACT_EXPERIMENT_MATRIX_20260820.md).
Ma trận bằng chứng thực tế và phần còn thiếu nằm tại
[`EXPERIMENT_SUPPLEMENT_MATRIX_20260822.md`](EXPERIMENT_SUPPLEMENT_MATRIX_20260822.md).

## 1. Phạm vi chạy

Campaign chỉ giữ bằng chứng trực tiếp cho factorial encoding/strengthening,
LEX-COS/LEX-OCS, corrected-v2 policy validation và Gurobi/CPLEX exact-objective
agreement.

| Campaign | Runs | Timeout | Worst-case core-hour |
|---|---:|---:|---:|
| original factorial: 48 × 8 × weighted | 384 | 300 s | 32,00 |
| original policy: 42 × R × weighted/LEX-COS | 84 | 300 s | 7,00 |
| corrected-v2 EvalMaxSAT scalability: 48 × R × 3 policies | 144 | 300 s | 12,00 |
| Gurobi/CPLEX: 20 × 2 backends × 2 policies | 80 | 300 s | 6,67 |
| EvalMaxSAT commercial: 20 × R × weighted/LEX-COS | 40 | 300 s | 3,33 |
| corrected-v2 Gurobi policy primary: 48 × 3 policies | 144 | 300 s | 12,00 |
| corrected-v2 CPLEX audit: 16 × 3 policies | 48 | 300 s | 4,00 |
| **Tổng measured** | **924** |  | **77,00** |

Ngoài bảng có 4 EvalMaxSAT LEX-COS calibration và 48 corrected exact-solver
calibration runs ở timeout 300 s, cùng 18 commercial correctness-smoke runs ở
timeout 30 s; không đưa timing của chúng vào bài. Pareto/epsilon, weight
confirmation, uncertainty, routing và
corrected load stress không được gọi bởi phase `all`.

### Căn cứ chọn timeout

[Nghiên cứu gốc](<../Optimizing resource allocation in home care services using MaxSAT.tex>)
dùng giới hạn 1 giờ và 16 GB cho mỗi execution trên Xeon E-2234; giá trị lớn
nhất trong các thời gian trung bình của nhóm đã được chứng nhận ở bảng kết quả
là 158,34 s. Dữ liệu EvalMaxSAT diagnostic cũ trong
`results/comparison_pivot.csv` còn có
một optimum hợp lệ ở 270,647 s. Vì vậy 120 s sẽ censor một ca đã biết có thể
được chứng nhận, còn 300 s vượt thời gian chứng nhận lớn nhất đã quan sát trong
audit. Mốc 300 s được áp dụng đồng nhất cho mọi measured top-level run; với
lexicographic policy, đây là ngân sách cộng dồn cho toàn bộ stages.

Đây là một giới hạn compact đã khai báo trước, không phải phép tái lập giới hạn
1 giờ của nghiên cứu gốc. Mọi timeout vẫn được giữ trong solved count và PAR-2;
không diễn giải timeout là infeasible. `hard_grace_seconds=60` chỉ cho tiến trình
cha thu output rồi cưỡng bức dừng một binary treo, không cộng thêm solver time.

## 2. Protocol khóa

Tất cả measured runs dùng một non-Spot GCP `c4-highcpu-8`, Ubuntu 24.04 LTS,
một solver process bị giới hạn vào một pinned vCPU. Gurobi/CPLEX còn được khóa
một native solver thread. Không chạy workload khác. EvalMaxSAT phải là đúng
Linux x86-64 binary đã dùng trên GCP trước đây:

```text
SHA-256 = 97614c996e1173ca0672ec46da153656046db1d84b9362a8561161ee750779f7
```

```text
B = sorting-network / implied none / symmetry none
R = totalizer / implied both / symmetry slot-service
LEX-COS = CONT -> OT -> SIM
LEX-OCS = OT -> CONT -> SIM
```

`R` là nhãn reference, không mặc định hàm ý nhanh hơn. B--R comparison tái sử
dụng hai cell của factorial. RQ1 và commercial validation chạy lại weighted
cùng LEX-COS trên sample riêng; mọi measured row dùng cùng timeout 300 s.

Factorial dùng đủ 16 lớp × seeds 1--3. Original LEX-COS dùng 14 lớp × cùng ba
seeds; loại `30_15_4` và `40_25_5` vì đã được xem trong commercial development.
Corrected-v2 dùng 16 critical strata × evaluation seeds 1001--1003 và chạy cả
weighted, LEX-COS, LEX-OCS trong cùng campaign block.

## 3. Chuẩn bị VM

Trong fresh clone của publication commit:

```bash
sudo apt-get update
sudo apt-get install -y build-essential git jq libgmp-dev python3 python3-pip \
  python3-venv tmux util-linux curl rsync
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e '.[test]' psutil
```

Đặt bản EvalMaxSAT Linux x86-64 đã lưu trữ của nhóm lên VM và kiểm hash:

```bash
sudo install -d /opt/evalmaxsat
sudo install -m 0755 /path/to/archived/EvalMaxSAT_bin \
  /opt/evalmaxsat/EvalMaxSAT_bin
sha256sum /opt/evalmaxsat/EvalMaxSAT_bin
```

Không dùng file `EvalMaxSAT` ở root của worktree trên macOS: đó là Mach-O
ARM64 và có hash khác. Campaign chỉ chấp nhận binary Linux có hash ở trên.

Cài Gurobi và IBM ILOG CPLEX Optimization Studio theo license của nhóm, rồi đặt:

```bash
export EVALMAXSAT_BIN=/opt/evalmaxsat/EvalMaxSAT_bin
export GUROBI_HOME=/absolute/path/to/gurobi/platform
export CPLEX_STUDIO_DIR=/absolute/path/to/CPLEX_Studio
export HCORAP_CPU_CORE=0
export HCORAP_EXPECTED_COMMIT=iciit2027-exp-v3
export HCORAP_BACKUP_DIR=/mnt/hcorap-backup
```

`HCORAP_BACKUP_DIR` phải ở ngoài worktree và nằm trên persistent storage.
Measured phase bị từ chối nếu HEAD không đúng revision, worktree dirty, backup
directory không hợp lệ hoặc VM thiếu 20 GB disk trống.

Kiểm tra revision và ngân sách:

```bash
git status --short
git rev-parse HEAD
git rev-parse "${HCORAP_EXPECTED_COMMIT}^{commit}"
python3 experiments/validate_campaign_manifest.py
python3 experiments/validate_publication_campaign.py
```

Expected output của validator phải là 924 measured runs, 277.200 worst-case
seconds, 77 core-hour và contract `valid: true`.

## 4. Preflight

```bash
mkdir -p vm-logs
tmux new -s hcorap
bash experiments/gcp_prepare_and_run.sh preflight \
  2>&1 | tee vm-logs/preflight.log
bash experiments/gcp_prepare_and_run.sh commercial-preflight \
  2>&1 | tee vm-logs/commercial-preflight.log
```

Preflight phải:

- xác minh Linux, ít nhất 8 vCPU, 15.000.000 KiB RAM và 20 GB trống;
- xác minh EvalMaxSAT binary bằng SHA-256 và chạy weighted/LEX-COS smoke;
- build C++ và chạy all tests;
- sinh/verify corrected-v2 suite, witness, hash, matrix và seed split;
- dry-run các MaxSAT configs và kiểm exact instance/task counts;
- chạy weighted/LEX-COS correctness smoke; trước measured matrix, yêu cầu ít
  nhất 2/4 LEX-COS development instances đạt optimum trong 300 s;
- force-build commercial binary, kiểm license và chạy 18 smoke rows;
- đạt 6/6 weighted/LEX-COS instance-policy agreement groups giữa Gurobi, CPLEX
  và reference enumerator.

Nếu lỗi, sửa code/config trên commit mới. Không đổi binary, timeout hoặc config
rồi resume vào result directory cũ.

## 5. Chạy toàn bộ campaign

```bash
export CONFIRM_PUBLICATION_CAMPAIGN=YES
bash experiments/run_all_remaining_publication.sh
```

Wrapper cố định `WORKERS=1`, bật `CONFIRM_FULL_CAMPAIGN=YES`, ghi log timestamp
vào `vm-logs/`, ngăn hai campaign chạy đồng thời, checkpoint cả khi bị ngắt và
chạy:

```text
build/test/benchmark/task-count checks
-> commercial preflight + 18 smoke rows
-> warm-up
-> 4-row EvalMaxSAT LEX-COS scalability gate
-> C1 factorial hard gate
-> C2 original weighted/LEX-COS
-> C3 corrected-v2 weighted/LEX-COS/LEX-OCS
-> C4 commercial MIP + C5 EvalMaxSAT commercial weighted/LEX
-> corrected exact-solver calibration
-> C6 Gurobi corrected primary + C7 CPLEX stratum audit
-> analysis -> package -> checkpoint
```

Factorial vừa là evidence vừa là hard gate. Pipeline dừng nếu thiếu row, có
technical/validation error, unverified optimum, paired weighted-objective
mismatch hoặc peak RSS vượt 12 GB. Gate không yêu cầu R nhanh hơn B và không có
evidence branch hậu nghiệm cho epsilon/weight/LEX.

### Chỉ chạy phần còn thiếu sau audit 22/08

Nếu `experiments/results` đã chứa đủ 732 measured rows C1--C5, không gọi phase
`all`. Khôi phục source commit ghi trong `environment.json`, checkout clean
publication commit mới, rồi chạy:

```bash
bash experiments/run_remaining_corrected_evidence.sh --check-only
export CONFIRM_PUBLICATION_CAMPAIGN=YES
bash experiments/run_remaining_corrected_evidence.sh
```

Wrapper này kiểm tra 732 rows cũ, chạy 48 calibration rows không đo, sau đó chỉ
chạy/resume C6--C7 (192 measured rows, 16 core-hours worst case). Nó dừng trước
solver nếu source commit của dữ liệu cũ không resolve trong clone.

## 6. Chạy theo phase và resume

```bash
bash experiments/gcp_prepare_and_run.sh screen \
  2>&1 | tee vm-logs/screen.log
jq . experiments/results/screening_decision.json

export CONFIRM_FULL_CAMPAIGN=YES
bash experiments/gcp_prepare_and_run.sh original-primary \
  2>&1 | tee vm-logs/original-primary.log
bash experiments/gcp_prepare_and_run.sh corrected-primary \
  2>&1 | tee vm-logs/corrected-primary.log
bash experiments/gcp_prepare_and_run.sh commercial \
  2>&1 | tee vm-logs/commercial.log
bash experiments/gcp_prepare_and_run.sh corrected-commercial-evidence \
  2>&1 | tee vm-logs/corrected-commercial-evidence.log
bash experiments/gcp_prepare_and_run.sh analyze \
  2>&1 | tee vm-logs/analyze.log
bash experiments/gcp_prepare_and_run.sh package \
  2>&1 | tee vm-logs/package.log
```

Runner resume theo authoritative run ID và không chạy lại completed rows. Sau
reboot, chạy lại đúng phase command. Chỉ resume row chưa hoàn tất hoặc lỗi kỹ
thuật; không thay sample hậu nghiệm. Nếu binary/config thay đổi thì dùng commit
mới và result directory mới.

Sau mỗi phase:

```bash
find experiments/results -path '*/validation.json' \
  -print -exec jq '.complete' {} \;
find experiments/results -path '*/analysis*.json' -print
df -h .
```

Checkpoint tự động sao chép results, native logs và phase logs sang
`HCORAP_BACKUP_DIR`. Không dùng Spot VM và không phân tích một primary subset
chưa đủ expected rows.

### 6.1. Kiểm tra chuyển giao Totalizer-only cho LEX-COS

Đây là phần bổ sung duy nhất còn đáng chạy sau ma trận 924 rows. Nó kiểm tra
liệu cấu hình nhanh nhất trong factorial weighted có còn tốt khi giải
continuity-first trên corrected-v2 hay không. Pilot ghép cặp hai cấu hình trên
16 strata của seed 1002:

```text
T0 = totalizer / implied none / symmetry none
R  = totalizer / implied both / symmetry slot-service
policy = LEX-COS; timeout = 300 s; workers = 1
```

Kiểm tra ma trận và chạy tự động:

```bash
bash experiments/run_corrected_lex_encoding_transfer.sh --check-only
export CONFIRM_LEX_TRANSFER=YES
bash experiments/run_corrected_lex_encoding_transfer.sh all \
  2>&1 | tee vm-logs/lex-encoding-transfer.log
```

Pilot có 32 runs, tối đa 2,67 core-hour. Script chỉ chạy confirmation 96 runs
(8 core-hour tối đa) nếu T0 đạt ít nhất một gate: thêm ròng 2 optimum, tiến thêm
một criterion trên ít nhất 4/16 cặp, hoặc giảm PAR-2 ít nhất 10%. Nếu cả ba gate
đều không đạt, script dừng sau pilot. Không gộp 16 pilot instances vào estimate
của confirmation; confirmation tự chạy lại đủ 48 cặp dưới cùng commit và
protocol.

## 7. Exact outputs cần có

| Output | Điều kiện |
|---|---|
| `gcp_original_ablation` | 384 rows; 48/cell |
| `gcp_evalmaxsat_lex_calibration` | 4 non-measured LEX-COS rows; ít nhất 2 optimum |
| `gcp_original_lex_primary` | 84 rows; 42/policy under R |
| `gcp_corrected_primary` | 144 rows; 48/policy for weighted/LEX-COS/LEX-OCS |
| `gcp_commercial_corrected_calibration` | 48 non-measured rows; calibration gate pass |
| `gcp_commercial_corrected_primary` | 144 rows; 48 Gurobi rows/policy |
| `gcp_commercial_corrected_audit` | 48 rows; 16 CPLEX rows/policy on seed 1002 |
| `gcp_commercial_original` | 80 rows; 20/backend/policy |
| `gcp_maxsat_commercial_validation` | 40 rows; 20/policy under R |
| `gcp_primary_analysis` | valid; 8 cells, 12 contrasts, 48 B--R, 42 original policy pairs, 48 corrected sensitivity pairs |
| `gcp_corrected_analysis` | structurally valid; EvalMaxSAT scalability only |
| `gcp_corrected_exact_analysis` | `manuscript_eligible=true`; exact policy evidence |
| `gcp_cross_paradigm_analysis` | valid; 20 groups/policy |
| `gcp_corrected_lex_encoding_transfer_*` | optional; pilot decision plus 96-row confirmation only after `GO` |

Objective deltas chỉ dùng jointly-optimum, verifier-passing pairs. Solved count
và PAR-2 giữ toàn bộ rows, kể cả timeout. Three-backend agreement chỉ được báo
khi cả EvalMaxSAT, Gurobi và CPLEX đều chứng minh optimum.

## 8. Sinh bản thảo và artifact

```bash
python3 experiments/generate_manuscript_results.py
python3 experiments/audit_publication_evidence.py
bash experiments/package_experiment_artifacts.sh
```

Generator sinh một figure hai panel, hai bảng compact, quantitative prose,
abstract findings, conclusion và `manuscript-provenance.json`. Review PDF/diff,
commit generated bundle, rồi chạy `freeze_manuscript_bundle.py` từ clean commit;
freeze kiểm SHA-256 của CSV/JSON nguồn và fragments.

Artifact nằm trong `artifacts/hcorap_iciit2027_<UTC>.tar.gz` cùng `.sha256` và
phải chứa source, configs, benchmark/sidecars, raw results, native logs,
generated tables, HCORAP binaries, EvalMaxSAT SHA-256 và environment snapshot.
Chỉ đặt `HCORAP_INCLUDE_SOLVER_BINARY=YES` nếu giấy phép EvalMaxSAT cho phép
phân phối lại binary trong artifact.

Không nhập runtime từ `results/`, `results_addition/` hoặc pilot directories
vào publication tables. Các nguồn đó chỉ còn vai trò diagnostic/historical.

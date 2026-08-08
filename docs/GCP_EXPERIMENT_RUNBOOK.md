# Runbook thực nghiệm rút gọn HCORAP trên GCP — ICIIT 2027

## 1. Quyết định rút gọn

Campaign mặc định chỉ giữ bằng chứng trực tiếp cho bốn đóng góp của bài:

1. tác động của Totalizer, implied constraints và symmetry breaking;
2. hàm mục tiêu `LEX-COS = CONT -> OT -> SIM`;
3. kiểm tra trên corrected benchmark v2;
4. đối chứng exact MIP bằng Gurobi và CPLEX.

Ma trận cũ có 16.040 measured runs, tối đa xấp xỉ 1.171 core-hour. Ma trận đã
khóa trong `experiments/configs/reduced_campaign_manifest.json` còn 4.896 runs,
1.176.960 giây timeout cộng dồn, tức 326,93 core-hour hay 13,62 ngày nếu mọi run
đều chạm timeout và chỉ dùng một worker. Mức giảm là khoảng 69,5%. Thời gian
thực tế thường thấp hơn vì các run giải xong sớm.

Các nhánh bị loại khỏi lệnh `all`:

| Nhánh | Quyết định | Lý do |
|---|---|---|
| factorial 8 cấu hình trên đủ 800 instance | thay bằng 160-instance ablation + 800-instance baseline/proposed | 6.400 runs tốn nhất, trong khi claim chính chỉ cần phân rã trên mẫu paired và xác nhận hai cấu hình đầu-cuối |
| full Pareto/epsilon confirmation | hoãn; chỉ giữ screen 3 mức delta | mỗi delta cần nhiều stage MaxSAT; không phải contribution chính |
| full weight confirmation | hoãn; chỉ giữ screen 4 trọng số | weighted objective gốc và lexicographic mới quan trọng hơn một lưới tham số rộng |
| corrected-v2 relaxed/saturated load stress | hoãn | tăng kích thước nhưng không trực tiếp kiểm tra encoding hay policy chính |
| availability uncertainty | hoãn | đây mới là stress test, chưa phải robust optimization |
| commercial epsilon/corrected/CP | loại khỏi campaign chính | chi phí license/runtime cao và làm loãng so sánh exact baseline |
| routing | không chạy | model chưa có depot, duration, travel time hay route arcs |

Không diễn giải epsilon, weight và uncertainty screen như kết luận xác nhận.
Nếu reviewer yêu cầu, các phase `pareto`, `weight-confirmation`, `uncertainty`
vẫn được giữ trong code để chạy bổ sung sau.

## 2. Ma trận còn lại

| Campaign | Thiết kế | Runs | Timeout | Tối đa core-hour |
|---|---|---:|---:|---:|
| original factorial ablation | 160 instance × 8 cấu hình, weighted | 1.280 | 120 s | 42,67 |
| corrected multiobjective screen | 32 instance × LEX-COS/epsilon `0,.05,.10` | 128 | 60 s | 2,13 |
| corrected weight screen | 32 instance × `(1,1),(1,4),(4,1),(8,8)` | 128 | 60 s | 2,13 |
| original lex scalability | 80 instance × 2 cấu hình × weighted/LEX-COS | 320 | 300 s | 26,67 |
| original weighted primary | đủ 800 instance × baseline/proposed | 1.600 | 300 s | 133,33 |
| original LEX-COS primary | 280 held-out instance × baseline/proposed | 560 | 300 s | 46,67 |
| LEX-OCS sensitivity | 80 instance × baseline/proposed | 160 | 300 s | 13,33 |
| corrected-v2 primary | 160 evaluation-critical × weighted/LEX-COS | 320 | 300 s | 26,67 |
| commercial original | 100 instance × Gurobi/CPLEX × weighted/LEX-COS | 400 | 300 s | 33,33 |
| **Tổng measured** |  | **4.896** |  | **326,93** |

Ngoài bảng trên có 36 commercial correctness-smoke runs, timeout 30 giây,
không đưa vào số liệu bài báo. Screen và confirmatory rows đều được giữ trong
artifact, nhưng phải ghi rõ vai trò khi trình bày.

## 3. Protocol máy và solver

Tất cả measured runs dùng cùng một non-Spot VM `c4-highcpu-8`, Ubuntu 24.04
LTS, một solver process, một thread và một vCPU được pin. Không chạy workload
khác đồng thời. Open-WBO phải ở đúng commit:

```text
80f3073e41028b219b0b0ad7c61fba28351f88e6
```

Baseline và proposed được khóa như sau:

```text
baseline = sorting-network / implied none / symmetry none
proposed = totalizer / implied both / symmetry slot-service
```

LEX-COS chính dùng thứ tự `CONT -> OT -> SIM`; LEX-OCS sensitivity dùng
`OT -> CONT -> SIM`. Hai lớp `30_15_4` và `40_25_5` đã dùng trong commercial
development nên bị loại khỏi LEX-COS confirmatory set. Corrected-v2 dùng
calibration seed 1–10 và evaluation seed 1001–1010, không giao nhau.

## 4. Chuẩn bị VM

Trong một fresh clone của commit được dùng cho bài:

```bash
sudo apt-get update
sudo apt-get install -y build-essential git jq libgmp-dev python3 python3-pip \
  python3-venv tmux util-linux curl
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e '.[test]' psutil
```

Build Open-WBO đã khóa:

```bash
export HCORAP_OPEN_WBO_ROOT=/opt/hcorap-open-wbo
sudo git clone https://github.com/sat-group/open-wbo.git "$HCORAP_OPEN_WBO_ROOT"
sudo chown -R "$(id -u):$(id -g)" "$HCORAP_OPEN_WBO_ROOT"
git -C "$HCORAP_OPEN_WBO_ROOT" checkout 80f3073e41028b219b0b0ad7c61fba28351f88e6
git -C "$HCORAP_OPEN_WBO_ROOT" submodule update --init --recursive
make -C "$HCORAP_OPEN_WBO_ROOT" -j8
```

Cài Gurobi và IBM ILOG CPLEX Optimization Studio theo license của nhóm. Sau đó
khai báo đường dẫn SDK thực tế:

```bash
export OPEN_WBO_SOURCE_DIR=/opt/hcorap-open-wbo
export OPEN_WBO_BIN=/opt/hcorap-open-wbo/open-wbo
export OPEN_WBO_COMMIT=80f3073e41028b219b0b0ad7c61fba28351f88e6
export GUROBI_HOME=/absolute/path/to/gurobi/platform
export CPLEX_STUDIO_DIR=/absolute/path/to/CPLEX_Studio
export HCORAP_CPU_CORE=0
```

Không chạy publication campaign từ worktree dirty. Trước khi chạy:

```bash
git status --short
git rev-parse HEAD
python3 experiments/validate_campaign_manifest.py
```

## 5. Preflight

Chạy preflight trước để kiểm tra VM, compiler, solver, tests, benchmark và exact
task count:

```bash
mkdir -p vm-logs
tmux new -s hcorap
bash experiments/gcp_prepare_and_run.sh preflight 2>&1 | tee vm-logs/preflight.log
bash experiments/gcp_prepare_and_run.sh commercial-preflight \
  2>&1 | tee vm-logs/commercial-preflight.log
```

Preflight sẽ:

- yêu cầu Linux, ít nhất 8 vCPU, 15.000.000 KiB RAM và 20 GB trống;
- xác minh source và binary Open-WBO cùng thuộc pinned commit;
- build C++ và chạy toàn bộ pytest;
- sinh và verify 160 calibration + 160 evaluation-critical corrected-v2
  instances, full-coverage witness, hash, exact Cartesian matrix và seed split;
- dry-run tám MaxSAT config đang hoạt động và từ chối sai expected count;
- build lại Gurobi/CPLEX binary, thử license và chạy 36 correctness-smoke rows;
- đối chiếu nghiệm exact giữa commercial solvers và reference enumerator.

Nếu preflight lỗi, sửa code/config và commit mới. Không đổi semantics, timeout,
binary hay config rồi `--resume` một result directory cũ.

## 6. Chạy toàn bộ ma trận rút gọn

Script một lệnh cho toàn bộ campaign là:

```bash
export CONFIRM_REDUCED_CAMPAIGN=YES
bash experiments/run_iciit2027_reduced_campaign.sh
```

Script cố định `WORKERS=1`, bật xác nhận publication phase, ghi log có timestamp
vào `vm-logs/`, rồi chạy theo thứ tự:

```text
build/test -> benchmark verify -> budget/dry-run verify -> warm-up
-> screen + GO/NO-GO -> original primary -> corrected primary
-> commercial preflight/primary -> artifact package
```

`all` dừng ngay nếu screen trả `NO-GO`, có technical/validation error, optimum
không verified, objective mismatch, license lỗi hoặc campaign thiếu rows. Không
đặt biến để lách gate. Đọc `experiments/results/screening_decision.json`, sửa
implementation hoặc thu hẹp claim rồi chạy lại trên commit mới.

## 7. Chạy theo phase và resume

Nếu muốn kiểm tra từng chặng:

```bash
bash experiments/gcp_prepare_and_run.sh screen 2>&1 | tee vm-logs/screen.log
jq . experiments/results/screening_decision.json

export CONFIRM_FULL_CAMPAIGN=YES
bash experiments/gcp_prepare_and_run.sh original-primary \
  2>&1 | tee vm-logs/original-primary.log
bash experiments/gcp_prepare_and_run.sh corrected-primary \
  2>&1 | tee vm-logs/corrected-primary.log
bash experiments/gcp_prepare_and_run.sh commercial \
  2>&1 | tee vm-logs/commercial.log
bash experiments/gcp_prepare_and_run.sh package \
  2>&1 | tee vm-logs/package.log
```

Mọi runner dùng authoritative run ID và `--resume`; sau reboot chỉ cần chạy lại
đúng command. Không duplicate completed runs. Warm-up dùng 10 calibration
instances ngoài số measured và không ghi vào bảng manuscript.

Kiểm tra nhanh sau mỗi phase:

```bash
find experiments/results -path '*/validation.json' \
  -print -exec jq '.complete' {} \;
find experiments/results -path '*/analysis*.json' -print
```

## 8. Dữ liệu và bảng được phép dùng

| Claim | Nguồn dữ liệu |
|---|---|
| hiệu ứng Totalizer/implied/symmetry | `gcp_primary_analysis/factorial_*` |
| baseline so với proposed trên 800 instance | `gcp_original_weighted_primary` |
| LEX-COS quality/runtime | `gcp_primary_analysis/lex_confirmatory_*` |
| LEX-OCS policy sensitivity | `lex_policy_sensitivity_pairs.csv` |
| corrected-v2 weighted/LEX-COS | `gcp_corrected_primary` |
| commercial agreement/runtime | `gcp_commercial_original` |
| epsilon/weight exploratory evidence | hai `gcp_*_screen_analysis` directories |
| reproducibility | environment, resolved campaign, hashes, raw JSON và native logs |

Không dùng raw runtime lịch sử thiếu manifest/provenance trong bảng chính.
Không gọi three-delta screen là Pareto frontier confirmation. Không gọi
availability code là robust optimization. Routing, full Pareto, full weight
confirmation, uncertainty và load stress phải nằm ở limitation/future work,
trừ khi được chạy bổ sung và báo cáo tách biệt.

Artifact cuối nằm trong `artifacts/hcorap_iciit2027_<UTC>.tar.gz` cùng file
`.sha256`. Archive chứa source, configs, benchmark/sidecars, raw results,
native logs, generated tables, binary và environment snapshot.

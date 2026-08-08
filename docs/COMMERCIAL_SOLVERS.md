# Cấu hình Gurobi MIP, CPLEX MIP và CP Optimizer

Tài liệu này mô tả implementation C++ dùng cho thực nghiệm thương mại. Mục tiêu
là giữ **cùng feasible schedules, cùng objective semantics, cùng timeout và
cùng verifier** giữa các backend:

| Backend CLI | Formulation | Solver |
|---|---|---|
| `gurobi-mip` | `mip-e` | Gurobi Optimizer |
| `cplex-mip` | `mip-e` | CPLEX Optimizer |
| `cplex-cp` | `cp-t` | IBM ILOG CP Optimizer |
| `cplex-cp` | `cp-i` | IBM ILOG CP Optimizer |

`reference-enumerator` chỉ là oracle vét cạn cho instance rất nhỏ. Không đưa
backend này vào bảng runtime.

## 1. Kiến trúc đã triển khai

Executable chung là `bin/release/hcorap_commercial`. Các thành phần:

- `src/hcorap_commercial.cpp`: CLI, cumulative timeout, chính sách
  weighted/lexicographic/epsilon, JSON và kiểm tra optimum giữa các stage;
- `src/proposed/cpp/commercial/HCORAPMIPModel.cpp`: dựng một intermediate
  representation MIP-E duy nhất;
- `GurobiMIPBackend.cpp` và `CplexMIPBackend.cpp`: dịch đúng intermediate
  representation đó sang hai API solver, không dựng hai mô hình riêng;
- `CplexCPBackend.cpp`: hai formulation CP-T và CP-I;
- `CommercialTypes.cpp`: kiểm tra instance và verifier độc lập từ danh sách
  assignment \((a,s,t)\);
- `ReferenceBackend.cpp`: oracle nhỏ để kiểm thử objective policies.

MIP-E chỉ tạo \(x_{ast}\) trên candidate khả thi. Biến coverage, projected
agent--service, workload, continuity và overtime đều được liên kết hai chiều.
Overtime dùng threshold binaries nên giá trị chính xác ngay cả khi stage hiện
tại không tối ưu overtime.

CP-T dùng một allowed-assignment tuple cho mỗi service. Tuple chứa agent, slot,
coverage, reward và hai collision key. Trong soft coverage, mỗi service có
sentinel riêng; vì vậy hai service không được phục vụ không xung đột giả trong
`allDifferent`.

CP-I chỉ tạo optional interval \(I_{as}\) nếu cặp agent--service có ít nhất một
candidate slot. `alternative` chọn đúng một agent khi master interval present;
`forbidStart` áp calendar candidate; `noOverlap` chỉ nhận các interval thực sự
tồn tại. Master interval là mandatory trong full coverage và optional trong
soft coverage.

## 2. Build không có SDK thương mại

Lệnh sau luôn build được executable và oracle nhỏ:

```sh
make -j4 YICES=0 hcorap_commercial
./bin/release/hcorap_commercial --list-backends
```

Ba backend thương mại sẽ báo `"compiled": false`. Chạy chúng sẽ dừng với thông
báo build thiếu SDK, thay vì âm thầm dùng một solver khác.

## 3. Build với Gurobi

Cần cài Gurobi Optimizer, C++ headers/libraries và license hợp lệ. Đặt
`GUROBI_HOME` tại thư mục chứa `include/gurobi_c++.h` và `lib/`:

```sh
export GUROBI_HOME=/absolute/path/to/gurobi/platform
make -j4 YICES=0 GUROBI=1 hcorap_commercial
./bin/release/hcorap_commercial --list-backends
```

Makefile tự nhận diện versioned core library, ví dụ `libgurobi130`. Có thể
override nếu installation dùng tên khác:

```sh
make -j4 YICES=0 GUROBI=1 \
  GUROBI_CORE_LIB=gurobi130 hcorap_commercial
```

Nếu C++ ABI của `libgurobi_c++` không khớp compiler, cần build lại C++ wrapper
theo hướng dẫn trong installation của Gurobi và dùng cùng compiler cho HCORAP.

## 4. Build với CPLEX Optimizer và CP Optimizer

Đặt `CPLEX_STUDIO_DIR` tại root của IBM ILOG CPLEX Optimization Studio, nơi có
ba thư mục `concert`, `cplex` và `cpoptimizer`:

```sh
export CPLEX_STUDIO_DIR=/absolute/path/to/CPLEX_Studio
make -j4 YICES=0 CPLEX=1 hcorap_commercial
./bin/release/hcorap_commercial --list-backends
```

Makefile tự tìm tên architecture dưới `concert/lib/*/static_pic`. Nếu có nhiều
architecture hoặc layout tùy chỉnh, truyền đúng tên thư mục:

```sh
make -j4 YICES=0 CPLEX=1 \
  CPLEX_ARCH=YOUR_ARCH_DIRECTORY hcorap_commercial
```

Một binary có thể chứa cả hai họ SDK:

```sh
make -j4 YICES=0 GUROBI=1 CPLEX=1 hcorap_commercial
```

Build objects được tách theo suffix `_gurobi`/`_cplex`, nên chuyển giữa các
build mode không tái sử dụng nhầm object đã compile với macro khác. Binary cuối
cùng vẫn ở `bin/release/hcorap_commercial`.

## 5. Cấu hình mặc định đã khóa

| Thuộc tính | Gurobi MIP | CPLEX MIP | CP Optimizer |
|---|---:|---:|---:|
| threads/workers | 1 | 1 | 1 |
| random seed | 0 | 0 | 0 |
| time mode | wall clock | wall clock | `ElapsedTime` |
| relative MIP gap | 0 | 0 | không áp dụng |
| absolute MIP gap | 0 | 0 | không áp dụng |
| linear feasibility tolerance | \(10^{-6}\) | \(10^{-6}\) | không áp dụng |
| integrality tolerance | \(10^{-5}\) | \(10^{-5}\) | miền nguyên CP |
| absolute optimality tolerance CP | không áp dụng | không áp dụng | 0 |
| relative optimality tolerance CP | không áp dụng | không áp dụng | 0 |

CLI tương ứng:

```text
--threads 1 --seed 0 --mip-gap 0 --absolute-mip-gap 0
```

Mỗi policy có một cumulative end-to-end timeout. Trước mỗi solver call, driver
trừ thời gian đã dùng bởi parse, license/backend initialization và các stage
trước. Mỗi backend tiếp tục trừ model-build time trước khi đặt native solver
time limit. JSON báo riêng `build_seconds`, `solve_seconds` và
`verification_seconds`.

Mọi run trong schema so sánh hiện tại bắt buộc MIP gap bằng 0. Nếu cho gap
dương, CLI từ chối run vì incumbent gần tối ưu không được phép mang nhãn
`OPTIMUM`; điều này đặc biệt quan trọng khi optimum của một stage trở thành
inherited bound của stage sau.

Với MIP, `--parameter-file` được đọc trước, sau đó các tham số fairness ở trên
được CLI ghi đè. Điều này cho phép tuned campaign thay các tham số search nhưng
không thay threads, seed, time limit, optimality gaps, linear feasibility
tolerance hoặc integrality tolerance. Riêng CPLEX MIP, `DetTimeLimit` cũng được
reset về mặc định sau khi đọc parameter file để một deterministic-time limit
không cạnh tranh với global wall-clock timeout; `ClockType` được khóa ở giá trị
2 để `TimeLimit` luôn đo wall-clock thay vì CPU time. CP Optimizer không nhận
parameter file trong implementation này; các tham số khóa được đặt trực tiếp
bằng API.

CPLEX MIP chỉ ánh xạ detailed status `AbortTimeLim` thành `TIMEOUT` hoặc
`TIMEOUT_FEASIBLE`. `AbortDetTimeLim`, `AbortUser`, node limit, memory limit,
solution limit và các nguyên nhân dừng khác được báo `ERROR`, kể cả khi solver
đã có incumbent. Quy tắc này ngăn PAR-2 và tỷ lệ timeout bị trộn với những run
dừng bởi một termination control khác trong parameter file. Một tuned campaign
phụ dùng parameter file vì vậy không được đặt termination limit ngoài
wall-clock `TimeLimit`.

Campaign GCP chính trong `experiments/configs/gcp_commercial_*.json` không dùng
parameter file: nó dùng native defaults của solver version đã ghi vào JSON và
chỉ ghi đè các tham số fairness/determinism ở bảng trên. Không được thêm
parameter file vào giữa lúc resume campaign chính.

## 6. Chạy từng backend

Weighted full coverage:

```sh
./bin/release/hcorap_commercial INSTANCE.txt \
  --backend gurobi-mip --formulation mip-e \
  --method weighted --wc 1 --wo 1 \
  --threads 1 --seed 0 --timeout 300 \
  --print-assignments --output gurobi.json

./bin/release/hcorap_commercial INSTANCE.txt \
  --backend cplex-mip --formulation mip-e \
  --method weighted --wc 1 --wo 1 \
  --threads 1 --seed 0 --timeout 300 \
  --print-assignments --output cplex_mip.json
```

Hai CP formulation:

```sh
./bin/release/hcorap_commercial INSTANCE.txt \
  --backend cplex-cp --formulation cp-t \
  --method weighted --threads 1 --seed 0 --timeout 300 \
  --print-assignments --output cp_t.json

./bin/release/hcorap_commercial INSTANCE.txt \
  --backend cplex-cp --formulation cp-i \
  --method weighted --threads 1 --seed 0 --timeout 300 \
  --print-assignments --output cp_i.json
```

Các policy khác dùng cùng option:

```sh
--method lex-continuity
--method lex-overtime
--method epsilon --delta 0.05
--soft-coverage
```

Native log là opt-in để tránh I/O log làm nhiễu campaign mặc định:

```sh
--solver-log solver.log
```

## 7. Preset và campaign runner

Các preset:

```text
experiments/configs/commercial/gurobi_mip.env
experiments/configs/commercial/cplex_mip.env
experiments/configs/commercial/cplex_cp_t.env
experiments/configs/commercial/cplex_cp_i.env
experiments/configs/commercial/all_backends.env
experiments/configs/commercial/reference_tiny.env
```

Chạy ma trận bốn solver/formulation:

```sh
. experiments/configs/commercial/all_backends.env
TIMEOUT=300 \
experiments/run_commercial_experiments.sh \
  experiments/results/commercial_main \
  INSTANCE_1.txt INSTANCE_2.txt
```

Tiếp tục campaign:

```sh
RESUME=1 experiments/run_commercial_experiments.sh \
  experiments/results/commercial_main \
  INSTANCE_1.txt INSTANCE_2.txt
```

Runner ghi `environment.txt`, `manifest.tsv` và một JSON cho mỗi
instance/backend/formulation/method/delta. Tên run có 12 ký tự đầu của SHA-256
instance để tránh collision giữa hai file cùng basename. Các biến thường dùng:

```text
METHODS="weighted lex-continuity lex-overtime"
RUN_EPSILON=1
DELTAS="0 0.01 0.025 0.05 0.10"
SOFT_COVERAGE=0
NATIVE_LOGS=0
GUROBI_PARAM_FILE=/path/to/gurobi.prm
CPLEX_PARAM_FILE=/path/to/cplex.prm
```

Không dùng cùng result directory sau khi đổi binary, solver version, parameter
file hoặc schema.

## 8. JSON và điều kiện chấp nhận nghiệm

Status chung:

- `OPTIMUM`: solver báo optimal và incumbent qua verifier;
- `INFEASIBLE`: mô hình được chứng minh infeasible;
- `TIMEOUT_FEASIBLE`: hết giờ nhưng có incumbent hợp lệ;
- `TIMEOUT`: hết giờ chưa có incumbent;
- `ERROR`: lỗi API, status không hỗ trợ, verifier từ chối hoặc objective/bound
  không nhất quán.

Mỗi incumbent được tính lại coverage, similarity, continuity, overtime,
workload và weighted reference score. Verifier kiểm tra candidate membership,
service uniqueness/full coverage, agent-slot, user-slot và capacity. Driver còn
kiểm tra inherited bounds. Khi solver báo `OPTIMUM`, objective của incumbent
phải khớp best bound trong sai số \(10^{-4}\).

Nếu một stage sau timeout mà chưa có incumbent, JSON có thể giữ nghiệm hợp lệ
từ stage gần nhất; `incumbent_stage_index` cho biết chính xác nghiệm thuộc stage
nào. Không được diễn giải nghiệm đó là optimum của policy chưa hoàn tất.

## 9. Validation hiện tại và bước kiểm tra bắt buộc trên máy có license

Trong môi trường phát triển không có Gurobi/CPLEX SDK, các phần đã kiểm tra là:

- build không SDK và phát hiện backend availability;
- intermediate MIP-E;
- CLI/policy/cumulative timeout;
- verifier và oracle vét cạn trên tiny instances;
- weighted, hai lexicographic policy, epsilon, soft coverage và timeout
  incumbent.

Trước campaign chính, trên máy có license phải:

1. build lần lượt `GUROBI=1`, `CPLEX=1`, rồi combined build;
2. chạy `--list-backends` và lưu output;
3. chạy cả bốn solver/formulation trên `tests/instances/tradeoff.txt` và
   `partial_coverage.txt`;
4. xác nhận objective vector giống oracle;
5. chạy một full-coverage infeasible instance và một timeout nhỏ;
6. kiểm tra native log ghi đúng threads, seed, time limit và gap;
7. khóa solver version, compiler version, license type và parameter-file hash
   trong protocol trước khi chạy benchmark.

Tài liệu API tham chiếu:

- [Gurobi C++ API](https://docs.gurobi.com/projects/optimizer/en/current/reference/cpp/overview.html)
- [Gurobi model API](https://docs.gurobi.com/projects/optimizer/en/current/reference/cpp/model.html)
- [CPLEX Concert model](https://www.ibm.com/docs/en/icos/22.1.2?topic=application-creating-model-ilomodel)
- [CPLEX solve status và detailed status](https://www.ibm.com/docs/en/icos/22.1.2?topic=application-solving-model-ilocplex)
- [CPLEX clock type](https://www.ibm.com/docs/en/icos/22.1.2?topic=parameters-clock-type-computation-time)
- [CPLEX deterministic time limit](https://www.ibm.com/docs/en/icos/22.1.2?topic=parameters-deterministic-time-limit)
- [CP Optimizer `IloCP`](https://www.ibm.com/docs/en/icos/22.1.2?topic=c-ilocp-2)
- [CP Optimizer `allowedAssignments`](https://www.ibm.com/docs/en/icos/22.1.1?topic=functions-iloallowedassignments)
- [CP Optimizer `alternative`](https://www.ibm.com/docs/en/icos/22.1.1?topic=classes-iloalternative)
- [CP Optimizer `noOverlap`](https://www.ibm.com/docs/en/icos/22.1.2?topic=c-ilonooverlap-1)
- [CP Optimizer `forbidStart`](https://www.ibm.com/docs/en/icos/22.1.1?topic=functions-iloforbidstart)

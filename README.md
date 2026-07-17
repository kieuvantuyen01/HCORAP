# HCORAP: thực nghiệm MaxSAT đồng nhất bằng C++

Repository gồm mã C++/instances đi kèm bài báo **Optimizing Resource
Allocation in Home Care Services using MaxSAT** và phần mở rộng nghiên cứu đa
mục tiêu. Đường chạy dùng cho bảng thực nghiệm chính hiện được triển khai hoàn
toàn bằng C++ để tránh so runtime Python với baseline C++.

Các thành phần chính:

- `bin/release/hcorap2sat`: encoder C++ của tác giả, dùng cho audit/tái lập;
- `bin/release/hcorap_multi`: weighted, hai policy lexicographic và
  epsilon-constraint bằng C++;
- `src/proposed/cpp/encodings`: hard model C++ và các strategy encoding dùng
  cho ablation;
- `experiments/run_cpp_experiments.sh`: điều phối campaign không có Python trên
  đường đo thời gian;
- `src/proposed/hcorap`: oracle/verifier/generator Python dùng để kiểm thử chéo,
  không dùng làm implementation trong bảng runtime chính.

Kế hoạch nghiên cứu nằm trong `Ke_hoach_nghien_cuu_HCORAP.tex`; quy tắc benchmark
chi tiết nằm trong `docs/FAIR_EXPERIMENT_PROTOCOL.md`.

## Build C++

```sh
make -j4 YICES=0
```

Lệnh này tạo cả `hcorap2sat` và `hcorap_multi` với cùng compiler flags. Phần mở
rộng dùng một backend MaxSAT C++ chung. Backend đã kiểm thử là Open-WBO 2.1 tại
commit `80f3073e41028b219b0b0ad7c61fba28351f88e6`.

```sh
git clone https://github.com/sat-group/open-wbo.git /path/to/open-wbo
git -C /path/to/open-wbo checkout 80f3073e41028b219b0b0ad7c61fba28351f88e6
git -C /path/to/open-wbo submodule update --init --recursive
make -C /path/to/open-wbo -j4
```

Open-WBO cần GMP. Xem hướng dẫn macOS và protocol khóa solver/compiler trong
`docs/FAIR_EXPERIMENT_PROTOCOL.md`.

## Chạy các phương pháp C++

Weighted baseline tương đương objective gốc:

```sh
./bin/release/hcorap_multi INSTANCE.txt \
  --solver /path/to/open-wbo/open-wbo \
  --method weighted --wc 1 --wo 1 --timeout 300
```

Hai chính sách lexicographic:

```sh
./bin/release/hcorap_multi INSTANCE.txt \
  --solver /path/to/open-wbo/open-wbo \
  --method lex-continuity --timeout 300

./bin/release/hcorap_multi INSTANCE.txt \
  --solver /path/to/open-wbo/open-wbo \
  --method lex-overtime --timeout 300
```

Một điểm epsilon-constraint:

```sh
./bin/release/hcorap_multi INSTANCE.txt \
  --solver /path/to/open-wbo/open-wbo \
  --method epsilon --delta 0.05 --timeout 300
```

Encoding cardinality mặc định vẫn là sorting network. Biến thể Totalizer mới
được chọn độc lập với phương pháp tối ưu:

```sh
./bin/release/hcorap_multi INSTANCE.txt \
  --solver /path/to/open-wbo/open-wbo \
  --method weighted --cardinality-encoding totalizer --timeout 300
```

Hai biến thể có cùng ngữ nghĩa threshold; Totalizer được mã hóa hai chiều để
verifier và các tầng lexicographic không phụ thuộc vào giá trị tùy ý của biến
phụ. Trường `cardinality_encoding` trong JSON phân biệt từng run.

Implied constraints được chọn bằng một option độc lập:

```sh
./bin/release/hcorap_multi INSTANCE.txt \
  --solver /path/to/open-wbo/open-wbo \
  --method weighted \
  --cardinality-encoding totalizer \
  --implied-constraints both-plus \
  --timeout 300
```

Các giá trị hợp lệ là `none` (mặc định), `user-slots`, `slot-capacity`, `both`
và `both-plus`. Xem định nghĩa, điều kiện đúng và ablation protocol trong
`docs/IMPLIED_CONSTRAINTS.md`.

Symmetry breaking là trục độc lập thứ ba:

```sh
./bin/release/hcorap_multi INSTANCE.txt \
  --solver /path/to/open-wbo/open-wbo \
  --cardinality-encoding totalizer \
  --implied-constraints both-plus \
  --symmetry-breaking slot-service \
  --method weighted --timeout 300
```

Các giá trị là `none` (mặc định), `slots`, `services`, `slot-service`, `all`.
Xem `docs/SYMMETRY_BREAKING.md`.

Thêm `--soft-coverage` cho overload stress test. Coverage được tối đa hóa và
cố định trước các objective còn lại. `--print-assignments` đưa assignment vào
JSON; `--output result.json` ghi kết quả ra file.

Timeout là ngân sách tích lũy cho toàn bộ policy. `elapsed_seconds` đo:

```text
parse + encode + serialize WCNF + solve + C++ verification
```

Mỗi stage còn báo số biến/clauses, thời gian dựng công thức, thời gian solver và
optimum. Mọi nghiệm `OPTIMUM` đều phải qua verifier C++ độc lập trước khi được
ghi nhận.

## Campaign C++ tái lập

```sh
SOLVER_ID=open-wbo-2.1-80f3073 TIMEOUT=300 \
experiments/run_cpp_experiments.sh \
  /absolute/path/to/open-wbo \
  experiments/results/cpp_pilot \
  tests/instances/tradeoff.txt \
  instances/paperInstances/TXT_10-25_4-5_U30/instance_30_15_4_47.txt
```

Để chạy riêng Totalizer hoặc chạy ablation paired, source cấu hình tương ứng
trước lệnh trên:

```sh
. experiments/configs/cpp/totalizer.env
# hoặc:
. experiments/configs/cpp/cardinality_ablation.env
```

Ablation implied constraints hoặc toàn bộ ma trận $2\times5$:

```sh
. experiments/configs/cpp/implied_ablation.env
# hoặc:
. experiments/configs/cpp/full_encoding_ablation.env
```

Ablation symmetry hoặc toàn bộ ma trận $2\times5\times5$:

```sh
. experiments/configs/cpp/symmetry_ablation.env
# hoặc:
. experiments/configs/cpp/full_configuration_matrix.env
```

Runner sinh một JSON cho mỗi run và `manifest.tsv` chứa SHA-256, instance,
ba trục encoding, method, delta, output và exit code. Nó còn sinh
`configuration_matrix.tsv`, `runs.csv` và `configuration_summary.csv`; hai file
CSV dùng UTF-8 BOM và mở trực tiếp bằng Excel. Dùng result directory
mới khi đổi schema/campaign. Shell chỉ điều phối; phần được đo vẫn là C++
end-to-end.

Screening chỉ chạy weighted baseline (không chạy lexicographic hoặc epsilon):

```sh
METHODS=weighted RUN_EPSILON=0 TIMEOUT=60 \
experiments/run_cpp_experiments.sh \
  /absolute/path/to/open-wbo \
  experiments/results/weighted_screening \
  INSTANCE.txt
```

`METHODS` nhận danh sách gồm `weighted`, `lex-continuity` và
`lex-overtime`; `RUN_EPSILON=0` tắt toàn bộ các mức `DELTAS`. Mặc định runner
vẫn chạy ba method cùng năm mức epsilon như trước. Đặt `METHODS=''` và
`RUN_EPSILON=1` để chạy epsilon-only.

## Kiểm thử

Kiểm thử C++ end-to-end với cùng Open-WBO binary:

```sh
OPEN_WBO_BIN=/absolute/path/to/open-wbo \
PYTHONPATH=src/proposed python3 -m pytest -q
```

Python ở đây chỉ là test harness/oracle. Bộ test kiểm tra:

- WCNF legacy hợp lệ và optimum bằng RC2 trên instance nhỏ;
- sorting network và Totalizer có cùng optimum trên regression instance;
- năm cấu hình implied constraints giữ nguyên optimum và qua verifier;
- năm cấu hình symmetry-breaking giữ nguyên optimum và qua verifier;
- toàn bộ ma trận $2\times5\times5$ giữ nguyên optimum trên instance đối xứng;
- user-slot cardinality đúng trong cả full và partial coverage;
- weighted score và đối chiếu với encoder C++ gốc;
- hai policy lexicographic cho đúng hai nghiệm trade-off;
- epsilon tại delta 0 và 0.2;
- soft coverage và verifier nghiệm;
- parser, generator, metrics và timeout semantics.

`tests/instances/tradeoff.txt` có các kết quả mong đợi:

| Phương pháp | SIM | CONT | OT |
|---|---:|---:|---:|
| weighted `(1,1)` | 9 | 1 | 0 |
| lex-continuity | 8 | 0 | 1 |
| lex-overtime | 9 | 1 | 0 |
| epsilon, delta 0 | 9 | 1 | 0 |
| epsilon, delta 0.2 | 8 | 0 | 1 |

Weighted có hai nghiệm đồng tối ưu điểm 8; bảng ghi nghiệm Open-WBO trả về.

## Vai trò của Python

Cài package kiểm thử khi cần:

```sh
python3 -m pip install --user -e ".[test,cpsat]"
```

Các lệnh `python3 -m hcorap` vẫn hữu ích để inspect, sinh benchmark paired/nested,
brute-force/RC2 cross-check và kiểm tra nghiệm. Chúng không được dùng để tạo số
runtime so sánh trực tiếp với C++. CP-SAT Python chỉ là prototype xác minh mô
hình; nếu đưa CP-SAT vào bảng hiệu năng chính, phải chuyển sang OR-Tools C++ và
áp cùng timing scope/timeout.

## Định nghĩa metric

- `coverage = COV`, chuẩn hóa bởi `S`;
- `similarity = SIM`, chuẩn hóa bởi `r_max*S`, kèm upper bound theo candidate;
- `continuity = sum_q max(0,D_q-1)`; giá trị nhỏ hơn tốt hơn;
- `overtime = sum_a max(0,workload_a-HN(a))`;
- `overtime_cost = |P|*overtime`.

Trong overload, continuity chỉ được so sánh giữa các nghiệm có cùng coverage.

## Mã C++ gốc của bài báo

Sinh công thức MaxSAT từ một instance:

```sh
./bin/release/hcorap2sat -e=1 -f=dimacs -S=0 INSTANCE.txt
```

`hcorap2sat` được giữ làm mốc audit. Bảng so sánh phương pháp chính dùng
`hcorap_multi --method weighted --wc 1 --wo 1` làm B0 C++ tương đương, vì nhờ
đó B0/lex/epsilon có cùng parser, hard model, writer và backend. Optimum B0 được
cross-check với mã gốc trên instance nhỏ và instance chính thức trước campaign.

Với full coverage, quan hệ được kiểm tra là:

```text
objective_gốc - sum_q(|SEQ(q)| - 1)
  = SIM - CONT - |P|*OT
```

Tái lập runtime của binary gốc phải được báo ở bảng audit riêng nếu encoding,
WCNF format hoặc backend chưa hoàn toàn đồng nhất; không trộn số đó vào bảng
chính rồi diễn giải như khác biệt thuật toán.

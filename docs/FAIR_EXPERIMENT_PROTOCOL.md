# Giao thức thực nghiệm công bằng cho HCORAP

## Quyết định triển khai

Đường chạy dùng để so sánh chính thức là **C++ end-to-end**:

- `hcorap_multi --method weighted`: B0, tương đương objective weighted gốc;
- `hcorap_multi --method lex-continuity`: ưu tiên continuity;
- `hcorap_multi --method lex-cos`: `CONT -> OT -> SIM`, policy chính;
- `hcorap_multi --method lex-overtime`: ưu tiên overtime;
- `hcorap_multi --method epsilon`: similarity-budget epsilon-constraint;
- một binary EvalMaxSAT Linux x86-64 đã khóa SHA-256 cho tất cả các phương pháp.

Python trong `src/proposed/hcorap/` chỉ là oracle, generator, verifier thứ hai và công
cụ phân tích. Không lấy runtime Python để đưa vào bảng so sánh hiệu năng chính.
Baseline CP-SAT Python cũng không được đưa vào bảng runtime chính; chỉ bổ sung
baseline này sau khi đã chuyển sang OR-Tools C++ hoặc tách thành một thí nghiệm
cross-framework được ghi nhãn rõ ràng.

Mã C++ mở rộng nằm trong `src/proposed/cpp/encodings`. Cardinality encoding là
một biến thí nghiệm tường minh: `sorting-network` là baseline mặc định và
`totalizer` là biến thể. Totalizer dùng clauses hai chiều để output thứ `k` vẫn
tương đương với “ít nhất `k+1` input đúng”, giống semantics của sorting network.
Không thay đổi cardinality encoding ngầm giữa các phương pháp.

Implied constraints là một trục ablation riêng với năm mức `none`, `user-slots`,
`slot-capacity`, `both`, `both-plus`. Bảng so sánh method phải cố định cả
`cardinality_encoding` lẫn `implied_constraints`. Định nghĩa chi tiết nằm trong
`IMPLIED_CONSTRAINTS.md`.

Symmetry breaking là trục thứ ba với năm mức `none`, `slots`, `services`,
`slot-service`, `all`. Chỉ các lớp tương đương exact mới được order. So sánh
method phải cố định `symmetry_breaking`; ablation symmetry phải chạy paired.
Định nghĩa và điều kiện an toàn nằm trong `SYMMETRY_BREAKING.md`.

Thư viện SMT C++ kế thừa có lỗi trong helper `addPBGEQ`: vector phủ định chưa
được cấp kích thước và cận bù được tính nhưng không truyền vào `addPB`. Để giữ
baseline audit nguyên vẹn, mã mới không sửa helper cũ mà mã hóa cục bộ
`sum(q_i*x_i) >= K` thành `sum(q_i*(not x_i)) <= sum(q_i)-K`. Các tầng cố định
coverage/similarity và các test lexicographic/epsilon đều đi qua đường này.

B2 đầy đủ ban đầu dùng grid delta `0, 0.01, 0.025, 0.05, 0.10`. Cả full
confirmation và exploratory screen hiện nằm ngoài compact ICIIT campaign. Code
vẫn phải lưu similarity reference optimum, ceiling lower bound, realized loss
và bốn stage full-coverage nếu nhánh này được chạy trong nghiên cứu sau; không
được nhập các pilot epsilon lịch sử vào publication tables.

## Hai phép so sánh phải tách riêng

1. **So sánh phương pháp tối ưu (bảng chính):** mọi phương pháp dùng cùng parser,
   hard model, WCNF writer, compiler flags, EvalMaxSAT binary và verifier C++.
   B0 là weighted `(wc, wo)=(1,1)`. Compact campaign so B0 với LEX-COS và
   LEX-OCS dưới cùng cấu hình/thời hạn; epsilon-constraint không nằm trong
   measured matrix.
2. **Tái lập mã tác giả (bảng audit):** chạy `hcorap2sat` nguyên gốc để đối chiếu
   feasibility và optimum. Không trộn runtime này vào bảng chính khi encoding,
   WCNF format hoặc backend khác nhau. Chỉ so runtime trực tiếp sau khi cả hai
   dùng cùng solver binary và cùng quy tắc tính thời gian.

## Backend đã kiểm thử

EvalMaxSAT Linux x86-64 được dùng làm backend tham chiếu, nhất quán với solver
family được nghiên cứu gốc chọn sau preliminary comparison. Binary publication
được khóa bằng SHA-256:

```sh
97614c996e1173ca0672ec46da153656046db1d84b9362a8561161ee750779f7
```

Đặt binary đã lưu trữ lên GCP và kiểm tra trước khi chạy:

```sh
install -m 0755 /path/to/archived/EvalMaxSAT_bin /opt/evalmaxsat/EvalMaxSAT_bin
sha256sum /opt/evalmaxsat/EvalMaxSAT_bin
```

Không thay solver giữa các phương pháp trong cùng một campaign. File Mach-O
ARM64 ở máy phát triển không phải publication binary. Preflight trên Linux phải
chứng minh được weighted optimum và cả ba stage LEX-COS trên WCNF do encoder
hiện hành sinh ra trước khi bất kỳ measured row nào được chạy.

## Phạm vi đo thời gian

Trường `elapsed_seconds` của C++ bao gồm:

```text
parse instance + dựng encoding + ghi WCNF + giải + kiểm chứng nghiệm
```

Timeout là ngân sách tích lũy cho toàn bộ policy trên một instance, không phải
mỗi tầng. JSON còn lưu thời gian dựng công thức và thời gian solver theo tầng.
Với phép đo bộ nhớ, bọc cùng binary bằng `/usr/bin/time`; không thay đổi command
hoặc timeout giữa các phương pháp.

Verifier C++ kiểm tra model, objective values và mọi inherited bound. Trạng thái
optimality vẫn dựa trên `OPTIMUM` của backend; không mô tả kết quả là
proof-carrying hoặc independently certified nếu chưa kiểm tra proof trace
MaxSAT độc lập.

## Điều kiện phải khóa trước campaign

- cùng máy, một CPU core, governor/power mode và giới hạn RAM;
- cùng compiler, cờ `-O3 -DNDEBUG -std=c++11` và cùng commit mã nguồn;
- cùng EvalMaxSAT binary SHA-256 và tham số solver;
- cùng thứ tự instance ngẫu nhiên đã lưu seed;
- chạy blocked-instance: xáo thứ tự instance và xáo cấu hình trong từng block;
- publication run dùng một worker được pin vào một vCPU;
- cùng timeout tích lũy và định nghĩa `OPTIMUM`, `TIMEOUT`, `UNSATISFIABLE`;
- cùng cardinality encoding trong bảng so sánh method; nếu ablation encoding,
  chạy paired và phân tầng kết quả theo `cardinality_encoding`;
- cùng implied-constraint config trong bảng so sánh method; nếu ablation implied
  constraints, chạy paired và phân tầng theo `implied_constraints`;
- cùng symmetry-breaking config trong bảng so sánh method; nếu ablation
  symmetry, chạy paired và phân tầng theo `symmetry_breaking`;
- chạy warm-up ngoài tập đo; lặp ít nhất ba lần nếu backend không deterministic;
- lưu SHA-256 instance, command, stdout/stderr, JSON nghiệm và manifest;
- không loại timeout hoặc lỗi kỹ thuật sau khi xem chất lượng nghiệm.

KPI hiệu năng gồm solved count, time-to-proof, PAR-2, peak memory, số biến,
hard clauses và soft clauses. KPI chất lượng gồm coverage, similarity,
continuity, overtime và overtime cost do verifier C++ tính lại.

## Chạy campaign C++

```sh
make -j4 YICES=0
chmod +x experiments/run_cpp_experiments.sh
SOLVER_ID=evalmaxsat-97614c996e11 TIMEOUT=300 \
experiments/run_cpp_experiments.sh \
  /opt/evalmaxsat/EvalMaxSAT_bin \
  experiments/results/cpp_pilot \
  tests/instances/tradeoff.txt \
  instances/paperInstances/TXT_10-25_4-5_U30/instance_30_15_4_47.txt
```

Shell chỉ điều phối. Mỗi JSON báo `language: C++` và tự đo toàn bộ đường chạy
C++; do đó không có Python trong đường đo thời gian. Chạy ablation paired bằng:

```sh
. experiments/configs/cpp/cardinality_ablation.env
```

Runner ghi `cardinality_encoding` vào cả JSON, tên run và `manifest.tsv`.
Tương tự, `implied_constraints` và `symmetry_breaking` được ghi vào cả ba vị trí.
`configuration_matrix.tsv` lưu ma trận encoding thực tế; `runs.csv` và
`configuration_summary.csv` là bảng UTF-8 có BOM, mở trực tiếp bằng Excel.

## Kiểm tra bắt buộc trước full experiment

```sh
EVALMAXSAT_BIN=/opt/evalmaxsat/EvalMaxSAT_bin \
PYTHONPATH=src/proposed python3 -m pytest -q
```

Python ở lệnh này là test harness, không phải implementation được benchmark.
Các test bắt buộc kiểm tra optimum 8 của weighted tiny instance, hai nghiệm
trade-off khác nhau của hai policy, hai mức epsilon, soft coverage, WCNF và
verifier C++.

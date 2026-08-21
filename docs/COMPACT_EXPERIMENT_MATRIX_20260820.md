# Ma trận thực nghiệm compact cho ICIIT 2027

> **Đã được cập nhật:** phần trạng thái dữ liệu và ma trận còn thiếu trong tài
> liệu này đã được thay thế bởi
> [`EXPERIMENT_SUPPLEMENT_MATRIX_20260822.md`](EXPERIMENT_SUPPLEMENT_MATRIX_20260822.md).
> Không dùng kết luận “0/732” bên dưới để quyết định chạy lại.

Cập nhật ngày 20/08/2026. File máy-đọc chuẩn là
`experiments/configs/reduced_campaign_manifest.json`. Tài liệu này phân biệt ba
loại dữ liệu: publication evidence cần chạy mới, dữ liệu lịch sử chỉ để chẩn
đoán, và functional smoke không được đưa vào bảng runtime.

## 1. Quyết định rút gọn

Giữ lại đúng bốn nhóm bằng chứng cần để bảo vệ các claim của bài:

1. factorial đầy đủ để tách hiệu ứng Totalizer, implied constraints và symmetry;
2. weighted so với LEX-COS trên benchmark gốc;
3. corrected-v2 để kiểm tra policy và thứ tự LEX-COS/LEX-OCS khi overtime bớt sparse;
4. Gurobi/CPLEX để kiểm tra exact-objective agreement khác paradigm.

Nguyên tắc giảm tải là tái sử dụng một run khi toàn bộ khóa thực nghiệm giống
nhau: instance hash, method, configuration, binary/solver, timeout và protocol.
Không gộp hai run chỉ vì tên instance giống nhau.

## 2. Ma trận cấu hình giữ lại

Ký hiệu: SN = sorting network, TOT = Totalizer, IC = implied constraints,
SB = exact slot-service symmetry breaking. Baseline `B` là SN/none/none;
reference `R` là TOT/both/slot-service.

| Cell | Cardinality | IC | SB | Vai trò |
|---|---|---|---|---|
| F1 = B | SN | none | none | audit baseline |
| F2 | SN | none | slot-service | direct SB contrast dưới SN |
| F3 | SN | both | none | direct IC contrast dưới SN |
| F4 | SN | both | slot-service | interaction cell |
| F5 | TOT | none | none | direct encoding contrast |
| F6 | TOT | none | slot-service | interaction cell |
| F7 | TOT | both | none | direct IC/SB contrast |
| F8 = R | TOT | both | slot-service | reference configuration |

Không giữ `user-slots`, `slot-capacity`, `both-plus`, `slots`, `services` hoặc
`all` trong publication factorial. Các treatment này hữu ích ở pilot nhưng làm
nổ số cell và làm khó quy kết tác động trong bài năm trang.

## 3. Ma trận measured cuối cùng

| ID | Dataset và phép lấy mẫu | Cấu hình/phương pháp | Runs | T | Bằng chứng |
|---|---|---|---:|---:|---|
| C1 | original, 16 lớp × seed 1--3 = 48 | 8 factorial cells × weighted | 384 | 300 s | RQ2, RQ3 và B--R end-to-end |
| C2 | original, 14 lớp × seed 1--3 = 42 | R × weighted/LEX-COS | 84 | 300 s | RQ1 với paired order và cùng timeout |
| C3 | corrected-v2 critical, 16 strata × seed 1001--1003 = 48 | R × weighted/LEX-COS/LEX-OCS | 144 | 300 s | policy validation và priority-order sensitivity dưới excess workload |
| C4 | original commercial subset, 2 lớp × seed 1--10 = 20 | Gurobi/CPLEX × weighted/LEX-COS | 80 | 300 s | cross-paradigm exactness |
| C5 | cùng commercial subset | EvalMaxSAT R × weighted/LEX-COS | 40 | 300 s | ghép nhóm cùng-budget với C4 |
| **Tổng** |  |  | **732** |  | **61,00 core-hour worst case** |

Hai lớp `30_15_4` và `40_25_5` bị loại khỏi C2 vì đã được xem trong quá
trình commercial development. Chúng chỉ xuất hiện trong C4--C5 với vai trò tập
validation được khai báo tường minh. Corrected-v2 chỉ giữ profile `critical` và
ba evaluation seeds đầu; không giữ relaxed/saturated hoặc thêm seeds vì không
cần cho claim hiện tại.

Ngoài measured matrix có 4 EvalMaxSAT LEX-COS scalability-calibration runs trên
development seeds 41--42 ở timeout 300 s và 18 commercial correctness-smoke
runs: 3 toy instances × 3 backends × weighted/LEX-COS, timeout 30 s. Chúng là
gate phần mềm/khả thi, không là publication runtime evidence. Measured campaign
chỉ bắt đầu nếu ít nhất 2/4 calibration rows đạt optimum.

Không tái sử dụng 42 weighted-R rows của C1 cho C2 dù các instance trùng nhau.
C2 xen kẽ weighted và LEX-COS trong cùng campaign block và lưu assignment cho
kiểm tra objective; thiết kế này kiểm soát execution order tốt hơn cho phép so
sánh policy. C5 cũng phải giữ cả hai MaxSAT policies để tạo đủ nhóm kiểm chứng
ba backend. Đây là các lượt lặp có mục đích, không phải duplication ngoài ý
muốn.

Timeout 300 s được khóa cho toàn bộ measured matrix. Nghiên cứu gốc dùng 1 giờ
và 16 GB cho mỗi execution, nhưng thời gian trung bình certified lớn nhất được
báo cáo là 158,34 s. Diagnostic lịch sử tại `results/comparison_pivot.csv` có
optimum ở 270,647 s, cho thấy mốc 120 s cũ sẽ censor một ca đã biết có thể chứng
nhận. Mốc 300 s không dùng để trộn runtime cũ vào bài; nó chỉ là căn cứ pilot
khai báo trước cho ngân sách compact. Timeout của lexicographic policy là cộng
dồn qua mọi stage; timeout rows được giữ trong solved count và PAR-2.

## 4. Kết quả hiện đang có và khả năng sử dụng

| Nguồn | Nội dung thực tế | Tình trạng | Được dùng thế nào |
|---|---|---|---|
| `results/` | tám weighted CSV lịch sử, tối đa 800 instances/cell | thiếu 54 records; không có raw JSON, logs, environment hoặc manifest đầy đủ | chỉ diagnostic và kiểm tra objective; không đưa runtime vào bài |
| `results_addition/main_8cfg_evalmaxsat` | 1.996/2.000 rows của 8-cell ablation | thiếu 4; khác frozen commit/sample/provenance | historical encoding diagnostic và timeout calibration |
| `results_addition/lex_8cfg_evalmaxsat` | 1.689/4.000 rows | coverage lệch/thiếu; policy cũ; solver bị loại | exploratory history only |
| `results_addition/epsilon_8cfg_evalmaxsat` | 1.848/10.000 rows | bị censor theo file order, commit/config không đồng nhất | không dùng làm evidence |
| `results_addition/commercial_*` | 284/400 và 3.149/3.200 raw JSON ở hai campaign | development classes; policy/protocol cũ; provenance không đạt publication | solver/parser diagnostics only |
| `experiments/results/pilot_*` | các pilot encoding/full-coverage/multiobjective | chạy để chọn thiết kế, không phải sample/protocol publication | hỗ trợ quyết định thiết kế, không đi vào bảng chính |
| `experiments/results/corrected_v2_*` | 10 + 10 functional/pilot runs | validation đầy đủ cho phạm vi toy/pilot nhưng sai cỡ mẫu/protocol | kiểm tra implementation only |
| `experiments/results/weight_sensitivity_pilot` | 16 functional runs | cỡ mẫu quá nhỏ và branch đã hoãn | không đưa claim weight sensitivity |
| `experiments/results/uncertainty_functional_*` | 2 nominal + 4 scenario runs | functional test, chưa phải uncertainty study | không đưa claim robustness |

Kết luận audit: hiện có **0/732 publication-eligible measured rows** theo
manifest compact. Các file cũ vẫn có giá trị phát hiện lỗi và kiểm tra xu hướng,
nhưng không thể trộn vào ma trận mới vì khác solver, commit, timeout, policy,
sampling và provenance. Việc chạy lại 732 rows là cần thiết để có một dataset
đồng nhất; không phải vì toàn bộ kết quả cũ vô giá trị.

## 5. Những cấu hình bị loại

| Nhánh cũ | Quyết định | Lý do |
|---|---|---|
| factorial 160 instances | còn 48 | vẫn giữ đủ 16 lớp và 3 paired seeds/lớp; mỗi direct contrast có 48 block trước timeout filtering |
| weighted B/R trên 800 instances | bỏ campaign riêng | B/R 80 pairs đã có trong factorial; 800 pairs tốn 133,33 giờ mà không thêm factor attribution |
| lex scalability 2 configs × 2 methods | bỏ | trùng mục tiêu RQ1; chỉ giữ R là cấu hình policy reference |
| LEX-COS trên B và R | chỉ giữ R | paper không claim encoding-policy interaction; giảm một nửa policy runs |
| LEX-OCS trên benchmark gốc | chuyển sang corrected-v2 | original có OT quá sparse; corrected critical trả lời trực tiếp sensitivity của thứ tự ưu tiên |
| corrected-v2 160 instances | 48 instances × 3 policies | giữ đủ 16 strata, ba evaluation seeds/stratum và ghép cả priority sensitivity trong cùng block |
| commercial 100 instances | còn 20 | validation agreement, không ước lượng runtime population; hai lớp × mười seeds đủ để phát hiện mismatch hệ thống |
| epsilon/Pareto, weight sensitivity | hoãn toàn bộ measured branch | không phải contribution cốt lõi; pilot không tạo bằng chứng đủ mạnh |
| routing, uncertainty | loại khỏi campaign | model/claim hiện tại chưa hỗ trợ nghiên cứu thực nghiệm đầy đủ |

Ba seeds mỗi stratum là mức tối thiểu để vừa giữ toàn bộ 16 strata vừa có lặp
trong stratum. Phân tích chính vẫn paired theo instance ($n=48$ cho mỗi direct
factor contrast), còn kết quả theo stratum chỉ mang tính mô tả; không ước lượng
effect riêng cho từng stratum từ ba quan sát.

So với campaign 1.270 runs ngay trước lần rà soát này, ma trận mới giảm 538 runs
(42,36%) và giảm worst-case từ 105,83 xuống 61,00 core-hour. Với một worker tuần
tự, worst case là 2,54 ngày; thời gian thực tế thường thấp hơn do nhiều run kết
thúc trước timeout.

## 6. Thứ tự chạy và điều kiện dừng

1. `preflight`: build, tests, benchmark verification, config/task-count checks.
2. `commercial-preflight`: license/backend check và 18 correctness-smoke runs.
3. `solver-calibration`: chạy 4 LEX-COS development rows; dừng nếu dưới 2 optimum.
4. `screen`: chạy C1. Đây vừa là primary evidence vừa là hard gate; dừng nếu có
   technical error, unverified optimum, weighted-objective mismatch hoặc RSS
   vượt 12 GB.
5. `original-primary`: chạy C2; weighted và LEX-COS dùng cùng timeout 300 s.
6. `corrected-primary`: chạy C3 theo block ba policy trên từng instance.
7. `commercial`: chạy C4--C5 với cùng timeout 300 s cho hai MaxSAT policies.
8. `analyze`, `package`, freeze manuscript.

Lệnh một lần trên VM sau khi đã tạo clean publication commit/tag:

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

Nếu cần quan sát từng phase, dùng `experiments/gcp_prepare_and_run.sh` theo thứ
tự trên. Runner hỗ trợ resume theo run ID; không rút thêm instance hậu nghiệm
sau khi đã xem kết quả. Chỉ resume các row lỗi kỹ thuật hoặc chưa hoàn tất.

## 7. Điều kiện đủ để đưa vào bản thảo

- đúng 732 measured rows, không duplicate hoặc unexpected run ID;
- mọi solver-reported `OPTIMUM` qua independent verifier;
- đủ 48 instances ở mỗi factorial cell và đủ 12 direct factor contrasts;
- đủ 42 weighted/LEX-COS R pairs và 48 corrected LEX-COS/LEX-OCS R pairs trước khi lọc
  jointly-optimum cho objective deltas;
- đủ 48 corrected-v2 rows cho mỗi policy;
- đủ 20 three-backend groups cho mỗi policy, báo agreement chỉ khi cả ba backend
  chứng minh optimum;
- bảng, quantitative prose và provenance được sinh từ frozen raw data;
- development, pilot và historical rows không được nhập vào publication tables.

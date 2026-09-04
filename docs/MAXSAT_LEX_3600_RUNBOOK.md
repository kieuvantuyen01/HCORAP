# EvalMaxSAT LEX-COS ở giới hạn 3.600 giây

Tài liệu này mô tả chiến dịch bổ sung nhằm đánh giá lại khả năng giải chính
xác của EvalMaxSAT cho mục tiêu từ điển LEX-COS. Đây là một chiến dịch mới,
không thay đổi hoặc ghi đè các kết quả 300 giây đã có.

## 1. Những điểm đã sửa

Driver C++ và pipeline thực nghiệm đã được bổ sung bốn khả năng cần thiết:

1. truyền ngân sách còn lại của từng lần gọi solver qua tham số `--TCT` của
   EvalMaxSAT;
2. gửi `SIGTERM`, chờ tối đa 5 giây và thu nghiệm tốt nhất mà EvalMaxSAT xuất
   ra khi hết thời gian;
3. kiểm tra độc lập nghiệm này rồi ghi trạng thái `TIMEOUT_FEASIBLE`, thay vì
   bỏ toàn bộ kết quả của một lần chạy chưa chứng minh tối ưu;
4. hỗ trợ hai cách giải LEX-COS mới để so sánh với cách ba giai đoạn hiện tại:
   một cận tương thích ở giai đoạn cuối và một mục tiêu từ điển một lần gọi.

Mỗi JSON schema version 3 ghi rõ trạng thái và incumbent của từng giai đoạn,
số tiêu chí đã được chứng minh tối ưu, thời lượng `--TCT`, thời gian chờ khi
dừng solver và nghiệm đã được verifier chấp nhận.

## 2. Ma trận cố định

Tất cả MaxSAT runs dùng EvalMaxSAT đã khóa SHA-256, Totalizer, không thêm
implied constraints, không symmetry breaking, một worker, một CPU được pin và
một ngân sách cộng dồn 3.600 giây cho mỗi instance-policy.

| Pha | Tập dữ liệu | Cấu hình EvalMaxSAT | Số runs |
|---|---:|---:|---:|
| Pilot | 16 corrected-v2 calibration instances, seed 1 | baseline ba giai đoạn, cận giai đoạn cuối, một lần gọi | 48 |
| Pilot reference | cùng 16 instances | Gurobi MIP-E LEX-COS | 16 |
| Confirmation nếu pilot GO | 48 corrected-v2 evaluation instances, seeds 1001--1003 | baseline và đúng một candidate | 96 |
| Confirmation reference | cùng 48 instances | Gurobi MIP-E LEX-COS | 48 |

Ba biến thể MaxSAT là:

- `staged-aligned`: CONT, OT rồi SIM, chia sẻ một ngân sách 3.600 giây;
- `staged-incumbent-bound`: giống baseline nhưng dùng độ tương thích của nghiệm
  sau giai đoạn OT làm cận dưới hợp lệ cho giai đoạn SIM;
- `single-call-dominance`: gộp đúng thứ tự CONT, OT, SIM vào một mục tiêu bằng
  các hệ số trội được suy ra từ cận trên của từng tiêu chí.

Hai candidate không được chạy đồng thời trên tập confirmation. Analyzer chọn
tối đa một candidate khi pilot cho thấy cải thiện về số optimum, PAR-2, số
nghiệm khớp hai tiêu chí ưu tiên với Gurobi, hoặc số vector mục tiêu khớp hoàn
toàn. Candidate cũng không được mất một instance mà baseline đã đạt cùng tiêu
chí chất lượng. Ngưỡng tăng tối thiểu là 10% số instances và không dưới hai
instances. Nếu không có cải thiện, pilot trả `STOP` và không chạy confirmation.

Ngân sách MaxSAT xấu nhất là 48 core-hours nếu pilot dừng hoặc 144 core-hours
nếu một candidate được chọn và chạy confirmation. Đây là cận trên;
run kết thúc ngay khi solver chứng minh tối ưu. Gurobi được chạy lại để tạo
đối chứng có cùng tập instance và provenance, nhưng dự kiến kết thúc sớm hơn
nhiều so với giới hạn.

## 3. Chuẩn bị trên GCP

Máy đo là Linux x86-64, loại `c4-highcpu-8`, RAM 16 GB. Cài Python packages,
Gurobi Optimizer và license theo `docs/COMMERCIAL_SOLVERS.md`. Đặt đúng
EvalMaxSAT Linux đã dùng trong nghiên cứu trước, không dùng executable macOS ở
root của repository.

```bash
export EVALMAXSAT_BIN=/opt/evalmaxsat/EvalMaxSAT_bin
export GUROBI_HOME=/absolute/path/to/gurobi/platform
export HCORAP_EXPECTED_COMMIT=$(git rev-parse HEAD)
```

Script từ chối chạy nếu hash EvalMaxSAT khác hash trong
`experiments/configs/reduced_campaign_manifest.json`. Với measured runs,
script cũng từ chối dirty worktree hoặc revision khác
`HCORAP_EXPECTED_COMMIT`.

Chạy kiểm tra trước:

```bash
./experiments/run_maxsat_lex_3600.sh preflight
```

Preflight build lại driver MaxSAT, build Gurobi backend, chạy toàn bộ test,
dry-run sáu cấu hình và giải hai smoke instances bằng EvalMaxSAT thật.

## 4. Chạy toàn bộ chiến dịch

Nên chạy trong `tmux` hoặc một dịch vụ giữ tiến trình của VM. Sau khi preflight
thành công:

```bash
export CONFIRM_MAXSAT_LEX_3600=YES
./experiments/run_maxsat_lex_3600.sh all
```

Có thể tách hai pha:

```bash
export CONFIRM_MAXSAT_LEX_3600=YES
./experiments/run_maxsat_lex_3600.sh pilot
./experiments/run_maxsat_lex_3600.sh confirm
```

Runner ghi một manifest record ngay sau mỗi run và hỗ trợ `--resume`, vì vậy
có thể chạy lại đúng lệnh sau khi VM bị gián đoạn. Không xóa hoặc sửa thư mục
kết quả giữa hai lần chạy. Nếu có persistent disk ngoài worktree, đặt thêm:

```bash
export HCORAP_BACKUP_DIR=/mnt/hcorap-backup
```

Script sẽ `rsync` checkpoint sau mỗi campaign hoàn tất.

## 5. Kết quả đầu ra

Pilot tạo các thư mục:

- `experiments/results/gcp_maxsat_lex_3600_pilot/`;
- `experiments/results/gcp_maxsat_lex_3600_pilot_gurobi/`;
- `experiments/results/gcp_maxsat_lex_3600_pilot_analysis/`.

Confirmation tạo Gurobi reference, đúng một thư mục MaxSAT được chọn và
`experiments/results/gcp_maxsat_lex_3600_confirmation_analysis/`.

Mỗi analysis directory chứa bảng ghép cặp theo instance, bảng tổng hợp và
`maxsat_lex_3600_decision.json`. Analyzer đánh dấu `INVALID` nếu thiếu run,
trùng instance, sai cấu hình, sai timeout, thiếu nghiệm đã kiểm tra hoặc một
OPTIMUM của MaxSAT không khớp vector mục tiêu tối ưu của Gurobi.

Chỉ cập nhật bảng và lập luận trong bản thảo sau khi confirmation hợp lệ. Nếu
candidate thắng pilot nhưng không lặp lại cải thiện trên confirmation, giữ
kết quả như một kết quả âm và không tuyên bố candidate tốt hơn baseline.

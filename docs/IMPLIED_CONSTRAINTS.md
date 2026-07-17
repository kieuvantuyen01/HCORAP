# Implied-constraint configurations for HCORAP

Các cấu hình trong tài liệu này chỉ thêm ràng buộc suy ra từ hard model hoặc
thay một encoding bằng encoding tương đương. Chúng không thay đổi objective hay
tập lịch khả thi.

## Cấu hình

| CLI value | Nội dung |
|---|---|
| `none` | Baseline, không tạo biến/ràng buộc implied mới |
| `user-slots` | Cardinality của các time slot được dùng bởi từng user |
| `slot-capacity` | Bound công suất mỗi slot từ maximum bipartite matching |
| `both` | `user-slots` và `slot-capacity` |
| `both-plus` | `both`, projected service assignment, effective workload cap và service-slot clustering |

Chọn cấu hình bằng `--implied-constraints VALUE`. Giá trị mặc định là `none`.
Trục này độc lập với `--cardinality-encoding sorting-network|totalizer`.

## Biến chiếu theo service và time slot

Các cấu hình khác `none` tạo:

```text
serviceSlot[s,h] <-> OR_a x[a,s,h]
```

Biến này cho phép các implied constraints làm việc trên số service được xếp,
không đếm lặp các lựa chọn agent của cùng một service.

## User occupied-slot cardinality

Với mỗi user `u`:

```text
userSlot[u,h] <-> OR_{s in SU(u)} serviceSlot[s,h]
```

Trong full coverage:

```text
sum_h userSlot[u,h] = |SU(u)|
```

Trong soft coverage, vế phải không được cố định:

```text
sum_h userSlot[u,h] = sum_{s in SU(u)} performed[s]
```

Hai tổng ở soft coverage được mã hóa bằng hai unary counters chính xác và
channel từng threshold. Vì vậy cấu hình vẫn đúng khi một phần service không có
candidate.

## Time-slot matching capacity

Tại mỗi slot `h`, preprocessing dựng đồ thị hai phía agent--user. Có cạnh
`(a,u)` nếu tồn tại service của user `u` mà agent `a` có thể thực hiện tại `h`.
Mọi lịch tại slot đó là một matching do hard model đã có AMO cho agent và user.
Nếu `K_h` là kích thước maximum matching, ta thêm:

```text
sum_s serviceSlot[s,h] <= K_h
```

Nếu partition `SU` không hợp lệ, implementation bỏ qua bound matching thay vì
thêm một cut có thể không an toàn.

## Các strengthening trong `both-plus`

- `performed[s] <-> OR_a y[a,s]` và AMO trực tiếp trên các `y[a,s]` khả thi;
- user-time AMO dùng `serviceSlot[s,h]` thay cho toàn bộ `x[a,s,h]`;
- workload counter bỏ các `y` không có time slot khả thi;
- workload cap dùng
  `min(HN+HE, number of candidate services, number of usable slots)`.

## Quy tắc đánh giá

Không gộp các cấu hình trước khi ablation. Chạy lần lượt `none`, `user-slots`,
`slot-capacity`, `both`, `both-plus` với cùng instance order, solver, timeout và
cardinality encoding. Mỗi JSON và dòng manifest lưu cả hai trục cấu hình.

Các regression bắt buộc:

- cùng SAT/UNSAT và optimum với `none` trên instance nhỏ;
- verifier chấp nhận mọi nghiệm tối ưu;
- user-slot equality đúng cho cả full và partial coverage;
- `none` giữ nguyên WCNF baseline;
- báo riêng variables, hard clauses, encode time, solve time và PAR-2.

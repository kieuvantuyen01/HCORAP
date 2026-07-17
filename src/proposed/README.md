# Proposed HCORAP implementations

Thư mục này tách phần mở rộng nghiên cứu khỏi encoder C++ gốc dùng để audit:

```text
proposed/
├── cpp/encodings/       # implementation C++ nằm trên đường benchmark
│   ├── CardinalityNetwork.*
│   ├── ImpliedConstraints.*
│   ├── SymmetryBreaking.*
│   └── HCORAPMultiObjectiveEncoding.*
└── hcorap/              # oracle, generator và verifier Python
```

`HCORAPMultiObjectiveEncoding` chứa mô hình dùng chung. Các thay đổi có thể làm
ablation được đặt sau một strategy/option nhỏ thay vì sao chép toàn bộ encoder.
Hiện có hai cấu hình cardinality:

| Tên CLI | Vai trò | Implementation |
|---|---|---|
| `sorting-network` | baseline, mặc định | sorting network hai chiều của SMT API |
| `totalizer` | biến thể thử nghiệm | Totalizer hai chiều, giữ output threshold chính xác |

Implied constraints là trục cấu hình thứ hai:

| Tên CLI | Nội dung |
|---|---|
| `none` | baseline, mặc định |
| `user-slots` | số slot được dùng của user khớp số service được phục vụ |
| `slot-capacity` | bound theo maximum matching tại mỗi slot |
| `both` | kết hợp hai implied constraints |
| `both-plus` | `both` cùng projection, tighter cap và slot clustering |

Symmetry breaking là trục thứ ba:

| Tên CLI | Nội dung |
|---|---|
| `none` | baseline, mặc định |
| `slots` | order các time slot tương đương |
| `services` | order các service tương đương |
| `slot-service` | kết hợp hai loại trên |
| `all` | thêm agent symmetry nếu detector tìm thấy |

Chọn bằng `--cardinality-encoding`, `--implied-constraints` và
`--symmetry-breaking`. JSON và manifest lưu đủ ba tên. Chi tiết tính đúng và
protocol ablation nằm trong `docs/IMPLIED_CONSTRAINTS.md` và
`docs/SYMMETRY_BREAKING.md`. Khi thêm cải tiến mới, cần giữ nguyên baseline mặc
định, thêm metadata và regression test feasibility/optimum trước campaign.

Python trong `hcorap/` không được dùng để báo runtime so sánh với C++; nó chỉ là
implementation độc lập để kiểm thử chéo.

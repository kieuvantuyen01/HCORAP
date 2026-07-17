# Symmetry-breaking configurations for HCORAP

Symmetry breaking giữ lại một đại diện canonical trong mỗi orbit nghiệm. Khác
với implied constraints, nó loại các lịch có nhãn khác nhau nhưng tương đương;
SAT/UNSAT và optimum phải được bảo toàn, nhưng tập assignment có nhãn không còn
giữ nguyên.

Đặt:

```text
E(a,s,h) = (r(a,s) > 0) and TSA(a,h) and TSS(s,h)
```

## Cấu hình

| CLI value | Nội dung |
|---|---|
| `none` | Baseline, không thêm symmetry-breaking clause |
| `slots` | Value precedence giữa các time slot tương đương |
| `services` | Value precedence giữa các service tương đương |
| `slot-service` | Kết hợp `slots` và `services` |
| `all` | `slot-service` và symmetry giữa các agent đồng nhất |

Chọn bằng `--symmetry-breaking VALUE`. Giá trị mặc định là `none`.

## Time-slot symmetry

Hai slot thuộc cùng lớp khi vector `E(a,s,h)` giống nhau với mọi agent và
service. Các slot không chứa candidate nào được bỏ qua vì chúng không tạo ra
assignment khác nhau. Với hai slot liên tiếp `h1 < h2` trong một lớp, encoder
order hai vector `serviceSlot[:,h]` bằng value precedence. Vì một service chỉ
có thể xuất hiện ở một slot, constraint này chọn duy nhất thứ tự theo service id
nhỏ nhất và đẩy slot rỗng về cuối.

## Service symmetry

Hai service chỉ được gộp khi chúng:

- thuộc cùng user trong `SU`;
- thuộc cùng sequence trong `SEQ`;
- có cùng reward đối với mọi agent;
- có cùng vector candidate `E(a,s,h)`.

Assignment `(agent, slot)` được order theo chỉ số `agent * TS + slot`. Trong
soft coverage, precedence cũng chọn một prefix service canonical nếu chỉ một
phần của lớp được phục vụ.

## Agent symmetry

Hai agent chỉ tương đương khi `HN`, `HE`, toàn bộ reward và candidate vector
giống nhau. Cấu hình `all` order các vector `y[a,:]`. Detector không thêm clause
nếu instance không có lớp agent tương đương.

## Khảo sát corpus

Trên 800 paper instances, detector tìm thấy symmetry có candidate thực sự ở:

| Loại | Instance | Tỷ lệ | Lớp lớn nhất |
|---|---:|---:|---:|
| Slot | 553 | 69.1% | 3 |
| Service | 340 | 42.5% | 5 |
| Agent | 0 | 0.0% | 0 |

Các điều kiện là exact, không gộp các trường hợp chỉ có cùng số candidate hoặc
cùng qualification nhưng khác reward.

## Regression bắt buộc

- `none` giữ nguyên WCNF baseline;
- năm cấu hình giữ nguyên SAT/UNSAT và optimum;
- nghiệm trả về qua verifier C++;
- kiểm tra cả full coverage và soft coverage;
- kiểm tra đủ ma trận cardinality, implied constraints và symmetry breaking;
- ghi riêng variables, clauses, encode time, solve time và PAR-2.

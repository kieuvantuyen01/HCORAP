# Rà soát kết quả research-depth pilot — 10/09/2026

## Kết luận quyết định

Pilot đạt các điều kiện tính toán để mở rộng ba campaign thương mại lên 48
instance: policy diagnostics, độ nhạy trọng số và độ nhạy năng lực. Không cần
chạy lại pilot. Thử nghiệm MaxSAT đã đủ để làm một ablation kỹ thuật ngắn; chỉ
cần mở rộng nếu bài báo dành một câu hỏi nghiên cứu riêng cho hiệu năng mã hóa.

Các kết luận dưới đây vẫn là kết quả pilot trên 16 instance được chọn có chủ
đích. Chúng chưa thay thế kết quả full và chưa nên được trình bày như ước lượng
cho toàn bộ HCORAP-LC.

Có một giới hạn về provenance cần ghi nhận. Mọi `environment.json` ghi commit
`78d5a650a747824eb944352494b3d6ec3716e797`, với worktree sạch, nhưng commit này
chưa tồn tại trong clone hiện tại hoặc trên `origin`. Cấu hình pilot hiện tại
khớp đúng tất cả SHA-256 đã ghi trong kết quả. Theo quyết định ngày 10/09/2026,
khác biệt commit được xem là không ảnh hưởng đến kết quả và không chặn lượt
full. Pilot chỉ dùng để chọn thiết kế; các con số đưa vào bài sẽ lấy từ full run
ở một commit có thể truy xuất trên GitHub.

## Tính toàn vẹn của dữ liệu

| Campaign | Số run | Trạng thái | Kiểm định chính |
|---|---:|---|---|
| Diagnostics, Gurobi | 160 | 160 OPTIMUM | 16/16 mặt tối ưu Weighted đầy đủ |
| Diagnostics, CPLEX | 160 | 160 OPTIMUM | Khớp Gurobi ở 160/160 ô |
| Weight sensitivity, Gurobi | 144 | 144 OPTIMUM | 0 lỗi; 144 nghiệm được xác minh |
| Load sensitivity, Gurobi | 432 | 432 OPTIMUM | 432 lịch được xác minh độc lập |
| Zero-continuity, EvalMaxSAT | 64 | 60 OPTIMUM, 4 UNSATISFIABLE | 30 cặp global/local khả thi khớp vector |

Tổng cộng 960/960 tác vụ kết thúc, không có run thiếu, run thừa, lỗi kiểm định
hay hard timeout. Bốn trạng thái UNSATISFIABLE cùng thuộc instance Original
`instance_40_10_5_1.txt`, lặp lại ở SN/TOT và global/local; chúng không phải lỗi
solver.

Gurobi 11.0.3 và CPLEX với `solver_version=22020000` cùng chạy một luồng trên
Linux x86-64, affinity CPU `[0]`, bằng cùng binary commercial. Binary và
EvalMaxSAT đều có SHA-256 trong metadata. Năm SHA-256 cấu hình trong metadata
khớp với các file cấu hình hiện tại.

## Bằng chứng chính về thứ tự CONT → OT → SIM

### Khác biệt với Weighted không phải chỉ do tie-breaking

Trên cả 16 instance, bốn phép khảo sát min/max đã mô tả đầy đủ khoảng CONT và OT
trên mặt nghiệm tối ưu của Weighted. Kết quả cho thấy:

- 10/16 instance có **mọi** nghiệm Weighted tối ưu đều có CONT lớn hơn nghiệm
  COS. Đây là chứng nhận rằng Weighted không thể chọn một nghiệm đồng tối ưu để
  khôi phục mức liên tục của COS.
- 15/16 instance có OT nhỏ nhất trên mặt tối ưu Weighted vẫn lớn hơn OT của
  COS; 9/16 đồng thời chịu cả hai mất mát CONT và OT.
- Mặt tối ưu Weighted có nhiều mức CONT ở 15/16 instance và nhiều mức OT ở
  16/16. Độ rộng trung vị của khoảng CONT là 1, còn OT là 7. Vì vậy chỉ báo cáo
  một lịch Weighted có thể phụ thuộc mạnh vào tie-breaking, nhưng kết luận
  “unavoidable loss” ở trên không phụ thuộc vào lịch được trả về.

Đây là kết quả phù hợp nhất để giải thích đóng góp của thứ tự CONT → OT → SIM:
nó bảo đảm ưu tiên chăm sóc theo tầng và tránh việc một tổng có trọng số bù trừ
CONT hoặc OT để lấy SIM. Điểm khác biệt nên được trình bày bằng bảo đảm quyết
định và bằng chứng trên toàn bộ mặt tối ưu, thay vì nói rằng lexicographic là
mới tự thân.

### Giá của việc nới CONT có thể đọc trực tiếp

Đường ngân sách CONT với `k = 0, 1, 2` được chứng minh tối ưu ở cả 16 instance.
`k = 0` khôi phục đúng vector COS. Khi tăng từ 0 lên 1, ba instance giảm được
tổng cộng ba giờ OT; SIM thay đổi trung bình `+4.19`, trung vị `+6`, trong khoảng
`[-3, 8]`. Giá trị `-3` xảy ra ở một instance được giảm OT, vì OT đứng trước SIM.

Từ `k = 1` lên `k = 2`, không instance nào giảm thêm OT; SIM tăng ở 16/16
instance, trung bình `+3.19`. Từ `k = 0` lên `k = 2`, thay đổi trung bình là
`CONT +2`, `OT -0.19`, `SIM +7.38`.

Kết quả này tạo ra diễn giải thực tế rõ hơn so với một bảng so sánh ba hàm mục
tiêu: nhà lập lịch có thể thấy chính xác một đơn vị nới continuity mua được bao
nhiêu OT hoặc SIM, đồng thời vẫn giữ CONT là ràng buộc có kiểm soát.

## Độ nhạy của Weighted

Mỗi instance có từ 4 đến 8 vector mục tiêu khác nhau trong lưới chín cặp trọng
số. Ngay cả khi giữ tỷ lệ `wc:wo = 1:1`, tăng đồng thời hai hệ số vẫn làm thay
đổi nghiệm vì hệ số SIM được giữ bằng 1.

| `(wc, wo)` | CONT trung bình | OT trung bình | SIM trung bình | Trùng vector COS |
|---|---:|---:|---:|---:|
| `(1, 1)` | 2.44 | 12.88 | 590.13 | 0/16 |
| `(4, 4)` | 1.13 | 0.44 | 563.50 | 3/16 |
| `(8, 8)` | 0.50 | 0.06 | 557.94 | 14/16 |

Chuyển `(1,1) → (4,4)` làm đổi vector ở 16/16 instance; `(4,4) → (8,8)` đổi
12/16; `(1,1) → (8,8)` đổi 16/16. Không phát hiện vi phạm tính đơn điệu đối với
CONT khi tăng `wc` hoặc OT khi tăng `wo`.

Thông điệp phù hợp là Weighted cần calibration và có thể thay đổi theo thang
trọng số, còn thứ tự CONT → OT → SIM thể hiện trực tiếp ưu tiên của bài toán.
Không gọi các cặp cùng tỷ lệ là hàm mục tiêu tương đương.

## Độ nhạy theo năng lực

Campaign gồm 27 ô mô tả, tương ứng chín cấu hình năng lực và ba cặp policy,
trên 16 instance lồng trong ba họ cha. COS và OCS chỉ khác nhau ở ô gốc
`rho = 0.85`, `normal_fraction = 0.85`, đúng ba instance xung đột đã biết. Tám ô
năng lực còn lại không có xung đột COS–OCS trong pilot. Đây là dấu hiệu rằng
xung đột giữa hai thứ tự lexicographic phụ thuộc vùng năng lực, không phải một
hiện tượng phổ quát.

Weighted khác COS ở 10–16/16 instance tùy ô. Ba lát cắt minh họa:

| `rho`, normal fraction | Policy | CONT | OT | SIM |
|---|---|---:|---:|---:|
| 0.55, 1.00 | COS | 0.13 | 0.00 | 599.94 |
|  | Weighted | 2.38 | 0.00 | 606.75 |
| 0.85, 0.85 | COS | 0.19 | 0.19 | 555.31 |
|  | OCS | 0.38 | 0.00 | 555.63 |
|  | Weighted | 2.44 | 12.88 | 590.13 |
| 0.98, 0.85 | COS | 0.31 | 20.25 | 554.06 |
|  | Weighted | 2.75 | 20.56 | 562.94 |

`rho` thực tế lần lượt nằm trong `[0.5479, 0.5498]`, `[0.8451, 0.8475]` và
`[0.9740, 0.9756]`. Khi viết bài phải dùng các giá trị thực tế hoặc nói rõ đây
là mức đích. Các biến thể được xây để luôn khả thi, nên phép thử này không chứng
minh hành vi tại biên chuyển sang bất khả thi. Các instance trong cùng họ chia
sẻ dữ liệu gốc; không coi 432 cặp là 432 mẫu độc lập.

## Ablation zero-continuity

Trong 30 cặp khả thi, local AMO giữ nguyên vector tối ưu so với global bound.
Local nhanh hơn ở 13/15 cặp SN và 14/15 cặp TOT. Trung vị tỷ số thời gian
global/local là 1.151 cho SN và 1.154 cho TOT.

Mức giảm kích thước công thức lớn hơn mức tăng tốc đầu-cuối. Ở stage 2 và 3,
tỷ số trung vị số hard clauses global/local lần lượt khoảng `3.45×` cho SN và
`19.4×` cho TOT. Trung vị tỷ số solve time tại hai stage là `8.55×/13.56×` cho
SN và `12.88×/18.14×` cho TOT. Stage 1 không đổi và chiếm phần đáng kể, nên
end-to-end chỉ cải thiện khoảng 15%.

Kết quả này đủ cho một ablation giải thích hiệu quả mô hình hóa. Nó chưa đủ mạnh
để trở thành claim thuật toán trung tâm hoặc claim tăng tốc lớn.

## Cách dùng pilot để chạy full

1. Chuyển VM sang phiên bản mới nhất đã có trên GitHub và cố định commit cho
   toàn bộ full run:

   ```bash
   git fetch origin
   git switch --detach origin/main
   export HCORAP_EXPECTED_COMMIT=$(git rev-parse HEAD)
   ```

2. Không chạy lại pilot. Chạy full diagnostics, weights và load. Ba campaign
   full có 2.208 run; từ thời gian pilot, phần commercial dự kiến chỉ cần khoảng
   17 phút solver-time tuần tự, chưa kể build và I/O:

   ```bash
   experiments/run_research_depth_gcp.sh preflight
   export CONFIRM_RESEARCH_DEPTH_FULL=YES
   nohup experiments/run_research_depth_gcp.sh full \
     > research-depth-full.log 2>&1 &
   ```

3. Không mở rộng MaxSAT trong lượt full mặc định. Nếu cần claim hiệu năng mã
   hóa, thêm seed 2–3 và dùng instance family làm đơn vị phân tích; pilot hiện
   tại dự báo khoảng 3–4 giờ solver-time cho 128 run bổ sung.

4. Sau full, dùng 48 instance làm bảng kết quả chính. Pilot đã chọn có chủ đích
   và chứa ba conflict đã biết, nên không dùng tỷ lệ 10/16, 15/16 hoặc 3/16 làm
   tỷ lệ đại diện cuối cùng.

Không cần chạy thêm CPLEX ở full: audit hiện tại đã phủ 16 lớp kích thước và
khớp Gurobi ở 160/160 ô. Không cần lặp nhiều solver seed vì các run dùng một
luồng, gap bằng 0 và chỉ lấy kết quả đã chứng minh tối ưu; seed lặp sẽ đo biến
động runtime chứ không tăng bằng chứng về vector mục tiêu. Chỉ cân nhắc một sweep
năng lực mịn quanh `rho=0.85` sau khi xem full, nếu mục tiêu của bài là xác định
chính xác vùng chuyển tiếp COS--OCS.

## Câu chữ có thể dùng sau khi có full result

Đoạn sau giữ claim ở mức phù hợp và không dựa vào việc lexicographic optimization
tự thân là mới:

> We model the service priorities as a lexicographic sequence that first
> protects continuity of care, then controls overtime, and finally improves
> assignment similarity. This order makes the operational preference explicit
> and removes the need to calibrate exchange rates among criteria. Our
> optimal-face analysis further distinguishes a genuine consequence of the
> weighted objective from solver tie-breaking: for [X] of the 48 instances,
> every weighted-optimal solution has a larger continuity penalty than the COS
> solution. The continuity-budget experiment then quantifies the value of
> relaxing this priority by one or two units.

Gurobi–CPLEX nên được mô tả như kiểm tra độc lập độ đúng và độ ổn định của kết
quả. Chưa nên claim đây là nghiên cứu home-care đầu tiên dùng cả hai solver nếu
chưa có một rà soát tài liệu có hệ thống chứng minh điều đó.

# Rà soát kết quả research-depth full — 11/09/2026

## Kết luận

Ba campaign full đã hoàn thành và vượt toàn bộ cổng kiểm định. Không cần chạy
thêm solver cho phiên bản nghiên cứu hiện tại. Kết quả đủ để thay các nhận định
từ pilot bằng bằng chứng trên toàn bộ 48 instance HCORAP-LC và đưa vào bản thảo.

Một capacity sweep mịn quanh `rho=0.85` chỉ là phần mở rộng tùy chọn nếu muốn
nghiên cứu riêng biên chuyển tiếp giữa COS và LEX-OT. Nó không cần thiết cho
claim chính về khác biệt giữa CONT → OT → SIM và Weighted.

## Kiểm định dữ liệu

| Campaign | Kết quả | Kiểm định |
|---|---:|---|
| Policy diagnostics | 480/480 OPTIMUM | 48/48 mặt tối ưu Weighted đầy đủ |
| Weight sensitivity | 432/432 OPTIMUM | 0 lỗi, 432 nghiệm được xác minh |
| Load sensitivity | 1.296/1.296 OPTIMUM | 1.296 lịch được xác minh độc lập |

Tổng cộng 2.208/2.208 run hoàn tất, không có timeout, run thiếu, run thừa, lỗi
JSON, sai vector mục tiêu hoặc lỗi kiểm tra lịch. Ba campaign dùng cùng:

- commit `8b35c809daf37073c458307deab96ac2b55dc433`, đã có trên GitHub;
- worktree sạch;
- binary SHA-256
  `7234074473777532ffd91bb1046434a1feda8f36d1eea1b8ee306a0adbdb6fc7`;
- Gurobi 11.0.3, một solver thread, CPU affinity `[0]`;
- Linux x86-64, 8 logical CPUs.

Các SHA-256 cấu hình trong metadata khớp với ba file cấu hình hiện tại. Tổng
solver-time là khoảng 19,6 phút; run lâu nhất là 9,45 giây. Toàn bộ các ô full
trùng pilot cũng được đối chiếu: 16 mặt tối ưu, 48 đường budget, 144 cấu hình
trọng số và 432 cặp load đều khớp, không có bất đồng.

## Kết quả làm mạnh claim chính

### Khác biệt với Weighted là bắt buộc

Bốn phép khảo sát chính xác trên mỗi instance tìm min/max CONT và OT trong toàn
bộ tập nghiệm đạt điểm Weighted tối ưu. Trên 48 instance:

- 41 instance có CONT nhỏ nhất trên mặt Weighted vẫn lớn hơn CONT của COS;
- 43 instance có OT nhỏ nhất trên mặt Weighted vẫn lớn hơn OT của COS;
- 36 instance chịu đồng thời cả hai mất mát;
- 46 mặt tối ưu chứa nhiều mức CONT và cả 48 chứa nhiều mức OT.

Độ rộng trung vị của mặt tối ưu là 3 đơn vị CONT và 6,5 đơn vị OT. Vì vậy nghiệm
Weighted cụ thể có thể thay đổi theo tie-breaking, nhưng trong 41 và 43 trường
hợp nêu trên, không một lựa chọn tie nào khôi phục được kết quả COS. Đây là bằng
chứng mạnh nhất để phân biệt CONT → OT → SIM với một tổng có trọng số.

Nhóm 40 bệnh nhân cho tín hiệu đặc biệt nhất: 24/24 instance có cả CONT và OT
tốt hơn bắt buộc dưới COS. Ở nhóm 30 bệnh nhân, các con số tương ứng là 17/24
cho CONT, 19/24 cho OT và 12/24 cho cả hai.

### Continuity budget cho biết giá của việc nới ưu tiên

`k=0` khôi phục đúng vector COS ở 48/48 instance. Khi tăng `k` từ 0 lên 1:

- chỉ 3 instance giảm OT, tổng cộng 3 đơn vị;
- median SIM tăng 8 điểm;
- 46 instance tăng SIM, một không đổi và một giảm SIM do OT được ưu tiên trước.

Khi tăng từ 1 lên 2, không instance nào giảm thêm OT và median SIM tăng 4 điểm;
SIM tăng ở 48/48 instance. Kết quả biến thứ tự lexicographic thành một công cụ
ra quyết định đọc được: một mức nới CONT có giá trị rõ ràng thay vì một hệ số
trao đổi ẩn.

### Weighted phụ thuộc thang hệ số

Cả 48 instance tạo nhiều vector mục tiêu trong lưới chín trọng số; mỗi instance
có từ 4 đến 9 vector khác nhau. Với ba cặp cùng tỷ lệ `wc:wo=1:1`:

| Trọng số | Số instance trùng đúng vector COS |
|---|---:|
| `(1,1)` | 0/48 |
| `(4,4)` | 8/48 |
| `(8,8)` | 25/48 |

Chuyển `(1,1)→(4,4)` đổi vector ở 48/48 instance; `(4,4)→(8,8)` đổi 36/48.
Các cặp này không tương đương vì hệ số SIM giữ bằng 1. Claim phù hợp là Weighted
cần hiệu chỉnh thang tuyệt đối, trong khi COS mô tả trực tiếp thứ tự ưu tiên.

### Kết quả ổn định qua các mức năng lực

Trên chín ô năng lực, Weighted khác COS ở 40–48 trong số 48 instance. Vì vậy
khác biệt chính không chỉ xuất hiện tại cấu hình tải gốc.

COS và LEX-OT chỉ khác ở ba instance của ô gốc
`rho=0.85, normal_fraction=0.85`; tám ô còn lại không có xung đột. Kết quả này
giới hạn đúng phạm vi diễn giải: thứ tự giữa CONT và OT thay đổi vector mục
tiêu tại một trong chín mức năng lực đã thử. Lưới này chưa xác định được độ
rộng của vùng xung đột. Khác biệt với Weighted xuất hiện ở cả chín mức.

Các biến thể nằm trong sáu họ patient-seed và chia sẻ dữ liệu cha. Số liệu load
được dùng như so sánh matched có tính mô tả. Có 432 biến thể với ba policy;
các biến thể cùng họ không được coi là mẫu độc lập.

## Cách tái tạo số liệu trong bản thảo

```bash
python3 experiments/generate_research_depth_manuscript_results.py \
  --results-root results_research_depth_full \
  --output-dir LaTeX-Templates/paper/generated_research_depth

cd LaTeX-Templates/paper
TEXINPUTS=..: BSTINPUTS=..: \
  latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Generator kiểm tra số run, trạng thái hoàn thành, source commit, dirty flag,
optimal-face coverage và xác minh lịch trước khi ghi macro. File provenance lưu
SHA-256 của mọi bảng phân tích đầu vào.

## Trạng thái bản thảo

Bản thảo đã được cập nhật ở Abstract, RQ1, Contributions, Experimental
Methodology, Results, Discussion và Conclusion. Một bảng mới tóm tắt
optimal-face, continuity budget, weight sensitivity và capacity sensitivity.
Bản PDF sau cập nhật kết quả có tám trang. Lần biên tập tiếp theo ngày
11/09 chuyển chứng minh, chi tiết triển khai, kiểm định và bảng phụ sang
[`EXPERIMENTAL_SUPPLEMENT.md`](EXPERIMENTAL_SUPPLEMENT.md), rút bản chính
xuống sáu trang và giữ các kết quả khoa học chính.

Các claim nên giữ ở mức sau:

> Our contribution is the CONT → OT → SIM policy for HCORAP and an exact
> evaluation of its decision consequences. The optimal-face analysis shows
> that the gains over Weighted cannot generally be recovered through
> tie-breaking, while continuity budgets and weight and capacity sweeps make
> the scope and cost of the priority explicit.

Không dùng việc có cả Gurobi và CPLEX làm claim novelty. Hai solver cung cấp
kiểm tra độc lập và reference optima cho các kết quả về policy.

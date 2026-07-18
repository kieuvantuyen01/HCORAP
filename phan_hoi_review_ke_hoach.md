# Phản hồi và phương án sửa `review_ke_hoach.md`

Tài liệu này phân loại từng nhận xét trong review thành: **chấp nhận**,
**chấp nhận một phần**, hoặc **phản biện**. Mục tiêu không phải bảo vệ bản kế
hoạch cũ, mà là chỉ tiếp nhận những nhận xét đúng về logic/mô hình và sửa chính
xác những chỗ review cũng còn nhầm.

## Kết luận nhanh

| Mục review | Kết luận | Hành động |
|---|---|---|
| 2.1 Novelty | Chấp nhận phần lớn, nhưng dẫn chứng Demirović không chứng minh lexicographic MaxSAT | Hạ tuyên bố novelty; định vị đóng góp theo tổ hợp audit–benchmark–controlled comparison; bắt buộc baseline ngoài MaxSAT |
| Weighted sum hay benchmark | Chấp nhận | Xem calibrated weighted là strong baseline; cho phép kết quả âm |
| 2.2 Stability–CONT | Chấp nhận nhu cầu chứng minh; phản biện công thức hằng số trong review | Thêm mệnh đề với hằng số đúng `sum_q(|SEQ(q)|-1)` |
| 2.3 CONT_N | Chấp nhận cho coverage mềm | Định nghĩa metric theo số service thực sự được phục vụ; overload chỉ là stress test |
| 2.4 Load ratio rho | Chấp nhận | Dùng rho như chỉ báo; micro-pilot trên calibration seeds; freeze generator trước evaluation |
| 2.5 Equality trong lex | Phản biện “loại nghiệm tốt”; chấp nhận yêu cầu tie tests | Dùng bound bảo toàn optimum (`<=`, `>=`), vốn tương đương equality tại optimum |
| Chi phí nhiều tầng | Chấp nhận | Cumulative timeout, runtime gate, incremental solving và call-budget |
| 2.6 Tên Pareto | Chấp nhận | Đổi thành similarity-budget epsilon-constraint tri-objective với lexicographic completion |
| 2.7 “Tái lập một phần” | Chấp nhận | Nhắm tới toàn bộ 800 instance; báo statistic gốc cạnh PAR-2/cactus |
| 2.8 Weight sensitivity | Chấp nhận chi tiết implementation; chưa đủ cơ sở để kết luận log-grid chắc chắn “quá thưa” | Ghi rõ soft-clause weights; screening rồi adaptive refinement |
| 2.9 Statistical power | Chấp nhận | Pilot-based power/precision analysis; seed là đơn vị độc lập; Holm correction |
| 2.10 Tiến độ | Chấp nhận | Micro-pilot cuối tuần 5, integration pilot tuần 11, thêm 2 tuần contingency |

## 1. Novelty và lý do chọn MaxSAT

### Phần review đúng

Lexicographic optimization đã xuất hiện trong personnel rostering và home care;
multi-objective MaxSAT cũng là một hướng nghiên cứu đã tồn tại. Vì vậy không thể
xem “dùng lexicographic” tự thân là đóng góp đủ mạnh. Reviewer cũng có lý khi
hỏi tại sao không dùng CP-SAT/MILP.

### Phần cần phản biện

Demirović, Musliu và Winter (2019) dùng **Partial Weighted MaxSAT** cho staff
scheduling. Công trình đó không phải bằng chứng trực tiếp rằng họ đã dùng
lexicographic MaxSAT. Tuy nhiên, kết luận rộng hơn của review vẫn đứng vững nhờ
các nghiên cứu lexicographic trong rostering/home care và nghiên cứu
multi-objective MaxSAT khác.

### Cách sửa

- Không tuyên bố “lần đầu”.
- Định vị đóng góp là một gói có thể kiểm chứng: audit formulation/source,
  corrected paired benchmark, calibrated weighted baseline, priority policies,
  epsilon-constraint, verifier và ablation.
- CP-SAT hoặc MILP trở thành baseline bắt buộc thay vì “nếu nguồn lực cho phép”.
- MaxSAT được giữ để có controlled comparison với Boolean encoding gốc, không
  phải vì mặc định MaxSAT tốt hơn.
- Nếu calibrated weighted cho kết quả tương đương, báo đây là kết quả âm có giá
  trị: vấn đề chính nằm ở benchmark hoặc cách báo metric.

## 2. Weighted sum và benchmark là hai nguyên nhân khác nhau

Nhận xét này đúng và là sửa đổi quan trọng nhất. Benchmark gốc gần như không
kích hoạt overtime, nên không thể từ đó kết luận weighted sum thất bại.

Thiết kế mới phải tách hai yếu tố:

1. giữ optimizer, đổi official benchmark sang corrected benchmark;
2. giữ benchmark, đổi `(1,1)` sang calibrated weights;
3. giữ benchmark đã freeze, đổi weighted sang lex/epsilon-constraint;
4. dùng ablation để gán nguồn thay đổi.

Lexicographic không “tốt hơn” theo nghĩa tuyệt đối. Khác biệt có thể bảo vệ được
là semantics: weighted sum cho phép bù trừ; lexicographic cấm mọi bù trừ làm xấu
tầng ưu tiên cao. Weighted sum có thể mô phỏng lex bằng domination weights,
nhưng các weights này phụ thuộc bound/kích thước instance và không còn là một
trade-off bù trừ thông thường.

## 3. Stability và CONT: review đúng ý nhưng sai hằng số

Với sequence `q` có `n_q` services và `D_q` agents khác nhau, encoding gốc có:

```text
c[q,i] = 1  iff  D_q >= i,  i=1,...,n_q.
```

Do đó:

```text
STAB_q = sum_i (1-c[q,i]) = n_q-D_q.
CONT_q = D_q-1.
```

Suy ra, khi coverage là hard:

```text
STAB = sum_q(n_q-1) - CONT.
```

Review viết `C_inst=sum_q n_q`; công thức này lệch đúng số sequence. Hằng số
đúng là `sum_q(n_q-1)`. Bản kế hoạch sửa đã thêm mệnh đề và chứng minh, đồng
thời ghi rõ mệnh đề chỉ đúng khi mọi service được phục vụ.

## 4. Edge case coverage mềm

Nhận xét đúng nhưng chỉ áp dụng cho overload extension; HCORAP gốc dùng coverage
hard nên `D_q>=1`. Nếu coverage mềm, sequence không được phục vụ có thể có
`D_q=0`, làm `D_q-1` âm.

Cách sửa:

```text
K_q = số services của q đã được phục vụ
CONT_serv = sum_q max(0,D_q-1)
denominator = sum_q max(0,K_q-1)
```

Continuity chỉ được so sánh giữa nghiệm có cùng coverage. Overload được hạ xuống
stress test thứ cấp; tập low/critical chính vẫn giữ coverage hard.

## 5. `rho` và benchmark calibration

Review đúng: `rho=S/sum capacity` không chứa time windows, qualification,
simultaneity hoặc candidate scarcity. Các mức `0.60, 0.85, 1.05` không được coi
là nhãn tải có cơ sở trước pilot.

Quy trình sửa:

- dùng các giá trị trên như grid khởi đầu;
- chạy micro-pilot trên calibration seeds;
- ghi candidate density, singleton-candidate rate và qualification scarcity;
- gán nhãn theo feasibility/overtime thực nghiệm của baseline;
- freeze generator/thresholds;
- chỉ sau đó mới mở evaluation seeds.

Điều này tránh vừa thiết kế benchmark vừa điều chỉnh theo kết quả của phương
pháp đề xuất.

## 6. Correctness của sequential lexicographic

Nhận xét “equality có thể loại nhiều nghiệm tốt” không phải lỗi correctness.
Theo định nghĩa lexicographic, mọi nghiệm làm xấu tầng trước đều phải bị loại.
Nếu `OT*` là minimum thì `OT<=OT*` và `OT=OT*` tương đương trên feasible region.
Dùng inequality bound chỉ giúp encoding tự nhiên hơn.

Phần review đúng là phải test ties. Bản sửa bổ sung instance có nhiều nghiệm
đồng tối ưu và xác nhận tầng sau không làm xấu optimum tầng trước. Coverage chỉ
là tầng đầu trong overload; với full coverage nó là hard và được bỏ khỏi chuỗi,
giảm từ bốn xuống ba tầng.

## 7. Chi phí tính toán

Phê bình này đúng, nhưng “4x” không phải bound toán học: mỗi tầng có search space
khác nhau và incremental reuse có thể làm overhead thấp hoặc cao hơn bốn lần.
Cần báo cumulative runtime và runtime từng tầng.

Bản kế hoạch cũ còn có vấn đề lớn hơn review chưa nêu: chạy 16 weights trên 800
official instances đã là 12,800 solver calls, chưa tính corrected benchmark và
các tầng lex/epsilon. Thiết kế mới dùng screening subset, adaptive refinement,
policy shortlist, cumulative timeout và tính `N_call`/worst-case core-hours
trước mỗi campaign.

## 8. B2 không phải full Pareto enumeration

Review đúng về tên gọi, nhưng đề xuất “bi-objective” cũng chưa chính xác vì B2
có ba tiêu chí `(SIM, CONT, OT)`. Tên sửa là:

> similarity-budget epsilon-constraint tri-objective MaxSAT with
> lexicographic completion.

Năm budget là tập đại diện rời rạc, không phải toàn bộ Pareto front. Có thể
adaptive-refine nếu còn compute budget.

Hai lỗi bổ sung trong bản cũ:

1. Phải dùng `ceil((1-delta)SIM*)`, không dùng `floor`, để loss không vượt budget.
2. Sau khi min CONT và min OT, phải max SIM lần cuối trong ties. Nếu bỏ tầng này,
   nghiệm trả về có thể bị một nghiệm cùng CONT/OT nhưng SIM cao hơn dominate.

Sau lexicographic completion đầy đủ, mỗi điểm là globally non-dominated trong
ba objective; các điểm giữa nhiều budget có thể trùng và phải được gộp.

## 9. Tái lập baseline

“Tái lập một phần” quá mơ hồ. Bản sửa yêu cầu status/quality trên toàn bộ 800
instances và nhắm tới tái tạo mọi ô Bảng 5–6 trong time limit khả dụng.

Runtime tuyệt đối không cần khớp nếu hardware/solver khác, nhưng phải báo song
song:

- conditional mean theo cách bài gốc để so trực tiếp;
- solved/timeout counts, PAR-2 và cactus plot để không mắc survivor bias.

## 10. Weight sensitivity

Review đúng khi yêu cầu mô tả WCNF chính xác:

```text
< y[a,s],              r(a,s) >
< not c[q,i],          w_c    >
< not w[a,i],          w_o*|P|>
```

Không có scalarization ngoài solver. Log-grid `{1,2,4,8}^2` là screening hợp
lý nhưng không được gọi là đầy đủ; refine quanh vùng assignment/metric đổi.
Kết luận “quá thưa” chỉ có thể xác nhận sau screening, không nên giả định trước.

## 11. Power và đơn vị thống kê

Nhận xét về power đúng về nguyên tắc, nhưng corrected benchmark cũ mới chỉ nêu
10 pilot seeds, không cam kết 50 independent instances/configuration. Effect
thực chưa biết nên không thể tính power đáng tin cậy trước pilot.

Cách sửa là dùng calibration pilot để ước lượng variance/zero-difference rate,
đặt SESOI trước khi mở evaluation set, mô phỏng power cho Wilcoxon/sign test và
chọn số independent seeds. Base scenario/seed là đơn vị độc lập; các cấu hình
nested A/V trong cùng seed là repeated measures, không được tính như independent
samples. Dùng Holm correction và luôn báo effect size + confidence interval.

## 12. Tiến độ

Review đúng. Micro-pilot được chuyển lên cuối tuần 5, trước WP3/WP4. Tuần 11 là
integration pilot chứ không phải lần đầu kiểm tra benchmark. Tiến độ mới có ba
gate và hai tuần contingency:

- G1: freeze benchmark;
- G2: lex scalability;
- G3: freeze protocol/power/call-budget;
- tuần 15–16: sửa lỗi/rerun hoặc viết và đóng gói nếu không dùng buffer.

## Các sửa đổi bổ sung không có trong review

1. Thêm attainable similarity normalization, vì `SIM/(4S)` có thể thấp chỉ vì
   reward 4 không khả đạt trên một service.
2. Đổi nhãn value-laden `patient-first`/`worker-first` thành
   `continuity-priority`/`overtime-priority` khi chưa có stakeholder study.
3. Bắt buộc ít nhất một CP-SAT/MILP baseline.
4. Overload chỉ là stress test; kết luận chính dùng coverage hard.
5. Tính solver-call budget trước campaign để kế hoạch thực sự chạy được.


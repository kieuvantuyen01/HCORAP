# Phản biện chi tiết: submission_plan.md & main.tex

> **Ngày rà soát:** 09/08/2026
> **Tài liệu được đánh giá:** [submission_plan.md](file:///Users/tuyenkv/Documents/HCORAP/submission_plan.md) · [main.tex](file:///Users/tuyenkv/Documents/HCORAP/LaTeX-Templates/paper/main.tex)
> **Tài liệu đối chiếu:** [EXPERIMENT_GAP_AUDIT_20260808.md](file:///Users/tuyenkv/Documents/HCORAP/docs/EXPERIMENT_GAP_AUDIT_20260808.md) · [SCREENING_RESULTS_20260717.md](file:///Users/tuyenkv/Documents/HCORAP/docs/SCREENING_RESULTS_20260717.md) · [FAIR_EXPERIMENT_PROTOCOL.md](file:///Users/tuyenkv/Documents/HCORAP/docs/FAIR_EXPERIMENT_PROTOCOL.md) · [guide.md](file:///Users/tuyenkv/Documents/HCORAP/guide.md) · [literature_review.md](file:///Users/tuyenkv/Documents/HCORAP/literature_review.md)

---

## Tóm tắt đánh giá tổng thể

**Điểm mạnh nổi bật:** Kế hoạch có mức độ kỹ thuật cao, tư duy thực nghiệm nghiêm túc (gate G1–G3, data freeze, paired ablation, provenance). Định nghĩa tường minh policy LEX-COS và các câu hỏi nghiên cứu RQ1–RQ3 là rõ ràng và hợp lý. Quyết định cắt giảm campaign từ 16.040 xuống 4.896 runs là thực tế và được lập luận tốt.

**Vấn đề nghiêm trọng cần giải quyết trước khi chạy campaign:** Có ba nhóm vấn đề chính—(A) **mâu thuẫn nội bộ** giữa các tài liệu, (B) **khoảng trống bằng chứng** chưa được thừa nhận trong plan, và (C) **rủi ro deadline** thực tế. Nếu không xử lý, bản thảo có nguy cơ bị reject vì phương pháp luận hoặc không kịp deadline.

---

## I. Mâu thuẫn nội bộ giữa các tài liệu

### I.1 — Cấu hình "proposed" không khớp với screening results

**submission_plan.md (§3.1)** khóa:
```
proposed = totalizer / implied both / symmetry slot-service
```

Nhưng **SCREENING_RESULTS_20260717.md** đóng băng candidate cho pilot tiếp theo là:
```
sorting-network / both-plus / none
```

Và **FAIR_EXPERIMENT_PROTOCOL.md** ghi rõ: *"Symmetry `none` remains frozen provisionally because it had the lowest PAR-2 and preserves the baseline model."*

**Hậu quả:** Cấu hình `proposed` trong plan chọn `totalizer + both + slot-service`, nhưng pilot evidence lại ủng hộ `sorting-network + both-plus + none`. Plan hiện không giải thích vì sao chuyển sang `both` thay vì `both-plus`, và vì sao chọn `slot-service` thay vì `none` cho symmetry. Đây không phải quyết định nhỏ—screening cho thấy `both-plus` cải thiện PAR-2 từ 9.218 xuống 5.456 trên seed 2–5, trong khi `both` chưa được đánh giá riêng biệt ở quy mô đủ.

> [!WARNING]
> Phải bổ sung một đoạn giải thích rõ trong plan: tại sao chọn `both` chứ không phải `both-plus` (lý do kỹ thuật cụ thể), và tại sao thêm `slot-service` symmetry vào proposed configuration mặc dù screening không cho thấy lợi thế rõ ràng từ symmetry breaking.

### I.2 — Kích thước LEX-COS confirmatory set mâu thuẫn

**submission_plan.md** ghi:
- §3.1: "LEX-COS confirmatory trên 280 held-out instances thuộc 14 lớp"
- §4 (bảng): "original LEX-COS primary: 280 held-out × baseline/proposed → **560 runs**"

**EXPERIMENT_GAP_AUDIT_20260808.md (§3)** ghi:
- "Corrected-v2 confirmatory chỉ chạy **160** evaluation-critical instances bằng proposed configuration"
- Bảng screen: "corrected-v2 primary: **160** × weighted/LEX-COS → **320** runs"

Có hai bộ số khác nhau: 280 instance cho original LEX-COS primary và 160 instance cho corrected-v2. Đây là hai campaign riêng nhưng cách trình bày trong §3.1 và §4 có thể gây nhầm lẫn. Cần xác nhận: 280 original instances và 160 corrected-v2 instances là **hai tập riêng biệt hoàn toàn**, không giao nhau và không thay thế nhau.

### I.3 — Mâu thuẫn về vai trò của `sorting-network` baseline

Plan (§3.1) định nghĩa:
```
baseline = sorting-network / implied none / symmetry none
```

Nhưng screening cho thấy `both-plus` cho PAR-2 tốt hơn `none` đáng kể ngay cả ở baseline role. Nếu cấu hình `baseline` đang dùng `none` cho implied constraints trong khi `both-plus` được biết là tốt hơn, thì kết quả so sánh `baseline vs proposed` sẽ bị thiên vị có lợi cho proposed. Điều này không nhất thiết là sai—nếu đây là intentional ablation—nhưng cần được nêu rõ trong Threats hoặc Experimental Setup của bản thảo.

---

## II. Khoảng trống bằng chứng chưa được thừa nhận

### II.1 — LEX-COS confirmatory tại quy mô lớn chưa có pilot đủ

Gate G2 yêu cầu: *"LEX-COS hoàn tất ít nhất 60% B0-optimal instances ở ít nhất một config"* trên lex scalability screen (80 instances, timeout 300 s).

Audit cho thấy ở quy mô U30/A10/V5 với timeout 15 s, LEX-COS đã **timeout toàn bộ 10/10 instances**. Screening 300 s chưa được chạy. Không có bằng chứng hiện tại rằng LEX-COS sẽ qua được gate 60% ở timeout 300 s trên các instance lớp lớn (40_25_5, v.v.).

**Rủi ro:** Nếu G2 là NO-GO vì LEX-COS không đủ completion rate, toàn bộ campaign LEX-COS confirmatory (560 runs, 46.67 core-hour) sẽ bị dừng. Plan không có plan B rõ ràng cho tình huống này ngoài câu "dừng publication campaign".

> [!CAUTION]
> Khuyến nghị: trước ngày 10/08 (trước khi tạo GCP VM), chạy thử LEX-COS với timeout 300 s trên ít nhất 5–10 instance lớp khó nhất tại local (macOS binary) để đánh giá xem gate 60% có khả thi không. Nếu không, cần điều chỉnh claim hoặc timeout ngay từ đầu thay vì chờ đến 16/08.

### II.2 — Corrected-v2 chưa chạy ở quy mô publication

Audit ghi: *"Pilot hợp lệ ở quy mô U30/A10/V5, rho≈0.847, timeout 15 s cho 10/10 TIMEOUT."*

Corrected-v2 evaluation-critical campaign (160 instances, 320 runs, timeout 300 s) là "chứng minh signal tồn tại khi overtime ít sparse". Nhưng với U30/A10/V5 đã full timeout ở 15 s, không rõ 300 s có đủ không ở các instance lớn hơn. Plan cần thừa nhận rủi ro này explicitly.

### II.3 — Gurobi/CPLEX license chưa có trên GCP VM

Audit (§1.2) ghi rõ: *"Gurobi/CPLEX SDK và license chưa có."*

Nhưng lịch plan đặt commercial run vào ngày 27–29/08—chỉ 3 ngày, sau khi đã chạy xong phần lớn campaign. Nếu license gặp vấn đề (academic license không hoạt động trên GCP VM, IP restriction, quota), cả campaign commercial (400 runs, 33.33 core-hour, quan trọng cho validation claim) sẽ không thể chạy trong thời gian còn lại.

> [!WARNING]
> Cần giải quyết license Gurobi/CPLEX **trước ngày 10/08**, không phải sau khi đã chạy xong phần MaxSAT. Đặt commercial preflight sớm hơn trong lịch—có thể ngay ngày 10–11/08 song song với preflight MaxSAT.

### II.4 — Không có cơ chế xử lý khi G2 partial fail

Plan §6 viết: "Nếu G2 là NO-GO, dừng publication campaign." Nhưng G2 bao gồm nhiều tiêu chí độc lập:
- proposed optimum count ≥ 95% baseline
- LEX-COS completion rate ≥ 60% ở ít nhất một config
- peak RSS ≤ 12 GB

Nếu chỉ một tiêu chí fail (ví dụ LEX-COS completion < 60% nhưng encoding và weighted đều OK), "dừng toàn bộ" là quá cứng nhắc. Cần phân loại rõ hơn: tiêu chí nào là hard stop (e.g., objective mismatch, RSS), tiêu chí nào là soft warning cho phép narrow claim?

---

## III. Rủi ro lịch thực hiện

### III.1 — Buffer thực tế là 0 ngày nếu có sự cố VM

Lịch (§8):
- 09/08: khóa code
- 10–11/08: chuẩn bị GCP
- 12–15/08: screen (1.856 runs, max 73.6 core-hour)
- 16/08: gate review
- 17–23/08: original weighted primary (1.600 runs, 133.33 core-hour worst case)
- 28/09: internal submit

**Vấn đề 1 — Thời gian thực chạy:** 1.600 runs × 300 s worst case = 480.000 s ≈ **133 giờ** với 1 core. Trên 8 vCPU (mỗi run 1 vCPU pinned), song song 8 runs → ~16.7 giờ thuần chạy. Nhưng plan dành 7 ngày (17–23/08) = 168 giờ, tức buffer ~10x về compute. Đây là OK nếu VM chạy liên tục 24/7.

**Vấn đề 2 — Không có kế hoạch xử lý VM failure.** GCP VM có thể bị preempt (nếu dùng Spot), hết disk, network outage. Plan đề cập "resume" nhưng không có ngưỡng: nếu campaign original weighted chỉ hoàn thành 60% khi đến 23/08, có tiếp tục không?

**Vấn đề 3 — Thời gian viết bản thảo quá ngắn.** 06–13/09 = 8 ngày để viết toàn bộ nội dung bài 5 trang. Với người có kinh nghiệm, đây rất chật—đặc biệt khi cần verify số liệu từ frozen tables. Khuyến nghị dành 10–12 ngày cho giai đoạn viết, tức đẩy analysis lên sớm hơn.

> [!IMPORTANT]
> Đề xuất tái cân bằng lịch: bắt đầu viết Methods + Related Work ngay từ tháng 8 (dựa trên design đã khóa, không cần đợi frozen data). Chỉ Results mới cần đợi frozen data.

### III.2 — Ngày deadline 30/09 chưa được verify lại

Plan §8 ghi: "Nguồn chính thức hiện ghi full-paper deadline 30/09/2026." Cần kiểm tra [ICIIT 2027 Important Dates](https://www.iciit.org/date.html) ngay hôm nay. Lịch hội nghị hay thay đổi; nếu deadline là 15/09 thay vì 30/09, toàn bộ lịch sụp đổ.

---

## IV. Phản biện nội dung khoa học

### IV.1 — RQ3 có thể không trả lời được với chỉ một solver

RQ3 hỏi về "tương tác với cardinality encoding và cấu trúc instance". Với chỉ một solver (Open-WBO 2.1), kết quả có thể là artifact của heuristic cụ thể của solver đó. Plan thừa nhận trong Threats ("one pinned MaxSAT solver/version") nhưng không giải thích vì sao không thử ít nhất một solver thứ hai (EvalMaxSAT, RC2) ngay cả chỉ trên screening subset để kiểm tra robustness của claim.

### IV.2 — Staged LEX solving là weak point chưa được defend đủ

LEX-COS giải theo staged restart: solve CONT → fix bound → solve OT → fix bound → solve SIM. Cách này **không** đảm bảo lexicographic optimality nếu solver không chứng minh UNSAT của stage hiện tại trước timeout. Plan thừa nhận điều này ("classify as timeout") nhưng không giải thích:

1. Có bao nhiêu phần trăm runs sẽ timeout ở stage đầu (CONT minimization) trên original instances?
2. Nếu >30% runs timeout ngay từ stage đầu, reported LEX-COS "results" thực chất là partial.

Bằng chứng hiện tại (screening): `lex-continuity` timeout 6/6 instances ngay cả với `both-plus` ở timeout 30 s. `lex-cos` (dùng CONT ở stage đầu) sẽ có cùng vấn đề. Plan không đề cập bằng chứng này một cách tường minh.

### IV.3 — "Bidirectional Totalizer" chưa được document đủ trong main.tex

[main.tex L237-L243](file:///Users/tuyenkv/Documents/HCORAP/LaTeX-Templates/paper/main.tex#L237-L243): Outline note chỉ ghi *"output ok is equivalent to 'at least k+1 inputs are true'"*. FAIR_EXPERIMENT_PROTOCOL.md mô tả rõ hơn: *"Totalizer dùng clauses hai chiều."* Điều này quan trọng vì **bidirectional** là điều kiện để implied constraints và symmetry breaking không làm sai optimum. Cần cite đúng paper—BailleuxBoufkhad2003 là Totalizer một chiều; nếu dùng bidirectional thì cần cite variant paper.

### IV.4 — Related Work table thiếu một số stream quan trọng

[main.tex L161-L181](file:///Users/tuyenkv/Documents/HCORAP/LaTeX-Templates/paper/main.tex#L161-L181): Related Work table có 5 rows ngoài "This work". Tuy nhiên:

- **Demirovic et al. 2017** (MaxSAT staff scheduling) xuất hiện trong literature_review nhưng **không có trong bảng**. Đây là paper closest về methodology—MaxSAT + staff scheduling + gợi ý lexicographic optimization. Việc bỏ paper này khỏi bảng là lỗ hổng reviewer có thể phát hiện.
- **Morgado et al. 2014** (Core-guided MaxSAT với cardinality constraints) cũng không trong bảng dù liên quan trực tiếp đến encoding.
- **Jahren & Asín 2018** được cite trong text nhưng không trong bảng taxonomy.

> [!NOTE]
> Bảng hiện có cột "IC/SB" nhưng rows cho MaxSAT encoding papers (AsinEtAl2011, BofillEtAl2022, BogaertsEtAl2022) lại gộp chung. Nên tách thành ít nhất 2 rows để phân biệt cardinality encoding papers với symmetry/implied papers.

### IV.5 — Novelty claim cần được phân biệt rõ với Marques-Silva 2011

Thông điệp bài báo (submission_plan.md §1) là "Một policy lexicographic tường minh loại bỏ sự mơ hồ của weighted optimum". Tuy nhiên, Marques-Silva et al. 2011 đã làm lexicographic MaxSAT trong domain tổng quát. Novelty của bài này nằm ở **application** (HCORAP) và **empirical evaluation** kết hợp với Totalizer + implied + symmetry. Nhưng [main.tex §Introduction](file:///Users/tuyenkv/Documents/HCORAP/LaTeX-Templates/paper/main.tex#L118-L143) chưa có câu nào phân biệt rõ điều này.

Cần thêm câu: *"Unlike Marques-Silva et al. [2011], which studies general Boolean lexicographic optimization, we specialize the staged policy to three domain-specific objectives (CONT, OT, SIM) and evaluate its interaction with three encoding-level factors in a home-care application."*

---

## V. Phản biện main.tex outline cụ thể

### V.1 — Thứ tự Results sections (RQ2 trước RQ1) cần giải thích

[main.tex L300](file:///Users/tuyenkv/Documents/HCORAP/LaTeX-Templates/paper/main.tex#L300): Section 5 trình bày **RQ2–RQ3 trước RQ1**. Đây là lựa chọn design hợp lý (encoding factors nên được evaluate trước khi compare policies), nhưng nó đi ngược lại thứ tự RQ trong Introduction. Cần một câu transition giải thích: *"We first establish encoding effects (RQ2–RQ3) before comparing objective policies (RQ1), because policy comparisons use the fixed proposed configuration."*

### V.2 — Phần Method chưa có pseudocode cho staged algorithm

[main.tex L224-L230](file:///Users/tuyenkv/Documents/HCORAP/LaTeX-Templates/paper/main.tex#L224-L230): Outline note ghi "Explain the staged algorithm in four lines of prose or compact pseudocode." Với không gian 5 trang hạn chế, một pseudocode 8–10 dòng sẽ thay thế được 2–3 đoạn văn và giúp reviewer hiểu chính xác cumulative timeout semantics.

### V.3 — Abstract không có số cụ thể cho đến data freeze

[main.tex L93-L103](file:///Users/tuyenkv/Documents/HCORAP/LaTeX-Templates/paper/main.tex#L93-L103): Outline note cho Abstract liệt kê 5 moves nhưng move (5) "give two or three quantitative findings" là placeholder đến data freeze. Với bài 5 trang, abstract cần ít nhất 1–2 số cụ thể. Rủi ro: abstract viết vội sau data freeze thường kém chất lượng. Khuyến nghị viết abstract draft sớm (không dùng số) và chỉ điền số sau data freeze.

### V.4 — Validation table (Table 3) thiếu cột quan trọng

[main.tex L359-L374](file:///Users/tuyenkv/Documents/HCORAP/LaTeX-Templates/paper/main.tex#L359-L374): Bảng validation có 3 cột: Runs, Opt., Agree. Nhưng thiếu:
- **Policy** column (weighted vs LEX-COS) — reviewer cần biết agreement rate của từng policy riêng
- **Timeout** count — không biết bao nhiêu runs không reach optimum

Với chỉ 3 cột, bảng không đủ thông tin để convince reviewer về validity của MIP comparison.

### V.5 — Keyword list thiếu "Totalizer" và "home health care scheduling"

[main.tex L105-L106](file:///Users/tuyenkv/Documents/HCORAP/LaTeX-Templates/paper/main.tex#L105-L106): Keywords hiện tại đã có symmetry breaking—OK. Nhưng thiếu `Totalizer` và `home health care scheduling` (variant tìm kiếm phổ biến hơn `home-care`). Có thể thêm 1–2 keyword nếu conference cho phép.

---

## VI. Đề xuất ưu tiên xử lý

| Ưu tiên | Vấn đề | Hành động cụ thể | Deadline |
|---|---|---|---|
| 🔴 **Cao** | License Gurobi/CPLEX chưa có trên GCP | Kiểm tra + request academic license ngay, test trên VM mẫu | Trước 10/08 |
| 🔴 **Cao** | Mâu thuẫn `proposed config` vs screening evidence | Bổ sung 1 đoạn trong plan giải thích lý do chọn `both` thay `both-plus` và `slot-service` thay `none` | Trước khi commit |
| 🔴 **Cao** | Verify ICIIT deadline 30/09 | Truy cập [ICIIT Important Dates](https://www.iciit.org/date.html) và cập nhật plan | Hôm nay |
| 🟡 **Vừa** | LEX-COS completion rate chưa được test ở timeout 300 s | Chạy 5–10 instance lớn nhất tại local với 300 s timeout | Trước 10/08 |
| 🟡 **Vừa** | Related Work table thiếu Demirovic 2017 và Morgado 2014 | Thêm vào bảng; điều chỉnh narrative | Trong giai đoạn viết |
| 🟡 **Vừa** | Validation table thiếu Policy column và Timeout count | Redesign Table 3 với ≥5 cột | Trong giai đoạn viết |
| 🟡 **Vừa** | G2 không có phân loại hard stop vs soft warning | Bổ sung phân loại trong §6 | Trước ngày 16/08 |
| 🟢 **Thấp** | Abstract viết cuối có rủi ro chất lượng | Viết abstract draft sớm (không dùng số); điền số cuối | Sau data freeze |
| 🟢 **Thấp** | Pseudocode cho staged LEX algorithm | Thêm vào §4 trong main.tex | Trong giai đoạn viết |
| 🟢 **Thấp** | Novelty claim chưa phân biệt với Marques-Silva 2011 | Thêm một câu positioning trong Introduction | Trong giai đoạn viết |

---

## VII. Điểm tích cực cần giữ nguyên

Những điểm dưới đây được thiết kế tốt và **không nên thay đổi**:

1. **Cơ chế data freeze G3** với 12-point checklist phù hợp tiêu chuẩn reproducibility.
2. **Phân loại rõ screen / calibration / evaluation / confirmatory data** cần được giữ nguyên trong bản thảo.
3. **Không claim "all improvements are better"**—kết quả mixed hay instance-dependent đều là publishable.
4. **Quyết định hoãn routing, full Pareto, uncertainty campaign** là đúng về scope management cho ICIIT 5-page track.
5. **Ánh xạ claim → nguồn** (§7) tốt và nên được maintain xuyên suốt quá trình viết.
6. **Loại bỏ 30_15_4 và 40_25_5 khỏi LEX-COS confirmatory set** (vì đã xem trong commercial development) là quyết định integrity tốt.
7. **Pilot corrected-v2 đã xác nhận signal** (6 vector khác nhau từ weight grid)—cần cite điều này trong Introduction như motivation cho corrected benchmark.

---

## VIII. Câu hỏi cần trả lời trước khi tiếp tục

1. **Tại sao chọn `both` thay vì `both-plus` cho proposed configuration?** Nếu lý do là kỹ thuật (e.g., `both-plus` không stable với Totalizer encoding), cần document. Nếu lý do là tiện lợi, nên reconsider.

2. **LEX-COS timeout 300 s trên lớp lớn nhất (40_25_5) là bao nhiêu?** Nếu >50% timeout, gate G2 sẽ fail và cần có plan B rõ ràng hơn "dừng campaign".

3. **"Bidirectional Totalizer" có phải là standard Totalizer của Bailleux & Boufkhad 2003 không, hay là variant?** Nếu là variant, cần cite đúng paper và mô tả difference.

4. **ICIIT 2027 có yêu cầu artifact submission không?** Nếu có, reproducibility artifact cần được upload cùng bản thảo, không phải sau notification.

5. **Author order đã được confirm với tất cả co-authors chưa?** Guide.md ghi "Phải xác nhận lại trước khi upload"—deadline xác nhận này là khi nào?

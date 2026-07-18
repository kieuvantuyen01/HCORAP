# Phản biện kế hoạch nghiên cứu HCORAP
**Tài liệu gốc:** `Ke_hoach_nghien_cuu_HCORAP.tex` (754 dòng)
**Tham chiếu:** Unceta et al. (2024) — *Optimizing resource allocation in home care services using MaxSAT*

---

## 1. Điểm mạnh — Những gì kế hoạch làm đúng

| # | Nhận xét |
|---|---|
| ✅ | Xác định được 7 lỗ hổng cụ thể trong bài báo gốc (generator lỗi ngôn ngữ, cost ≈ 0, không chuẩn hóa metric, v.v.) |
| ✅ | Tách bạch "C++ gốc = baseline" vs "Python = proposed" — điều bài báo gốc không làm rõ |
| ✅ | Thiết kế nested benchmark ($A_{10} \subset A_{15} \subset A_{20} \subset A_{25}$) giải quyết confounding factor thực sự |
| ✅ | Định nghĩa 4 RQ và 4 hypothesis rõ ràng, có thể kiểm định |
| ✅ | Có bàn đến survivor bias trong runtime (không lấy trung bình chỉ trên instances đã solved) |
| ✅ | Ablation study được thiết kế để tách biệt nguồn cải tiến |

---

## 2. Các vấn đề phản biện theo nhóm

---

### 2.1. Về lý do chọn hướng nghiên cứu — Lexicographic/Pareto MaxSAT

**Vấn đề chính:** Kế hoạch chưa chứng minh được rằng Lexicographic/Pareto MaxSAT là hướng đi *tốt hơn* Weighted MaxSAT, mà chỉ lập luận nó là "khác" và "minh bạch hơn".

> [!CAUTION]
> **Phản biện 1 — Novelty yếu:** Lexicographic MaxSAT cho bài toán scheduling không phải đóng góp mới. Demirović et al. (2019) đã dùng Partial Weighted MaxSAT cho staff scheduling. Kế hoạch chưa phân biệt rõ "đây là lần đầu áp dụng Lexicographic MaxSAT cho HCORAP" với "đây là lần đầu ai đó dùng lex MaxSAT cho home care". Reviewer sẽ hỏi ngay: *Tại sao không phải CP-SAT hoặc ILP lexicographic — vốn có tooling tốt hơn?*

> [!WARNING]
> **Phản biện 2 — Giả định chưa được kiểm chứng:** Kế hoạch cho rằng weighted sum "không minh bạch trade-off". Nhưng bài báo gốc dùng $(w_c, w_o) = (1, 1)$ và cost ≈ 0 trên hầu hết instance. Điều đó có thể *không phải lỗi của weighted sum* mà là lỗi của benchmark (capacity thừa). Nếu sau WP2 benchmark v2 sửa được vấn đề load, thì weighted MaxSAT với $(w_c, w_o)$ được calibrate tốt có thể đã đủ. **Kế hoạch cần một lý luận mạnh hơn: lexicographic giải quyết vấn đề gì mà calibrated weighted sum không giải quyết được?**

---

### 2.2. Về định nghĩa metric — Stability vs Continuity

**Vấn đề:** Kế hoạch định nghĩa $\mathrm{CONT} = \sum_{q: |SEQ(q)|>1}(D_q - 1)$ là continuity *penalty*, nhưng bài báo gốc định nghĩa stability *reward* là $\sum_{q,i}(1 - c_{q,i})$.

Hai định nghĩa tương đương về giá trị tuyệt đối, nhưng **kế hoạch chưa chứng minh điều này**. Cụ thể:

- Stability (gốc) $= \sum_q \sum_{i=1}^{|SEQ(q)|} (1 - c_{q,i}) = \sum_q (|SEQ(q)| - D_q)$
- Do đó: Stability = $\sum_q |SEQ(q)| - \mathrm{CONT}$

Phần tử $\sum_q |SEQ(q)|$ là **hằng số phụ thuộc instance**, không phụ thuộc nghiệm. Vậy tối đa stability ↔ tối thiểu CONT. **Điều này đúng, nhưng kế hoạch không phát biểu rõ và không chứng minh tính tương đương** — có thể gây confusion cho reviewer.

> [!IMPORTANT]
> **Khuyến nghị:** Thêm một mệnh đề (Proposition) ngắn: $\mathrm{stability} = C_{\text{inst}} - \mathrm{CONT}$ trong đó $C_{\text{inst}} = \sum_q |SEQ(q)|$ là hằng số instance. Điều này biện minh cho việc dùng CONT thay stability.

---

### 2.3. Về công thức CONT_N — Chuẩn hóa

Kế hoạch (Eq. 4) định nghĩa:
$$\mathrm{CONT}_N = 1 - \frac{\sum_{q:|SEQ(q)|>1}(D_q-1)}{\sum_{q:|SEQ(q)|>1}(|SEQ(q)|-1)}$$

> [!WARNING]
> **Phản biện 3 — Mẫu số có thể bằng 0 khi không phải do điều kiện đã nêu:** Kế hoạch chỉ xử lý trường hợp mẫu số $= 0$ khi *không tồn tại sequence nào có $|SEQ(q)|>1$*. Nhưng còn trường hợp nào khác không? Không — đây là đúng. Tuy nhiên phạm vi $D_q$ là gì? $D_q \in [0, |SEQ(q)|]$ hay $[1, |SEQ(q)|]$? Nếu một sequence không được phục vụ (coverage soft), $D_q = 0$ và tử số âm → $\mathrm{CONT}_N > 1$. **Kế hoạch chưa xử lý edge case này khi coverage là soft constraint.**

---

### 2.4. Về $\rho$ và tính khả thi của benchmark v2

Kế hoạch dùng:
$$\rho = \frac{S}{\sum_a (HN(a) + HE(a))}$$

> [!WARNING]
> **Phản biện 4 — $\rho$ không đủ để kiểm soát feasibility.** Bài báo gốc ghi rõ: feasibility còn phụ thuộc vào time windows (TSA, TSS) và qualification. Kế hoạch *đã thừa nhận điều này* (trang 9: "$\rho$ chỉ là chỉ báo ban đầu"). Tuy nhiên, kế hoạch đề xuất 3 mức $\rho \approx 0.60, 0.85, 1.05$ mà **không có cơ sở lý thuyết hoặc thực nghiệm sơ bộ**. Trên benchmark gốc, A=10 đã có 24/50 instances UNSAT cho S=120 — đây là tín hiệu rằng feasibility có thể rất nhạy với qualification scarcity, không phải capacity. **Cần pilot nhỏ trước khi cam kết 3 mức tải.**

---

### 2.5. Về B1 — Lexicographic MaxSAT — Tính đúng đắn

Kế hoạch mô tả quy trình lex:
```
max COV → COV*
min OT  s.t. COV = COV* → OT*
min CONT s.t. COV=COV*, OT=OT* → CONT*
max SIM  s.t. COV=COV*, OT=OT*, CONT=CONT*
```

> [!CAUTION]
> **Phản biện 5 — Ràng buộc cố định giá trị tối ưu có thể loại bỏ nhiều nghiệm tốt:** Bước 2 cố định $\mathrm{COV} = \mathrm{COV}^*$ (equality), nhưng COV là discrete (số services phủ). Việc cố định *exact equality* thay vì inequality $\mathrm{COV} \geq \mathrm{COV}^*$ là không cần thiết và không gây vấn đề về correctness. Tuy nhiên, đối với OT và CONT, kế hoạch cũng dùng equality — điều này đúng về lex nhưng cần kiểm tra kỹ: nếu solver trả về $\mathrm{OT}^* = 5$ nhưng có nhiều nghiệm với OT=5 và CONT rất khác nhau, thì bước 3 (min CONT s.t. OT=5) sẽ hoạt động đúng. **Phần implementation cần kiểm thử trường hợp có nhiều optimal ties.**

> [!WARNING]
> **Phản biện 6 — Phức tạp tính toán của Lexicographic MaxSAT:** Mỗi tầng là một MaxSAT call đầy đủ. Với 4 tầng, mỗi instance có thể tốn gấp 4× thời gian so với single-objective. Trên instances lớn (S=200), bài báo gốc đã mất trung bình 157s với 1-objective MaxSAT. Kế hoạch đặt timeout 300-3600s nhưng **chưa phân tích xem 4-tầng lex MaxSAT có khả thi trong khoảng thời gian đó không.** Ít nhất cần ước tính worst-case.

---

### 2.6. Về B2 — Similarity-budget Pareto — Conceptual issue

Kế hoạch mô tả B2 như:
1. Tính $\mathrm{SIM}^* = \max \mathrm{SIM}$
2. Với mỗi $\delta$, thêm $\mathrm{SIM} \geq \lfloor(1-\delta)\mathrm{SIM}^*\rfloor$
3. Rồi tối thiểu hóa CONT, cố định CONT*, rồi tối thiểu hóa OT

> [!CAUTION]
> **Phản biện 7 — Đây không phải "Pareto MaxSAT" theo nghĩa chuẩn.** Pareto front thực sự là tập hợp *các nghiệm không bị dominated trên toàn bộ objective space*. Cách tiếp cận $\varepsilon$-constraint của B2 chỉ lấy mẫu Pareto front dọc một chiều (giảm similarity). Điều này hoàn toàn hợp lệ về mặt thực nghiệm, nhưng **tên "Pareto MaxSAT" có thể gây hiểu nhầm**. Tên chính xác hơn là: *$\varepsilon$-constraint bi-objective optimization* hoặc *similarity-constrained multi-objective MaxSAT*.

> [!WARNING]
> **Phản biện 8 — B2 chưa định rõ Pareto front 2D hay 3D.** Sau bước 3 "min CONT rồi min OT", ta có một điểm (SIM, CONT, OT) cho mỗi $\delta$ — đây là tập rời rạc 5 điểm, không phải Pareto front liên tục. Kế hoạch cần nêu rõ: (a) 5 điểm này đủ để phân tích trade-off không? (b) Có guarantee nào rằng các điểm này nằm trên Pareto front không? (Câu trả lời là "có" với điều kiện tối ưu từng tầng.)

---

### 2.7. Về WP1 — Tái lập baseline

Kế hoạch yêu cầu "tái lập một phần Bảng 5–6" trong WP1.

> [!WARNING]
> **Phản biện 9 — "Một phần" là không đủ.** Bảng 5 của bài báo gốc báo cáo thời gian trung bình *chỉ trên certified instances*, không phải tất cả instances — đây chính là survivor bias mà kế hoạch đã phê bình. Khi tái lập với PAR-2 như kế hoạch đề xuất, con số sẽ *khác đáng kể* với bảng gốc. Kế hoạch cần dự đoán trước điều này và giải thích trong báo cáo: "Sự khác biệt với bảng gốc là do phương pháp tính toán khác (PAR-2 vs average-over-certified), không phải do lỗi implementation."

---

### 2.8. Về WP3 — Sensitivity analysis

Kế hoạch dùng $w_c \in \{1,2,4,8\}$, $w_o \in \{1,2,4,8\}$ → 16 cặp.

> [!WARNING]
> **Phản biện 10 — Không gian tìm kiếm quá thưa.** Bài báo gốc dùng $(w_c, w_o) = (1,1)$ vì đây là natural MaxSAT encoding (mỗi violated agent = 1 soft clause với weight 1). Thay đổi $w_c$ và $w_o$ *không phải* là thay đổi trọng số trong MaxSAT một cách trực tiếp — cần phải nhân weight của các soft clauses liên quan. Kế hoạch cần mô tả **cụ thể cách implement**: thay đổi weight của clauses `⟨¬c_{q,i}, w_c⟩` và `⟨¬w_{a,i}, w_o \cdot |P|⟩` trong WCNF, hay dùng scalarization bên ngoài? Đây là chi tiết kỹ thuật quan trọng ảnh hưởng đến reproducibility.

---

### 2.9. Về tuyên bố "thống kê có ý nghĩa" trong tiêu chí thành công

> [!CAUTION]
> **Phản biện 11 — Statistical power chưa được tính.** Kế hoạch dùng Wilcoxon signed-rank test với paired design, nhưng chưa tính sample size cần thiết. Với 50 instances/cấu hình và alpha=0.05, power phụ thuộc vào effect size thực sự của cải tiến continuity. Nếu hiệu ứng nhỏ (e.g., CONT giảm 5%), 50 instances có thể không đủ để phát hiện với power 80%. **Cần power analysis sơ bộ trước khi cam kết tiêu chí "có ý nghĩa thống kê".**

---

### 2.10. Về tiến độ 14 tuần

> [!WARNING]
> **Phản biện 12 — Tiến độ tối ưu nhưng thiếu buffer.** Sơ đồ 14 tuần phân bổ:
> - Tuần 1–2: WP1 (baseline + verifier)
> - Tuần 3: Tái lập bảng 5–6
> - Tuần 4–5: WP2 (benchmark v2)
> - Tuần 6–7: WP3 (sensitivity)
> - Tuần 8–9: WP4 (lexicographic)
> - Tuần 10: WP5 (Pareto)
> - Tuần 11: Pilot
> - Tuần 12: Full experiment
> - Tuần 13–14: Phân tích + viết
>
> **Vấn đề:** WP2 (sửa generator) ở tuần 4–5 nhưng pilot ở tuần 11 — khoảng cách quá lớn. Nếu pilot phát hiện benchmark v2 không hoạt động đúng, cần quay lại WP2 nhưng lúc này WP4 và WP5 đã hoàn thành. **Đề xuất: chạy micro-pilot sau WP2 (cuối tuần 5) trước khi tiếp tục WP3.**

---

## 3. Tóm tắt đánh giá

| Hạng mục | Đánh giá |
|----------|----------|
| Xác định vấn đề của bài báo gốc | ⭐⭐⭐⭐⭐ Xuất sắc, chi tiết và chính xác |
| Định nghĩa metric mới | ⭐⭐⭐⭐ Tốt, cần thêm tương đương stability↔CONT |
| Lý do chọn Lex/Pareto MaxSAT | ⭐⭐⭐ Trung bình, chưa đủ motivation vs calibrated weighted |
| Thiết kế thực nghiệm | ⭐⭐⭐⭐ Tốt, cần thêm power analysis và micro-pilot |
| Tính đúng đắn kỹ thuật | ⭐⭐⭐⭐ Tốt, một số edge case cần xử lý |
| Novelty claim | ⭐⭐⭐ Cần làm rõ hơn so với literature |
| Tiến độ | ⭐⭐⭐ Khả thi nhưng thiếu contingency |

---

## 4. Khuyến nghị ưu tiên cao

1. **[Bắt buộc]** Bổ sung một mệnh đề ngắn chứng minh tương đương stability = $C_{\text{inst}} - \mathrm{CONT}$
2. **[Bắt buộc]** Giải thích tại sao Lex/Pareto MaxSAT tốt hơn calibrated weighted MaxSAT (không chỉ "khác")
3. **[Quan trọng]** Đổi tên "Pareto MaxSAT" → "$\varepsilon$-constraint bi-objective" để tránh overstatement
4. **[Quan trọng]** Thêm micro-pilot sau WP2 vào tiến độ
5. **[Quan trọng]** Mô tả cụ thể cách implement weight sensitivity trong WP3
6. **[Nên làm]** Thêm ước tính power analysis cho Wilcoxon test
7. **[Nên làm]** Xử lý edge case $D_q = 0$ trong CONT_N khi coverage là soft

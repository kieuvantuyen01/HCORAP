# Rà soát lần 2: main.tex sau chỉnh sửa (630 dòng → +51 dòng so với v1)

> **Thời điểm rà soát:** 2026-08-09T20:03 +07:00  
> **So với phiên bản trước:** 579 dòng → 630 dòng (+51 dòng)  
> **BibTeX:** Chưa thay đổi (vẫn 144 dòng, `ErrarhoutEtAl2016` vẫn dùng `@article`)

---

## A. Các vấn đề đã được khắc phục ✅

| Vấn đề cũ | Dòng cũ | Trạng thái |
|---|---|---|
| Dấu `\\` tạo forced line break trong title | L67 | ✅ Đã xóa, title wrap tự nhiên |
| "bidirectional Totalizer **threshold** encodings" không nhất quán | L96 | ✅ Đổi thành "bidirectional Totalizer encodings" (L98) |
| "artifact-backed provenance" mơ hồ | L101 | ✅ Đổi thành câu cụ thể (L102–103) |
| "concurrency" không phải thuật ngữ chuẩn | L128 | ✅ Đổi thành "slot-conflict" (L130) |
| `HE_a` không giải thích vai trò | L232 | ✅ Đã giải thích đầy đủ (L244–246) |
| Indices $\alpha, \beta$ chưa có range | L306 | ✅ Đã thêm bounds (L345–346) |
| `\nu(G_h)` chưa nói khi nào tính | L336 | ✅ Đã thêm "computed once during formula construction" (L375) |
| Thiếu quantifier trong implied constraint | L327 | ✅ Đã thêm `\forall u \in \mathcal{U}` (L364) |
| "both-plus" mơ hồ | L341 | ✅ Đã giải thích rõ (L377–379) |
| Value-precedence clause mô tả khó hiểu | L352 | ✅ Đã dùng ký hiệu chuẩn (L390–391) |
| LEX-OCS chưa giải thích | L146 | ✅ Đã thêm "Overtime–Continuity–Similarity order" (L151) |
| "None of these questions presupposes..." defensive | L157 | ✅ Đã xóa, câu được viết lại tích cực hơn |
| Abstract câu 3 quá dài | L95–98 | ✅ Đã tách thành 3 câu riêng |
| Keywords không specific | L111 | ✅ Đổi thành "implied constraints, symmetry breaking" (L114) |
| Thiếu giải thích CONT = caregiver-fragmentation penalty | L241 | ✅ Đã thêm đoạn giải thích (L259–274) |
| STAB ↔ CONT equivalence chưa tường minh | — | ✅ Đã thêm Eq. stability-continuity mới (L264–274) |
| Pseudocode step 2 dùng "encode" mơ hồ | bước 2 | ✅ Đổi thành "Construct the WCNF for min f_i" (L304) |
| Pseudocode step 4 "active objective value" mơ hồ | bước 4 | ✅ Đổi thành "Independently verify feasibility and recompute CONT, OT, SIM" (L306) |
| "Rebuilding formula" chưa giải thích | L286 | ✅ Đã thêm "resets learned solver state" (L321) |
| Implied constraints subsection: "confirmatory" → "primary" | L319 | ✅ Đổi thành "primary treatment" (L355) |
| Bảng Related Work: "Varies" cho Jabs misleading | L206 | ✅ Đổi thành "N/A" (L214) |
| Caption bảng: "A/S" mơ hồ | L201 | ✅ Đổi thành "assignment or scheduling" (L201) |
| Panel A bảng: "160 pairs" không rõ nghĩa | L504 | ✅ Thêm "(80 instances × 2 configs)" (L550) |
| Panel B thiếu configuration cho Corrected-v2 | L509 | ✅ Thêm "$R$" vào cell (L555–556) |
| "EvalMaxSAT" sentence quá implementation-specific | L544 | Còn giữ, chưa sửa (xem §B) |
| Verifier limitation chưa được nêu | — | ✅ Đã thêm câu mới (L607–609) |

---

## B. Vấn đề còn lại — phân tích chi tiết

### B1 · Vấn đề MỚI phát sinh sau chỉnh sửa [QUAN TRỌNG]

#### B1.1 — Macro `\resultplaceholder` không bao giờ render nội dung có nghĩa

**Dòng 48–52:**
```latex
\newcommand{\resultplaceholder}[1]{%
  \ifinternaloutline
    \textcolor{red}{[pending]}%
  \fi
}
```

Macro nhận tham số `#1` nhưng **không bao giờ sử dụng nó**. Tất cả lệnh gọi như `\resultplaceholder{n}` và `\resultplaceholder{x}` trong bảng (L548–565) đều render thành `[pending]` — tham số bị bỏ qua hoàn toàn. Điều này không gây lỗi nhưng có thể gây nhầm cho collaborator: họ có thể nghĩ `n` và `x` là placeholder khác nhau, trong khi thực ra cả hai đều render như nhau.

> **Đề xuất:** Không cần sửa gấp nếu chỉ bạn dùng file này. Nhưng nếu có collaborator khác, nên thêm comment hoặc đổi `\resultplaceholder{}` thành `\resultplaceholder` (không tham số).

#### B1.2 — `\mathcal Q` được định nghĩa lại: "continuity sets" thay vì "service sequences"

**Dòng 227–231 (v2):**
```
denote caregivers, requested services, time slots, users, and continuity sets,
respectively.  A caregiver is called an agent in the benchmark files,
and the benchmark field SEQ calls each continuity set a service
sequence even though no temporal ordering is modeled within it.
```

Đây là cải tiến tốt về precision, nhưng tạo ra **bất nhất thuật ngữ**:

- Phần §3.1 (L228, L251, L385, L386): dùng **"continuity set"** — đúng
- Phần §4 (L333): dùng **"continuity set"** — đúng
- Phần §3.2 (L288): `\min\mathrm{CONT} \longrightarrow \min\mathrm{OT}` — OK, không dùng tên
- **NHƯNG** bảng Related Work (L200): `\caption{...A/S denotes assignment or scheduling; ...}` — OK
- Phần §4.1 (L332): "The reified threshold vectors encode caregiver workload and the **number of distinct caregivers per continuity set**" — ✅ đúng
- **Phần §5 Results outline (L483–488):** "slot- and **service-symmetry breaking**" — OK
- **Phần Threats (L579):** "The original benchmark contains limited **excess-workload** variation" — đúng

**Vấn đề thực sự là:** Abstract (L91) vẫn dùng "caregiver fragmentation" nhưng §3.1 (L259–261) định nghĩa CONT = "additional distinct caregivers beyond the first in each **continuity set**" (không gọi là "fragmentation"). Cần nhất quán: hoặc Abstract dùng "continuity penalty", hoặc §3.1 dùng "fragmentation penalty".

#### B1.3 — Equation `|P|` (L280) vẫn chưa được giải thích đủ rõ

**Dòng 280–284:**
```latex
\mathrm{SIM}-w_c\mathrm{CONT}-w_o|P|\mathrm{OT},
...
where P\leq0 is the signed excess-workload penalty stored in the benchmark,
so |P| is its non-negative magnitude per excess-workload unit.
```

Đây là cải tiến so với v1 (đã thêm giải thích). Tuy nhiên:

1. **$P$ được viết như thể là một scalar**, nhưng "stored in the benchmark" — điều này cần clarify: đây là hằng số per-instance hay per-caregiver? Nếu là scalar, thì `|P|` được đọc bình thường. Nếu là vector, thì `|P|` sẽ bị hiểu là norm.

2. **Ký hiệu $P \leq 0$** — Reviewer sẽ hỏi: nếu $P$ luôn âm thì tại sao không viết thẳng `\mathrm{SIM} - w_c\mathrm{CONT} + w_o P \cdot \mathrm{OT}` (với $P < 0$)? Hoặc chỉ đơn giản là `w_o p \cdot \mathrm{OT}` với `p > 0`.

> **Đề xuất:** Đơn giản hóa hoàn toàn. Đổi thành `\mathrm{SIM} - w_c\,\mathrm{CONT} - w_o\,p\,\mathrm{OT}` với `$p > 0$ is the per-excess-unit penalty magnitude from the benchmark`. Loại bỏ ký hiệu $|P|$ và $P \leq 0$ hoàn toàn.

---

### B2 · Vấn đề còn tồn tại từ v1 [chưa sửa]

#### B2.1 — `ErrarhoutEtAl2016` vẫn dùng `@article` (BibTeX)

**references.bib L26–35:**
```bibtex
@article{ErrarhoutEtAl2016,
  ...
  journal = {IFAC-PapersOnLine},
```

IFAC-PapersOnLine là series của proceedings IFAC, không phải journal. ACM Reference Format sẽ render thành "Journal of IFAC-PapersOnLine" — không chính xác. Reviewer quen với IFAC sẽ biết đây là conference paper.

> **Sửa:** Đổi thành `@inproceedings` với `booktitle = {10th IFAC Symposium on Manufacturing Modelling, Management and Control (MIM 2016)}`

#### B2.2 — Số 4,996 vs 4,896 vẫn chưa được giải thích

**Dòng 437–440:**
```
The maximum manifest contains 4,996 measured runs: 1,280 factorial, two
128-run exploratory screens, a 320-run lexicographic scalability screen,
and the gated publication phases.  Screening records the exact selected
total, between 3,856 and 4,996.
```

Đây là **giải thích mới** rất tốt: 4,996 là **maximum manifest** và range là 3,856–4,996 tùy gate outcomes. submission_plan dùng "4,896" có lẽ là con số mid-point estimate.

Tuy nhiên, câu "Screening records the exact selected total, between 3,856 and 4,996" vẫn mơ hồ:
- **3,856** = minimum (nếu tất cả exploratory screens bị drop): `1,280 + 320 + gated phases`. Cần verify phép tính này.
- **4,996** = maximum (tất cả gates pass).

Vấn đề: "gated publication phases" không được định nghĩa trong câu này. Đọc standalone, reviewer không biết phases đó là gì.

> **Đề xuất:** Thay thế bằng: "The maximum manifest contains 4,996 measured runs across all phases; the minimum confirmatory set, excluding exploratory screens that fail their gates, contains 3,856 runs."

#### B2.3 — `\shortauthors` với 3 tác giả

**Dòng 88:**
```latex
\renewcommand{\shortauthors}{Kieu et al.}
```

ICIIT thường yêu cầu "Last1 and Last2" cho 2 tác giả và "Last1 et al." cho ≥4 tác giả. Với đúng 3 tác giả, convention chuẩn ACM là liệt kê đủ: "Kieu, Do, and To". Cần kiểm tra ICIIT author guidelines.

#### B2.4 — `EvalMaxSAT` sentence vẫn quá specific

**Dòng 590–591:**
```
EvalMaxSAT was excluded when the available candidate binary failed the
official-WCNF smoke test
```

Đây là implementation detail mà reviewer không thể verify. Nên abstract hóa hoặc đưa vào footnote.

---

### B3 · Vấn đề kỹ thuật mới — Phần §3.1 thêm STAB [CẦN KIỂM TRA KỸ]

**Dòng 263–274 (MỚI):**
```latex
Under full coverage, the stability reward in the
original HCORAP formulation is
\begin{equation}
  \mathrm{STAB}
  = \sum_{q\in\mathcal Q}(|\mathcal S_q|-n_q)
  = \sum_{q\in\mathcal Q}(|\mathcal S_q|-1)-\mathrm{CONT}.
  \label{eq:stability-continuity}
\end{equation}
Consequently, maximizing the original stability reward is equivalent to
minimizing CONT.
```

**Câu hỏi kỹ thuật quan trọng:**

Bước biến đổi thứ hai:
```
sum_q (|S_q| - n_q) = sum_q (|S_q| - 1) - CONT
```

Khai triển:
```
STAB = sum_q |S_q| - sum_q n_q
     = [sum_q |S_q|] - [sum_q n_q]
```

CONT = sum_q max(0, n_q - 1) = sum_q (n_q - 1) khi n_q >= 1 (luôn đúng vì full coverage)
     = sum_q n_q - sum_q 1
     = sum_q n_q - |Q|

Vậy:
```
STAB = sum_q |S_q| - sum_q n_q
     = sum_q |S_q| - (CONT + |Q|)
     = [sum_q (|S_q| - 1)] - CONT
```

→ Biến đổi **toán học đúng**. ✅

Tuy nhiên có một vấn đề semantic:

- Câu "maximizing the original stability reward is equivalent to minimizing CONT" **đúng** (vì `sum_q(|S_q|-1)` là hằng số cho một instance cố định).
- **Nhưng**: Unceta et al. 2024 có thực sự gọi thành phần này là "stability reward" không? Hay paper gốc dùng "continuity" trực tiếp? Nếu đây là tên do bạn đặt lại, cần note rõ: "We call this component STAB to distinguish it from our CONT penalty notation."

---

### B4 · Kiểm tra nhất quán thuật ngữ toàn bài

Tôi scan toàn bộ 630 dòng để kiểm tra các thuật ngữ quan trọng:

| Thuật ngữ | Nơi xuất hiện | Nhất quán? |
|---|---|---|
| "continuity set" | L228, L251, L266, L385–386 | ✅ Nhất quán |
| "caregiver fragmentation" | L94 (Abstract) | ⚠️ Chỉ xuất hiện 1 lần, §3.1 dùng "additional distinct caregivers beyond the first" |
| "excess workload" | L94, L245, L261, L579, L581 | ✅ Nhất quán |
| "OPTIMUM" (all-caps) | L307, L546, L553, L561, L564 | ✅ Nhất quán sau sửa |
| "pre-specified" | L99, L157, L429, L441, L501, L526, L571 | ✅ Nhất quán (v1 dùng "predeclared", v2 dùng "pre-specified") |
| "Open-WBO" | L305, L449 | ✅ Nhất quán |
| "WCNF" | L304, L591 | ✅ Nhất quán |
| "$B$" (config) vs "B0" (policy) | L373, L374, L408, L413–414 | ✅ Nhất quán, đã giải thích phân biệt |
| "corrected-v2" | L387, L427–431, L523 | ✅ Nhất quán |
| "c4-highcpu-8" | L446 | ✅ Đúng GCP machine type |
| "calibration" vs "evaluation" split | L428–429, L582 | ✅ Nhất quán |
| "vCPU" | L448 | ✅ v2 dùng "vCPU" thay vì "CPU" — chính xác hơn |
| "sorted" vs "blocked" order | L452 | ✅ "seeded blocked order" |

**Phát hiện bất nhất quan trọng:**

- **L94 vs L259:** Abstract dùng "caregiver fragmentation" nhưng §3.1 định nghĩa chính thức dùng "additional distinct caregivers beyond the first in each continuity set". Hai cách diễn đạt không mâu thuẫn về nghĩa nhưng **không dùng cùng một nhãn**.

- **L138 vs L182:** Introduction (L138) dùng "weighted partial MaxSAT"; Related Work (L182) dùng "weighted partial MaxSAT" — ✅ nhất quán.

- **L201 vs L216:** Table caption dùng "IC/SB denotes implied constraints or symmetry breaking" nhưng row "Constraint strengthening" (L216) cite BofillEtAl2022 và BogaertsEtAl2022. Caption gọi chung nhóm là "Constraint strengthening" nhưng BogaertsEtAl2022 là về certified symmetry breaking, không phải implied constraints. Nên đổi row label thành "Implied constraints and symmetry breaking" để khớp chính xác với cite.

---

### B5 · Phân tích Experimental Design §5 (sau chỉnh sửa)

#### B5.1 — "both-plus" giải thích mới (L377–379) chính xác nhưng chưa đủ

**Dòng 377–379:**
```
The both-plus option is excluded from the
factorial because it also changes projected assignments, effective workload
caps, and service-slot constraints.
```

Đây là giải thích **mới và tốt** hơn "bundles additional enhancements" (v1). Tuy nhiên:
- "projected assignments" — thuật ngữ này chưa được định nghĩa ở đâu trong bài. Reviewer sẽ hỏi.
- Nếu "projected assignments" là kỹ thuật nội bộ (projection trên biến quyết định), nên dùng thuật ngữ phổ thông hơn: "it also modifies the effective feasible set beyond logically implied bounds".

#### B5.2 — Campaign run counts: phép tính có thể sai

**Dòng 437–440:**
```
The maximum manifest contains 4,996 measured runs: 1,280 factorial, two
128-run exploratory screens, a 320-run lexicographic scalability screen, and
the gated publication phases.
```

Kiểm tra phép tính:
- 1,280 factorial ✅ (160 instances × 8 cells = 1,280)
- 2 × 128 = 256 exploratory screens
- 320 lexicographic scalability screen
- Subtotal screens + factorial: 1,280 + 256 + 320 = **1,856**
- Gated publication phases = 4,996 - 1,856 = **3,140 runs**

Gated phases theo submission_plan §4:
- Primary weighted (800 × 2 configs) = 1,600
- LEX-COS primary (280 × 2 configs) = 560
- LEX-OCS (80 × 2 configs × 2 policies) = 160 (nhưng chỉ 1 policy mới → 80 runs?)
- Corrected-v2 (160 × 2 policies) = 320
- Commercial (100 × 2 solvers × 2 policies) = 400
- MaxSAT validation (100 × 1 config × 1 policy) = 100
- **Tổng gated = 1,600 + 560 + 80? + 320 + 400 + 100 = ~3,060–3,140**

Các con số gần đúng với phép tính trên. Vấn đề là không thể verify chính xác từ text hiện tại vì LEX-OCS không rõ là 80 hay 160 runs.

> [!CAUTION]
> Trước khi submission cần có một bảng break-down campaign manifest đầy đủ (có thể đặt trong appendix hoặc README của artifact) để reviewer có thể verify số 4,996.

---

### B6 · Threats to Validity — Cải tiến đáng kể nhưng còn 1 lỗ hổng

**Đã thêm mới (L607–609):**
```
The verifier checks feasibility, inherited bounds, and objective values;
it does not check an independently generated MaxSAT proof trace, so
optimality labels remain backend-reported.
```

Đây là câu **rất quan trọng** về intellectual honesty. ✅

**Lỗ hổng còn lại:**

- **L590–591** (EvalMaxSAT): Câu này vẫn quá cụ thể về implementation. Giữ nếu muốn traceability, nhưng nên thêm footnote thay vì để trong main text của Threats.

- **Chưa nhắc đến:** Warm-up effect — 10 instances được dùng để warm up machine trước khi chạy (L451). Nếu warm-up instances được chọn từ cùng benchmark pool, thì có thể có một bias nhỏ. Nên note: "The ten warm-up instances are drawn from the benchmark but their measurements are excluded."

---

### B7 · Kiểm tra LaTeX markup (vấn đề kỹ thuật nhỏ)

#### B7.1 — Dòng 138–139: khoảng cách thừa

```latex
Unceta et al.~\cite{UncetaEtAl2024} encode HCORAP as weighted partial MaxSAT.\@
A
weighted scalar score...
```

Dòng 139 bắt đầu bằng `A` ở một dòng riêng, tạo khoảng trắng thừa trong compiled output (do word wrap). Nên gộp lại: `MaxSAT.\@ A weighted scalar score...`

#### B7.2 — Dòng 387: line break giữa câu

```latex
Swapping two members of either class consequently preserves all hard constraints and all
three objective values.
```

Dòng 387 có "all hard constraints and all" tách khỏi "three objective values." — LaTeX tự wrap, không phải vấn đề. OK.

#### B7.3 — Dòng 602–603: line break giữa câu gây paragraph indent sai

```latex
while $R$
bundles three changes.
```

$R$ ở cuối dòng 602 và "bundles three changes" ở đầu dòng 603 — LaTeX tự wrap, OK. Không ảnh hưởng output.

#### B7.4 — `\label{eq:stability-continuity}` mới (L268) chưa được `\ref{}` ở bất kỳ đâu

Equation mới có label nhưng không được tham chiếu trong text. Với ICIIT 5 trang không yêu cầu mọi equation phải được ref. Tuy nhiên nếu không ref, reviewer có thể hỏi tại sao cần label. Nên hoặc là thêm `(Eq.~\ref{eq:stability-continuity})` trong câu text, hoặc xóa label.

---

## C. Phân tích page budget (5 trang ICIIT)

Hiện tại main.tex build ra **5 trang** (confirmed từ build log phiên bản trước). Sau khi thêm 51 dòng (+3,200 bytes) và đặc biệt thêm Eq. (4) STAB và đoạn giải thích §3.1, tổng page count có thể đã tăng.

**Phần chiếm space lớn nhất:**
- `table*` Related Work: ~0.7–0.8 trang
- `figure*` placeholder (28mm): ~0.4 trang (sẽ được thay bằng kết quả thực)
- `table*` Policy/Validation: ~0.7–0.8 trang
- §3.1 + Eq. (1)–(4): ~0.6 trang (tăng so với v1)

> [!WARNING]
> Sau khi thêm §3.1 STAB equation và đoạn giải thích, cần build lại và kiểm tra page count. Nếu vượt 5 trang, phần Eq.(4) và đoạn giải thích dài về artifact field naming (L271–274) là ứng cử viên để cắt bớt trước.

---

## D. Tổng hợp: danh sách việc cần làm

### Ưu tiên cao — sửa trước khi data freeze

| # | Dòng | Vấn đề | Hành động |
|---|---|---|---|
| H1 | L94 | "caregiver fragmentation" không có trong §3.1 definition | Thống nhất: dùng "continuity-of-care penalty" hoặc "fragmentation penalty" nhất quán |
| H2 | L280–284 | `|P|` và `P ≤ 0` ký hiệu phức tạp, dễ gây nhầm | Đổi thành `p > 0` là scalar, bỏ ký hiệu `|P|` |
| H3 | L265–268 | STAB equation: cần verify "stability reward" là tên chính thức trong Unceta et al. | Đọc lại §4 của Unceta et al. 2024, nếu tên khác thì cite rõ |
| H4 | L377 | "projected assignments" chưa được định nghĩa | Thay bằng thuật ngữ phổ thông hơn |
| H5 | BibTeX | `ErrarhoutEtAl2016` vẫn `@article` | Đổi sang `@inproceedings` |

### Ưu tiên vừa — sửa trước submission

| # | Dòng | Vấn đề | Hành động |
|---|---|---|---|
| M1 | L138–139 | Khoảng cách thừa sau `MaxSAT.\@` | Gộp `A` lên cùng dòng |
| M2 | L88 | `\shortauthors` với 3 tác giả | Kiểm tra ICIIT format, có thể đổi thành "Kieu, Do, and To" |
| M3 | L216 | Row "Constraint strengthening" không khớp với cite | Đổi label thành "Implied constraints and symmetry breaking" |
| M4 | L268 | `\label{eq:stability-continuity}` không được ref | Thêm ref trong text hoặc xóa label |
| M5 | L437–440 | "gated publication phases" không được định nghĩa | Thêm 1 câu giải thích hoặc break-down |
| M6 | L590–591 | EvalMaxSAT sentence quá specific | Chuyển vào footnote |

### Ưu tiên thấp — polish cuối

| # | Dòng | Vấn đề | Hành động |
|---|---|---|---|
| L1 | L48–52 | `\resultplaceholder` bỏ qua tham số `#1` | Thêm comment giải thích hoặc đổi thành 0-argument |
| L2 | L451 | Warm-up instances source chưa nêu | Thêm câu note về exclusion |
| L3 | — | Page count sau edits chưa được verify | Chạy `latexmk` lại, kiểm tra |
| L4 | BibTeX | `BailleuxBoufkhad2004` thiếu `pages` field | Thêm `pages = {1--6}` (hoặc verify từ HAL) |
| L5 | BibTeX | `JahrenAsin2018` author rendering | Verify từ publisher record |

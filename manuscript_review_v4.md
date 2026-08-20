# Rà soát lần 4: main.tex v4 (526 dòng) — Sau tái cấu trúc kiến trúc lớn

> **Thời điểm:** 2026-08-10T15:24 +07:00  
> **So với v3:** 634 dòng → 526 dòng (−108 dòng)  
> **Build main.pdf:** ✅ **4 trang**, 480,580 bytes — không Overfull, 2 Underfull \\vbox nhỏ  
> **Build review.pdf:** Có sẵn (review.pdf = 489,256 bytes)

---

## A. Thay đổi kiến trúc lớn — Đánh giá tổng quan

### A1. Tách thành main.tex + review.tex ✅ (Cải tiến quan trọng)

**Trước (v3):** `\ifinternaloutlinetrue` hard-coded, tất cả nội dung editorial notes trong một file.

**Sau (v4):**
- `main.tex`: Build sạch **không có** notes editorial hoặc placeholders.
- `review.tex` (5 dòng): Chỉ define `\HCORAPInternalReview` rồi `\input{main.tex}`.
- Logic switch: `\ifdefined\HCORAPInternalReview` thay cho manual true/false.

**Ưu điểm lớn:**
- Không còn nguy cơ submit `main.pdf` với `\internaloutlinetrue` do quên sửa.
- Collaborator build `review.tex` để xem notes, nhưng `main.tex` mặc định là "clean".

### A2. Frozen-results gate ✅ (Cải tiến kiến trúc)

**Dòng 50–55:**
```latex
\newif\iffrozenresults
\IfFileExists{generated/abstract-findings.tex}{%
  \IfFileExists{generated/results.tex}{%
    \IfFileExists{generated/conclusion.tex}{\frozenresultstrue}{\frozenresultsfalse}%
  }{\frozenresultsfalse}%
}{\frozenresultsfalse}
```

**Tốt:** Cơ chế này yêu cầu cả 3 file phải tồn tại đồng thời — tránh "half-frozen" state. `generated/` chỉ có `README.md`, tức là hiện tại `\iffrozenresults = false`, mọi frozen content bị skip.

**Vấn đề tinh tế:**
- Abstract (L119–125): `\iffrozenresults … \input{generated/abstract-findings} … \else … \outlineblock{...} \fi` — Khi `\iffrozenresults = false` và `\ifinternaloutline = false` (tức là build main.pdf trước data freeze), `\outlineblock` **không render** (vì `\ifinternaloutline = false`). Điều này có nghĩa là **abstract hiện tại kết thúc ở câu "Hash-verified artifacts retain..."** mà không có findings nào. **Đây là intentional design** (pre-data-freeze), nhưng abstract đang thiếu findings sentence khi build main.pdf.

> [!CAUTION]
> Khi build `main.pdf` trước data freeze, abstract kết thúc sau "...optimal-objective checks against Gurobi and CPLEX. Hash-verified artifacts retain code, instances, configurations, and raw outputs." — không có findings, không có outline note. Abstract này **không complete** về mặt nội dung nhưng sẽ không báo lỗi. Cần đảm bảo reviewer nội bộ hiểu rằng phải dùng `review.pdf` để xem trạng thái thực.

### A3. Results section tách ra `internal-results-outline.tex` ✅

**Dòng 460–467:**
```latex
\iffrozenresults
  \input{generated/results}
\else
  \ifinternaloutline
    \input{internal-results-outline}
  \fi
\fi
```

Khi build `main.pdf` (trước data freeze): **Results section không tồn tại**. Khi build `review.pdf`: hiện `internal-results-outline.tex`. Sau data freeze: hiện `generated/results.tex`.

**Ưu điểm:** main.pdf không có Results placeholder nào.
**Hệ quả:** main.pdf hiện chỉ có 4 trang thay vì 5 — nhưng sẽ tăng lên khi frozen results được inject.

### A4. Conclusion section tương tự ✅

**Dòng 504–515:** Tương tự, Conclusion chỉ hiện trong `review.pdf` hoặc sau data freeze.

### A5. Tên tiêu đề đã thay đổi

**v3:** "Lexicographic Optimization for MaxSAT-Based Home-Care Resource Allocation: Totalizer Encoding, Implied Constraints, and Symmetry Breaking"  
**v4:** "Lexicographic MaxSAT for Home-Care Resource Allocation: Encoding and Constraint Strengthening"

Tiêu đề mới **ngắn hơn** và có subtitle ngắn gọn hơn. Tuy nhiên:
- Running head: "Lexicographic MaxSAT for HCORAP" ✅ phù hợp.
- Subtitle "Encoding and Constraint Strengthening" — **mơ hồ hơn**: không nêu rõ "Totalizer" (encoding cụ thể) hay "symmetry breaking". Tùy thuộc vào ICIIT page limit, subtitle ngắn có thể là trade-off hợp lý.
- **Dòng 84:** `Lexicographic MaxSAT for Home-Care Resource Allocation:\\ Encoding and Constraint Strengthening` — dấu `\\` tái xuất hiện (v2 đã sửa, v4 đưa lại). Với `\title{}` của acmart, `\\` trong title có thể gây extra space ở PDF. Cần kiểm tra PDF output.

### A6. Pseudocode → Equation (lex-stages) — Thay đổi quan trọng

**v3 (pseudocode 8 bước trong minipage):** Được thay bằng:

**v4 (L303–318):**
```latex
Let (f_1,f_2,f_3) = (CONT, OT, -SIM) and let F_0 be the schedules
satisfying all hard constraints.  For i=1,2,3, the staged solver constructs
\begin{equation}
  z_i = \min_{x\in F_{i-1}} f_i(x),\qquad
  F_i = \{x\in F_{i-1}: f_i(x)=z_i\}.
  \label{eq:lex-stages}
\end{equation}
Each stage rebuilds the WCNF over F_{i-1}, receives only the unused part of
one cumulative timeout, and passes its assignment to the independent verifier.
The next stage is entered only after Open-WBO reports Optimum; the
value z_i is then fixed by an exact equality.  Thus F_i contains precisely
the schedules attaining prefix (z_1,...,z_i), and a verified assignment in
F_3 is lexicographically optimal for the encoded model.  Any incomplete stage
is a policy timeout, not an optimum.
```

**Đánh giá:**
- Compact hơn nhiều so với pseudocode 8 bước — phù hợp với bài 5 trang.
- Equation (lex-stages) tốt về mặt toán học và rõ ràng.
- Câu giải thích sau equation đủ để understand implementation.

**Vấn đề mới:**
- Equation (lex-stages) có `F_i = {x ∈ F_{i-1} : f_i(x) = z_i}`. Điều này định nghĩa F_i **sau khi z_i đã được tìm** — đây là toán học OK. Nhưng "The next stage is entered only after Open-WBO reports Optimum" — câu này nhắc đến Open-WBO cụ thể, trong khi equation trình bày ở mức abstract. Sự hỗn hợp abstraction level này không sai nhưng có thể gây bối rối: reviewer có thể hỏi "what if a different solver is used?"

- **"a verified assignment in F_3 is lexicographically optimal for the encoded model"** — câu này chính xác nhưng quan trọng: tối ưu cho "encoded model", không nhất thiết là tối ưu toàn cục nếu encoding không hoàn toàn tương đương với problem. Cách nói này an toàn và đúng.

### A7. Thay đổi §5.1: campaign run count

**v3 (L441–445):**
```
The measured base contains 3,856 runs: 1,856 screens, 1,600 weighted B0 runs,
and 400 MIP runs.  The original lexicographic gate adds 820 (560 LEX-COS,
160 LEX-OCS, and 100 MaxSAT validation), and corrected-v2 adds 320, reaching
the 4,996-run maximum.
```

**v4 (L427–431):**
```
The 3,856-run base contains 1,280 factorial, 256 corrected-v2 exploratory, 320
lexicographic-scalability, 1,600 weighted-B0, and 400 MIP runs.  The original
lexicographic and corrected-v2 gates add at most 820 and 320 runs, respectively,
giving 4,996 maximum.
```

**Cải tiến:** Đã làm rõ "1,856 screens" = 1,280 factorial + 256 + 320. ✅ Phân loại "1,280 factorial" tách riêng khỏi "screens". Vấn đề H1 từ review v3 đã được khắc phục.

**Kiểm tra phép tính v4:**
- 1,280 + 256 + 320 + 1,600 + 400 = **3,856** ✅
- +820 (lexicographic gate) = 4,676
- +320 (corrected-v2 gate) = **4,996** ✅

**Vấn đề nhỏ mới:** "256 corrected-v2 exploratory" — từ "corrected-v2 exploratory" ngụ ý là cả 256 runs đều thuộc corrected-v2. Nhưng submission_plan.md §4 ghi "two 128-run exploratory screens" — chưa rõ cả hai đều là corrected-v2 hay có screen khác. Nếu đúng thì OK; nếu có screen trên benchmark khác thì cần nêu rõ.

---

## B. Rà soát từng section của v4

### B1. Abstract (L107–126) — Cấu trúc mới với frozen-results gate

**Nội dung (trước data freeze, build main.pdf):**
```
Home-care resource allocation...
The existing MaxSAT formulation uses a weighted aggregate that can conceal
trade-offs among caregiver fragmentation, excess workload, and assignment
similarity.  We introduce LEX-COS, an exact staged policy that optimizes these
criteria in that priority order.  We combine it with bidirectional Totalizer
encodings, logically implied constraints, and symmetry breaking for detected
slot and service equivalences.  A pre-specified paired campaign separates
objective-policy effects from encoding interactions on the original and
corrected-v2 benchmarks.  It uses cumulative stage budgets, independent
solution verification, and optimal-objective checks against Gurobi and CPLEX.
[outlineblock - invisible in main.pdf]
```

**Đánh giá nội dung:**
- **"optimizes these criteria in that priority order"** — Cách diễn đạt mới, ngắn gọn hơn nhưng **mất đi thông tin về lexicographic**. "Optimizes in priority order" có thể bị hiểu là weighted sum với priority weights, không nhất thiết là strict lexicographic. Nên thêm từ "lexicographically": "optimizes these criteria **lexicographically** in that priority order."
- "caregiver fragmentation, excess workload, and assignment similarity" — đây là 3 criteria tương ứng CONT, OT, SIM. Thứ tự liệt kê (fragmentation, workload, similarity) khớp với thứ tự LEX-COS (CONT, OT, SIM). ✅
- Abstract không còn câu "The design separates objective-policy effects from encoding interactions and treats routing and uncertainty as separate model extensions." — câu này đã bị xóa so với v3. Đây là thông tin quan trọng về scope và nên được giữ lại.

**Vấn đề còn tồn tại (từ review v2):**
- Hai tên cho CONT: "caregiver fragmentation" (L111) trong abstract mà không có từ nào khớp rõ với "caregiver-fragmentation penalty" trong §3.1. Tuy nhiên bây giờ abstract không nhắc đến "continuity-of-care penalty" nữa → **vấn đề B2 của review v3 đã được giải quyết một phần** vì abstract chỉ còn dùng "caregiver fragmentation". ✅

### B2. Introduction (L133–178) — Vẫn còn khoảng cách thừa

**Dòng 145–146:**
```latex
Unceta et al.~\cite{UncetaEtAl2024} encode HCORAP as weighted partial MaxSAT.\@
A
weighted scalar score is useful...
```

**Vấn đề này tồn tại từ v1 và chưa được sửa qua 3 lần chỉnh sửa.** Dòng 146 bắt đầu bằng chữ "A" đơn độc — LaTeX sẽ tạo khoảng trắng thừa khi compile. Nên gộp:

```latex
Unceta et al.~\cite{UncetaEtAl2024} encode HCORAP as weighted partial MaxSAT.\@ A
weighted scalar score is useful...
```

### B3. §3.1 Model — Thay đổi nhỏ quan trọng

**v3 (L253–255):**
```
For a continuity set q, let S_q be its services and let n_q=...
```

**v4 (L258–260):**
```
Let S_u and S_q be the services of user u and continuity set q, respectively,
and let n_q=...
```

**Tốt:** Định nghĩa `S_u` trước khi dùng trong §4 Implied Constraints (L352). Cải thiện rõ ràng. ✅

### B4. §3.2 — Phương trình B0 và w_c, w_o

**v4 (L286–293):**
```latex
With p=|P| denoting the non-negative magnitude of the per-instance signed
penalty field P≤0, policy B0 maximizes
\begin{equation}
  \mathrm{SIM}-w_c\mathrm{CONT}-w_o\,p\,\mathrm{OT},
  \label{eq:weighted}
\end{equation}
where w_c,w_o\in\mathbb Z_{\geq0} multiply CONT and OT; the primary
experiments use (w_c,w_o)=(1,1).
```

**Cải tiến từ v3:** `w_c,w_o\in\mathbb Z_{\geq0}` — đã định nghĩa domain. ✅ Vấn đề H3 từ review v3 đã được khắc phục.

**Vấn đề còn lại:** `w_c,w_o\in\mathbb Z_{\geq0}` — Tại sao phải là số nguyên không âm? Trong thực tế, weights thường là số thực dương (rational). Nếu đây là constraint thực sự của implementation (ví dụ WCNF chỉ hỗ trợ integer weights), thì cần giải thích lý do. Nếu không có lý do kỹ thuật, đổi thành `w_c,w_o\geq0` (real-valued) sẽ tổng quát hơn.

### B5. §3.2 — Equation (lex-stages) thay thế pseudocode

Đã đánh giá ở A6. Thêm một vấn đề:

**Dòng 313–314:**
```
The next stage is entered only after Open-WBO reports Optimum; the
value z_i is then fixed by an exact equality.
```

"Fixed by an exact equality" — trong WCNF, cố định $z_i$ bằng cách thêm hard clause `f_i(x) = z_i` vào $F_i$. Điều này cần thêm biến auxiliary hoặc constraint tùy encoding. Với reviewer MaxSAT, cụm "exact equality" có thể gây hỏi về cách implement: equality trên cardinality encoding không trivial. Nên thêm một câu giải thích kỹ thuật ngắn hoặc reference implementation. Hoặc đơn giản hóa: "encoded as an exact cardinality bound in the next stage's WCNF."

### B6. §4.2 Implied Constraints — Kiểm tra scope $\mathcal{S}_u$

**Dòng 351–352:**
```
v_{uh} ↔ ∨_{s∈S_u} t_{sh}
```

`S_u` bây giờ đã được định nghĩa tại §3.1 (L258). ✅ Vấn đề M của review v3 đã được giải quyết.

### B7. §4.3 Symmetry Breaking — Value-precedence clauses đã được định nghĩa

**v4 (L384–388):**
```latex
For consecutive equivalent values, \mathit{earlier}_i and
\mathit{later}_j denote assigning the earlier and later value to ordered
positions i and j; we impose
\mathit{later}_j\rightarrow\bigvee_{i<j}\mathit{earlier}_i.  Positions are
services for slot classes and ordered caregiver--slot pairs for service classes.
```

**Cải tiến từ v3:** Đã định nghĩa `earlier_i` và `later_j` rõ ràng. ✅ Vấn đề M3 từ review v3 đã được khắc phục.

**Vấn đề còn lại:** Ký hiệu này vẫn không standard. `earlier_i` = "assigning the earlier value to position i" — nhưng "value" là gì trong context slot grouping? Đây không phải là value trong sense CSP truyền thống. Với MaxSAT, clause này tương đương với: "if slot $j$ (or service $j$) is used, then some slot $i < j$ (or service $i < j$) must also be used". Cách diễn đạt hiện tại **đủ để hiểu** trong context của bài nhưng không chuẩn xác hoàn toàn về thuật ngữ SAT.

### B8. §5.1 — Configurations

**v4 (L404–412):**
```
Baseline configuration B uses sorting networks without implied constraints
or symmetry breaking.  Reference configuration R combines Totalizer,
both implied-constraint families, and symmetry breaking for detected
slot and service equivalences; the label does not assume a speedup.
```

**Tốt:** Bỏ "pre-specify unstrengthened" (v3) — tự nhiên hơn. "Reference configuration" thay vì "Composite" — đúng hơn về tone. ✅

**Vấn đề:**
- **"both implied-constraint families"** — ở đây dùng "families" nhưng trong §4.2 dùng "two logically implied constraints". Cần nhất quán: hoặc dùng "two implied-constraint families" hoặc "the `both` implied-constraint families" (với backtick).

### B9. §5.1 — Conditional LEX-COS gate mới

**v4 (L415–416):**
```
Conditional on a pre-specified scalability gate, LEX-COS uses 280 held-out
instances from 14 original classes...
```

**Đây là thay đổi quan trọng:** v3 không đề cập rõ "scalability gate" cho LEX-COS. v4 thêm "Conditional on a pre-specified scalability gate" — nhưng gate này **không được định nghĩa** trong §5 hoặc ở bất cứ đâu trong main.tex. Reviewer sẽ hỏi: gate này là gì? Threshold nào? Khi nào được check?

> [!CAUTION]
> "Scalability gate" được nhắc đến (L415) nhưng không định nghĩa trong bài. Nếu đây là gate quantitative (ví dụ: "≥50% of calibration runs reach OPTIMUM"), cần nêu ngưỡng cụ thể. Nếu là qualitative judgment, nên nói rõ ai quyết định và theo tiêu chí gì.

### B10. §5.2 Protocol — Cải tiến rõ nét

**v4 (L435–458):** Gọn hơn nhiều so với v3 (L447–480). Loại bỏ các chi tiết dư thừa.

**Vấn đề còn lại:**
- **"Ten fixed, non-evaluation corrected-v2 instances warm the VM"** (L440–441) — "non-evaluation" là thuật ngữ tự đặt. Nên nói rõ: "Ten fixed corrected-v2 instances from the disjoint calibration set (not the held-out evaluation set) warm the VM."
- **Commit hash rút ngắn:** v3 dùng hash đầy đủ 40 ký tự `80f3073e41028b219b0b0ad7c61fba28351f88e6`; v4 (L438) chỉ dùng 12 ký tự `80f3073e4102`. Với reproducibility, nên dùng hash đầy đủ. Comment "the artifact records the full source and binary hashes" (L438) giải quyết một phần nhưng hash ngắn trong text vẫn dễ gây nhầm nếu reviewer muốn verify.

---

## C. Rà soát internal-results-outline.tex

### C1. Bảng Panel B — Gộp Corrected-v2 rows

**v3 (Panel B, 2 rows):**
```
Corrected-v2, R, weighted: 160 runs
Corrected-v2, R, LEX-COS:  160 runs
```

**v4 (internal-results-outline.tex L58, 1 row):**
```
Corrected-v2, R, B0 / LEX-COS: 320 runs
```

**Cải tiến:** Gộp thành 1 row với 320 runs (= 160×2 policies). Nhưng "stage counts" placeholder (L58 và L59) bây giờ cần phân biệt cho 2 policies trong 1 cell — đây có thể khó trình bày trong bảng. Nên xem xét split lại khi có dữ liệu thực.

### C2. "Both opt." trong Panel A (L49 của outline file)

**v4 Panel A header:** `Both opt.` (rút gọn từ "Both OPTIMUM" của v3).

Trong sigconf với `\scriptsize`, space là vấn đề — rút gọn là hợp lý. Nhưng "opt." mơ hồ: optimal hay optimum? Nên giải thích trong caption hoặc dùng "OPTIMUM" như trong text.

---

## D. Rà soát BibTeX (lần 4)

| Entry | Thay đổi | Trạng thái |
|---|---|---|
| `ErrarhoutEtAl2016` | **Không thay đổi** — vẫn `@article` | ⚠️ Vẫn cần sửa thành `@inproceedings` |
| `BailleuxBoufkhad2004` | **Đã thêm** `pages = {263--268}` | ✅ Sửa từ review v2 |
| Tất cả entries khác | Không thay đổi | ✅ |

---

## E. Tổng hợp: so sánh tiến độ qua 4 lần rà soát

| Vấn đề | Review v1 | Review v2 | Review v3 | Review v4 |
|---|---|---|---|---|
| `\resultplaceholder` render `#1` | ❌ | ✅ | ✅ | ✅ |
| `|P|` ký hiệu phức tạp | ❌ | ✅ | ✅ | ✅ |
| `w_c, w_o` không được định nghĩa | ❌ | ❌ | ✅ | ✅ |
| Campaign math mâu thuẫn | ❌ | ❌ | ❌ | ✅ |
| "1,856 screens" bao gồm factorial | — | — | ❌ | ✅ |
| BibTeX `BailleuxBoufkhad2004` thiếu pages | ❌ | ❌ | ❌ | ✅ |
| Định nghĩa `S_u` trước §4 | ❌ | ❌ | ❌ | ✅ |
| `earlier_i / later_j` không định nghĩa | ❌ | ❌ | ❌ | ✅ |
| Dư thừa editorial notes trong main.pdf | ❌ | ❌ | ❌ | ✅ (kiến trúc mới) |
| Frozen-results gate | — | — | — | ✅ (mới) |
| `ErrarhoutEtAl2016` vẫn `@article` | ❌ | ❌ | ❌ | ❌ |
| Khoảng cách thừa L145–146 | ❌ | ❌ | ❌ | ❌ |
| "scalability gate" không định nghĩa | — | — | — | ❌ (mới) |
| `w_c,w_o ∈ Z≥0` có thể quá hạn chế | — | — | — | ❌ (mới) |
| `\\` trong title (dòng 84) | ✅ v1 sửa | — | — | ❌ (tái xuất hiện) |
| Abstract thiếu "lexicographically" | — | — | — | ❌ (mới) |
| Abstract thiếu scope statement | — | — | — | ❌ (mới) |

---

## F. Danh sách vấn đề còn lại — Phân loại ưu tiên

### Ưu tiên cao (ảnh hưởng correctness/clarity)

| # | Dòng | Vấn đề | Hành động |
|---|---|---|---|
| H1 | L84 | `\\` tái xuất hiện trong title | Xóa `\\`, để acmart tự wrap |
| H2 | L112 | "optimizes these criteria in that priority order" thiếu "lexicographically" | Thêm từ "lexicographically": "**lexicographically** optimizes these criteria" |
| H3 | L115–116 | Abstract bỏ câu về scope (routing/uncertainty exclusion) | Thêm lại: "treating routing and uncertainty as separate model extensions" |
| H4 | L415 | "pre-specified scalability gate" không định nghĩa | Định nghĩa gate: ngưỡng threshold cụ thể, ví dụ "if ≥X% of the 320-run scalability screen reach OPTIMUM within 300 s" |
| H5 | BibTeX L26 | `ErrarhoutEtAl2016` vẫn `@article` (IFAC conference) | Đổi thành `@inproceedings` |

### Ưu tiên vừa

| # | Dòng | Vấn đề | Hành động |
|---|---|---|---|
| M1 | L145–146 | Khoảng cách thừa sau `MaxSAT.\@` + `A` đơn độc | Gộp lại một dòng |
| M2 | L292 | `w_c,w_o∈Z≥0` có thể quá hạn chế | Đổi thành `w_c,w_o≥0` (real-valued) hoặc giải thích tại sao integer |
| M3 | L313–314 | "fixed by an exact equality" không giải thích cách implement | Thêm: "encoded as an exact cardinality bound in the next stage's WCNF" |
| M4 | L406 | "both implied-constraint families" không nhất quán với §4.2 | Đổi thành "the `\\texttt{both}` implied-constraint treatment" |
| M5 | L427 | "256 corrected-v2 exploratory" — cần verify cả 256 đều là corrected-v2 | Confirm với submission_plan |
| M6 | L438 | Commit hash rút ngắn còn 12 ký tự | Dùng hash đầy đủ 40 ký tự |
| M7 | L440–441 | "non-evaluation" mơ hồ | Đổi thành "from the disjoint calibration set" |

### Ưu tiên thấp (polish)

| # | Vị trí | Vấn đề |
|---|---|---|
| L1 | internal-results-outline.tex L49 | "Both opt." — nên giải thích trong caption |
| L2 | internal-results-outline.tex L58 | 1 row cho 320 corrected-v2 runs có thể khó trình bày |
| L3 | main.tex L119–125 | abstract-findings placeholder chỉ visible trong review.pdf — đảm bảo collaborators biết |

---

## G. Đánh giá kiến trúc tổng thể

Đây là **phiên bản tốt nhất** trong 4 lần rà soát. Các cải tiến kiến trúc (main/review split, frozen-results gate) là thiết kế đúng đắn và tránh được nhiều lỗi submission. Nội dung khoa học đã được tinh chỉnh tốt qua từng vòng. 

Chỉ còn **5 vấn đề ưu tiên cao** cần sửa trước khi submit, tất cả đều nhỏ và không ảnh hưởng đến correctness của method.

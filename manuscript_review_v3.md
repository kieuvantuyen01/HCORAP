# Rà soát lần 3: main.tex v3 (634 dòng) — Sau chỉnh sửa lần 2

> **Thời điểm:** 2026-08-09T21:48 +07:00  
> **So với v2:** 630 dòng → 634 dòng (+4 dòng)  
> **Build:** ✅ 5 trang, 500,599 bytes — không có Overfull, không có lỗi biên dịch  
> **Warnings build:** 4 Underfull \\vbox (cosmetic), `\balance` in second column (cosmetic với sigconf)

---

## A. Thay đổi từ v2 → v3: danh sách đầy đủ

### A1. `\resultplaceholder` — Đã sửa ✅

**v2 (L48–54):**
```latex
\newcommand{\resultplaceholder}[1]{%
  \ifinternaloutline
    \textcolor{red}{[pending]}%   ← bỏ qua #1
  \fi
}
```

**v3 (L48–54):**
```latex
% The argument identifies the expected frozen result (for example, n, x, or a
% generated visual) so that collaborators can distinguish pending artifacts.
\newcommand{\resultplaceholder}[1]{%
  \ifinternaloutline
    \textcolor{red}{[#1]}%   ← hiện tham số
  \fi
}
```

**Tốt.** Giờ mỗi `\resultplaceholder{n}` sẽ render thành `[n]` và `\resultplaceholder{x}` thành `[x]`. Collaborator nhìn vào bảng PDF có thể phân biệt được placeholder nào là số đếm (`n`) và placeholder nào là giá trị (`x`). Comment giải thích cũng được thêm vào — chuẩn.

### A2. Giải thích `|P|` — Đã sửa ✅

**v2 (L280–284):**
```latex
where P\leq0 is the signed excess-workload penalty stored in the benchmark,
so |P| is its non-negative magnitude per excess-workload unit.
```

**v3 (L281–287):**
```latex
With $p=|P|$ denoting the non-negative magnitude of the per-instance signed
penalty field $P\leq0$, policy B0 maximizes
\begin{equation}
  \mathrm{SIM}-w_c\mathrm{CONT}-w_o\,p\,\mathrm{OT},
```

**Tốt.** Ký hiệu `p` scalar được định nghĩa rõ ràng. Phương trình sạch hơn. Tuy nhiên — xem §B1.

### A3. Campaign run count — Đã sửa ✅

**v2 (L437–445):**
```
The maximum manifest contains 4,996 measured runs: 1,280 factorial, two
128-run exploratory screens, a 320-run lexicographic scalability screen, and
the gated publication phases.  Screening records the exact selected total,
between 3,856 and 4,996.
```

**v3 (L441–445):**
```
The measured base contains 3,856 runs: 1,856 screens, 1,600 weighted B0 runs,
and 400 MIP runs.  The original lexicographic gate adds 820 (560 LEX-COS,
160 LEX-OCS, and 100 MaxSAT validation), and corrected-v2 adds 320, reaching
the 4,996-run maximum.
```

**Rất tốt.** Breakdown rõ ràng cho phép verify từng con số.

### A4. Related Work table row label — Đã sửa ✅

**v2 (L216):** `Constraint strengthening~\cite{BofillEtAl2022,BogaertsEtAl2022}`  
**v3 (L218):** `Implied constraints and symmetry breaking~\cite{BofillEtAl2022,BogaertsEtAl2022}`

Đúng hơn. ✅

### A5. `\label{eq:stability-continuity}` — Đã được ref ✅

**v3 (L272):** `Equation~\eqref{eq:stability-continuity} shows that...` — label được tham chiếu đúng.

### A6. Warm-up instances — Đã giải thích ✅

**v2 (L451–452):** "Ten unmeasured instances warm the machine."  
**v3 (L454–456):** "Ten fixed corrected-v2 calibration instances warm the machine; they are disjoint from held-out evaluation and unmeasured."

Đã rõ nguồn gốc warm-up instances. ✅

### A7. `both-plus` — Đã giải thích đủ ✅

**v2 (L377–379):** "excluded because it also changes projected assignments, effective workload caps, and service-slot constraints"  
**v3 (L379–382):** "excluded because, beyond `both`, it adds equivalent service-level projection, workload-cap tightening, and service-slot clustering; all preserve feasible schedules."

Đã xóa thuật ngữ "projected assignments" mơ hồ, thay bằng mô tả cụ thể hơn. Và câu "all preserve feasible schedules" — tốt vì clarify rằng `both-plus` cũng an toàn, chỉ bị loại khỏi factorial vì bundle quá nhiều factors. ✅

---

## B. Vấn đề còn lại sau v3

### B1 · Phương trình B0 thiếu dấu chấm hỏi về where-clause [QUAN TRỌNG]

**Dòng 281–287:**
```latex
With $p=|P|$ denoting the non-negative magnitude of the per-instance signed
penalty field $P\leq0$, policy B0 maximizes
\begin{equation}
  \mathrm{SIM}-w_c\mathrm{CONT}-w_o\,p\,\mathrm{OT},
  \label{eq:weighted}
\end{equation}
The primary experiments use $(w_c,w_o)=(1,1)$.
```

**Vấn đề kỹ thuật ký hiệu:**

Câu giới thiệu trước phương trình kết thúc bằng "policy B0 maximizes" và phương trình (2) kết thúc bằng dấu phẩy `,`. Sau dấu phẩy, câu tiếp theo "The primary experiments use..." bắt đầu bằng chữ hoa "The" — **điều này ngầm hiểu là dấu phẩy ở cuối equation là dấu phẩy của câu văn** (mathematical punctuation), tức là phương trình là một phần của câu văn dài:

> "policy B0 maximizes [equation], [but what follows?]"

Thông thường sau dấu phẩy nên có một mệnh đề phụ giải thích (ví dụ: "where $p$ and $w_c, w_o$ are..."). Hiện tại `p` được giới thiệu trong câu trước phương trình (OK), nhưng `w_c` và `w_o` **không được giải thích** ở đây. Chỉ đến câu sau mới nói "The primary experiments use $(w_c,w_o)=(1,1)$" — điều này chứng minh chúng là weight parameters, nhưng không nói rõ domain và ý nghĩa.

> **Đề xuất:** Đổi phương trình thành kết thúc bằng dấu phẩy + thêm where-clause:
> ```latex
> \mathrm{SIM}-w_c\mathrm{CONT}-w_o\,p\,\mathrm{OT},
> ```
> Thêm sau equation: "where $w_c\geq0$ and $w_o\geq0$ are relative weight parameters; the primary experiments use $(w_c,w_o)=(1,1)$."

### B2 · "caregiver fragmentation" vs "continuity-of-care penalty" — VẪN CÒN MÂU THUẪN NHỎ

**Abstract (L96):** "caregiver fragmentation"  
**§3.1 (L262–263):** "CONT counts the additional distinct caregivers beyond the first in each continuity set and is a caregiver-fragmentation penalty"

Cả hai lần xuất hiện đều dùng "caregiver fragmentation" — vậy **đã nhất quán**. ✅ (Vấn đề này đã được giải quyết bởi việc thêm "caregiver-fragmentation penalty" vào §3.1 definition)

Tuy nhiên, Abstract (L98) còn dùng "continuity-of-care penalty":

> "an exact staged policy that first minimizes the **continuity-of-care penalty**, then excess workload"

→ Abstract dùng 2 cách gọi khác nhau cho CONT: "caregiver fragmentation" (L96) và "continuity-of-care penalty" (L98). Đây là lỗi nhất quán tinh tế. CONT = caregiver-fragmentation penalty = continuity-of-care violation penalty, nhưng hai cách gọi không được giới thiệu như synonyms trong abstract.

> **Đề xuất:** Chọn một cách gọi duy nhất trong abstract, hoặc thêm mệnh đề giải thích: "a caregiver-fragmentation penalty (that is, continuity-of-care violations)".

### B3 · `\label{eq:stability-continuity}` — Equation (4) có vấn đề semantic mới

**Dòng 267–271:**
```latex
\begin{equation}
  \mathrm{STAB}
  = \sum_{q\in\mathcal Q}(|\mathcal S_q|-n_q)
  = \sum_{q\in\mathcal Q}(|\mathcal S_q|-1)-\mathrm{CONT}.
  \label{eq:stability-continuity}
\end{equation}
```

Đây là phương trình mới và quan trọng về conceptual. Tuy nhiên có **vấn đề về phạm vi**: phương trình này chỉ đúng khi $n_q \geq 1$ với mọi $q$ (tức là mỗi continuity set được phục vụ bởi ít nhất một caregiver). Điều này đảm bảo bởi **full coverage** (mỗi service được assign đúng 1 lần).

Câu hiện tại (L264) đã note "Under full coverage" — **đúng**. ✅

Tuy nhiên vẫn còn một vấn đề: bước khai triển thứ hai:
```
∑_q (|S_q| - n_q) = ∑_q (|S_q| - 1) - CONT
```

Điều này đúng **nếu và chỉ nếu** mọi $n_q \geq 1$ (vì CONT = ∑_q max(0, n_q-1) = ∑_q (n_q - 1) khi mọi $n_q \geq 1$). Nhưng câu giải thích ở L272 chỉ nói:

> "Equation (4) shows that maximizing the original stability reward is equivalent to minimizing CONT."

Câu này **đúng** về mặt toán học vì "Under full coverage" đã được ghi ở L264. Nhưng reviewer có thể hỏi: "What if a continuity set has $n_q = 0$?" (tức là không có caregiver nào assign). Câu trả lời là full coverage ngăn điều này, nhưng điều đó nên được link rõ hơn.

> **Đề xuất nhỏ:** Thêm "Under full coverage (guaranteed by the hard constraints above)" thay vì chỉ "Under full coverage".

### B4 · Implied constraints §4.2 — Vấn đề về $t_{sh}$ scope

**Dòng 358–366:**
```
First, let t_{sh} denote that service s uses slot h, and let v_{uh} denote
that user u uses h.  We channel t_{sh} ↔ ∨_a x_{ash} and
v_{uh} ↔ ∨_{s∈S_u} t_{sh}.
```

**Vấn đề logic:** Ký hiệu $\bigvee_{s\in\mathcal S_u} t_{sh}$ — đây là OR over services of user $u$ in slot $h$. Điều này đúng về logic (`v_{uh}` = user $u$ uses slot $h$ = some service of $u$ is in slot $h$). ✅

Nhưng: channeling $v_{uh} \leftrightarrow \bigvee_{s\in\mathcal S_u} t_{sh}$ là **redundant** vì:
- $t_{sh} \leftrightarrow \bigvee_a x_{ash}$ đã được define
- $v_{uh} \leftrightarrow \bigvee_{s\in\mathcal S_u} \bigvee_a x_{ash}$

Điều này tạo ra một chuỗi channeling dài. Đây không phải lỗi mà là design choice cho propagation. OK.

Nhưng có vấn đề **ký hiệu**: $\bigvee_{s\in\mathcal S_u}t_{sh}$ — ký hiệu $\mathcal S_u$ chưa được định nghĩa trong §4, chỉ được định nghĩa trong §3 (dưới dạng services of user $u$). Với bài 5 trang, cross-section notation là bình thường nhưng nên thêm một dòng nhắc: "where $\mathcal S_u$ is the set of services of user $u$ (defined in §3)."

### B5 · Campaign math kiểm tra chi tiết (v3 L441–445)

```
3,856 = 1,856 screens + 1,600 weighted B0 + 400 MIP
```

Kiểm tra:
- 1,856 screens = 2×128 (exploratory) + 320 (lex scalability) + 1,280 (factorial)?
  - 256 + 320 + 1,280 = **1,856** ✅
- 1,600 weighted B0 = 800 instances × 2 configs ($B$ và $R$) = **1,600** ✅
- 400 MIP = 100 instances × 2 solvers × 2 policies = **400** ✅
- Base: 1,856 + 1,600 + 400 = **3,856** ✅

```
Lexicographic gate adds 820:
- 560 LEX-COS
- 160 LEX-OCS
- 100 MaxSAT validation
= 820
```

Kiểm tra:
- 560 LEX-COS = 280 instances × 2 configs = **560** ✅
- 160 LEX-OCS = 80 instances × 2 configs = **160** ✅
- 100 MaxSAT validation = 100 instances × 1 config ($R$) × 1 policy (LEX-COS) = **100** ✅
- Subtotal: 560 + 160 + 100 = **820** ✅

```
Corrected-v2 gate adds 320:
- 160 instances × weighted (R) = 160
- 160 instances × LEX-COS (R) = 160
= 320
```

Kiểm tra:
- 320 = 160 × 2 policies = **320** ✅

**Tổng: 3,856 + 820 + 320 = 4,996** ✅

> [!NOTE]
> Tất cả các con số trong campaign breakdown đều **nhất quán và đúng**. Đây là cải thiện lớn so với v1 chỉ nêu "4,996" mà không giải thích.

**Vấn đề còn lại:** "1,856 screens" nhưng trong text §5.1 (L457–458) còn nói "two exploratory corrected-v2 screens use 60 seconds, the factorial uses 120 seconds, and the lexicographic scalability...". Factorial = 1,280 runs thì **nằm trong screens** theo breakdown L441, nhưng trong §5.2 (L457), factorial và screens được nói như các thực thể riêng biệt với timeout khác nhau. Cần clarify: "1,856 screens" bao gồm factorial (120s) hay không?

- Nếu "1,856 screens" = 256 exploratory + 320 lex scalability + 1,280 factorial thì **tên "screens" không chuẩn** — factorial thường không gọi là "screen".
- Nếu "1,856 screens" = chỉ 256 + 320 = 576, thì 3,856 − 576 − 1,600 − 400 = **1,280 factorial chưa được tính vào đâu**, tức là phép tính trong L441 sai.

> [!CAUTION]
> **Vấn đề ưu tiên cao:** "1,856 screens" bao gồm hay không bao gồm 1,280 factorial cần được làm rõ ngay. Nếu factorial = screen thì cần nói "1,856 screens and factorial runs" để tránh nhầm lẫn. Nếu factorial riêng biệt, phép tính 3,856 = 1,856 + 1,600 + 400 là sai.

### B6 · Value-precedence clause §4.3 — Ký hiệu cần làm rõ thêm

**Dòng 393–394:**
```latex
For consecutive members of a detected equivalence class, value-precedence
clauses enforce $\mathit{later}_j\rightarrow\bigvee_{i<j}\mathit{earlier}_i$.
```

Ký hiệu $\mathit{later}_j$ và $\mathit{earlier}_i$ là tên biến Boolean tự do, không được định nghĩa trước đó. Reviewer sẽ hỏi:
- `later_j` = biến Boolean nào? Liên quan đến $x_{ash}$ hay $y_{as}$ hay gì khác?
- Mệnh đề này có dạng của Value Precedence Constraint (VPC) cổ điển, nhưng biểu diễn phi tiêu chuẩn.

Cách đọc hợp lý: "nếu member thứ $j$ được dùng ở một position nào đó, thì tồn tại member $i < j$ cũng được dùng ở position nhỏ hơn". Nhưng ký hiệu hiện tại không encode điều này — nó chỉ nói "`later_j` kéo theo disjunction của các `earlier_i`", không nói gì về ordering.

> **Đề xuất:** Thêm một câu định nghĩa ngắn: "Here $\mathit{later}_j$ is the Boolean indicator that position $j$'s canonical slot (resp. caregiver--slot pair) is occupied, and the clause requires a preceding position to be occupied first."

### B7 · Threats: "EvalMaxSAT" sentence vẫn còn

**Dòng 594–596:**
```
EvalMaxSAT was excluded when the available candidate binary failed the
official-WCNF smoke test; this avoids invalid cross-solver data but narrows
solver generality.
```

Vẫn chưa được abstract hóa như đề xuất trong review v2. Giữ nguyên nếu muốn traceability đầy đủ (có thể defend khi reviewer hỏi), nhưng nên xem xét đưa vào footnote.

---

## C. Kiểm tra nhất quán thuật ngữ toàn bài (lần 3)

| Thuật ngữ | Dòng xuất hiện | Nhất quán? |
|---|---|---|
| "pre-specified" | L99, L157, L429, L431, L441, L478–479, L523, L526, L571 | ✅ Nhất quán (thay "predeclared" cũ) |
| "OPTIMUM" (all-caps) | L309, L546, L551, L561, L564 | ✅ Nhất quán |
| "caregiver-fragmentation" | L96, L263 | ✅ Nhất quán (đã sửa) |
| "continuity-of-care penalty" | L98 | ⚠️ Dùng song song với "caregiver-fragmentation" trong abstract |
| "excess workload" | L96, L134, L245, L263, L430, L432, L584 | ✅ Nhất quán |
| "corrected-v2" | L102, L430–434, L454, L457 | ✅ Nhất quán |
| "$p$" (penalty scalar) | L281–284 | ✅ Đã định nghĩa rõ |
| "vCPU" | L451 | ✅ Chính xác |
| "Open-WBO 2.1" | L452 | ✅ Nhất quán |
| "lexicographic gate" | L442 | ⚠️ MỚI — chưa được định nghĩa trước đó |
| "corrected-v2 gate" | L443 | ⚠️ MỚI — chưa được định nghĩa trước đó |
| "1,856 screens" | L441 | ⚠️ Không rõ có bao gồm factorial không (xem §B5) |

---

## D. Kiểm tra coverage của 4 contributions vs 3 RQs

**Introduction claims (L152–167):**
1. C1: LEX-COS + LEX-OCS → RQ1 ✅ (§5.1 RQ1)
2. C2: Totalizer reifications + implied constraints → RQ2 ✅ (§5.1 RQ2)
3. C3: Symmetry breaking → RQ3 ✅ (§5.1 RQ3)
4. C4: Pre-specified evaluation → §5.3 Corrected-v2 and cross-solver validation ✅

**Nhưng:** C3 (symmetry breaking) không có RQ riêng — nó được gộp vào RQ3 cùng với C2 (implied constraints). Việc gộp này là OK với bài 5 trang, nhưng reviewer có thể note rằng "RQ3 covers implied constraints **and** symmetry breaking simultaneously" — không thể tách riêng hai effects. Cần xem xét có note này trong Threats không.

---

## E. Kiểm tra page budget chi tiết

Build output: **5 trang, 500,599 bytes**.

Dự kiến phân bổ:
- Trang 1: Title + Abstract + Keywords + Banner + Introduction (một phần)
- Trang 2: Introduction (phần còn) + Related Work + Table 1 (spanning) + §3 mở đầu
- Trang 3: §3 phần còn + §4 + §5 mở đầu
- Trang 4: §5 phần còn + Figure* placeholder + §6
- Trang 5: §6 (Table 2) + §7 Threats + §8 Conclusion + References

Với 5 trang đang dùng, sau khi thay thế placeholders bằng kết quả thực (Figure* và Table*), không gian có thể sẽ bị thu hẹp đáng kể. Cần theo dõi sau data freeze.

---

## F. Tổng hợp: tình trạng sau 3 lần rà soát

### Vấn đề đã giải quyết hoàn toàn (so với v1)

Từ 26 vấn đề ban đầu, **24/26** đã được khắc phục qua 2 lần chỉnh sửa (v1→v2→v3). Hai vấn đề chưa giải quyết: BibTeX `ErrarhoutEtAl2016` và "EvalMaxSAT" sentence.

### Vấn đề cần xử lý ngay (ưu tiên cao)

| # | Dòng | Vấn đề | Hành động |
|---|---|---|---|
| **H1** | L441 | "1,856 screens" — có bao gồm 1,280 factorial không? | Làm rõ: đổi thành "1,856 (1,280 factorial and 576 exploratory and scalability screens)" hoặc giải thích rõ |
| **H2** | L96 vs L98 | Abstract dùng 2 tên khác nhau cho CONT | Chọn 1 cách gọi: "caregiver-fragmentation penalty" hoặc "continuity-of-care penalty" |
| **H3** | L281–286 | $w_c$, $w_o$ chưa được giải thích ý nghĩa | Thêm "where $w_c\geq0$ and $w_o\geq0$ are non-negative relative weights" |
| **H4** | BibTeX | `ErrarhoutEtAl2016` vẫn `@article` | Đổi thành `@inproceedings` |

### Vấn đề ưu tiên vừa

| # | Dòng | Vấn đề | Hành động |
|---|---|---|---|
| M1 | L140–141 | Khoảng trắng thừa: `MaxSAT.\@` trên một dòng, `A` bắt đầu dòng tiếp | Gộp lại thành một dòng |
| M2 | L249 | "Under full coverage" trong STAB equation chưa link đến hard constraints | Thêm "guaranteed by the hard model" |
| M3 | L393–394 | `\mathit{later}_j` và `\mathit{earlier}_i` chưa được định nghĩa | Thêm câu định nghĩa ngắn |
| M4 | L442–443 | "lexicographic gate" và "corrected-v2 gate" xuất hiện lần đầu không giải thích | Thêm "(defined in the Experimental Design section)" hoặc ref rõ |
| M5 | L594–596 | EvalMaxSAT sentence quá specific | Chuyển vào footnote hoặc abstract hóa |
| M6 | BibTeX | `BailleuxBoufkhad2004` thiếu `pages` field | Thêm pages nếu verify được |

### Vấn đề đã ổn — KHÔNG cần sửa thêm

- Campaign math breakdown (3,856 + 820 + 320 = 4,996) ✅
- `\resultplaceholder` macro ✅
- `p=|P|` ký hiệu ✅
- Row label bảng Related Work ✅
- `\label{eq:stability-continuity}` đã được ref ✅
- Warm-up instance source ✅
- `both-plus` giải thích ✅
- Build thành công 5 trang ✅

---

## G. Checklist trước khi data freeze

```
[ ] H1: Làm rõ "1,856 screens" = 1,280 factorial + 576 exploratory
[ ] H2: Thống nhất abstract term cho CONT penalty
[ ] H3: Thêm definition cho w_c, w_o
[ ] H4: Sửa ErrarhoutEtAl2016 trong BibTeX
[ ] M1: Gộp dòng 140–141
[ ] M2: Link STAB equation với hard model guarantee
[ ] M3: Định nghĩa later_j / earlier_i
[ ] M4: Giải thích "gate" terms
[ ] Sau data freeze: xóa \outlineblock, đổi \internaloutlinefalse, điền kết quả thực
[ ] Verify page count sau khi điền kết quả thực (nguy cơ vượt 5 trang)
[ ] Xác nhận ICIIT 2027 format cho \shortauthors với 3 tác giả
[ ] Điền ORCID cho 3 tác giả (deadline 2026-08-31)
[ ] Verify "stability reward" là tên chính thức trong Unceta et al. 2024 §X
```

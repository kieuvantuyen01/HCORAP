# Rà soát chi tiết nội dung main.tex (bản thảo ICIIT 2027)

> **Phiên bản file:** 579 dòng, 30,928 bytes · Build: 5 trang, 3 Underfull \\vbox
> **Trạng thái build:** Không có lỗi LaTeX; không có citation undefined; không có \textcolor{red} vì \resultplaceholder chỉ render khi `\ifinternaloutlinetrue`.

---

## Tổng quan tình trạng so với phiên bản cũ

Bản hiện tại đã **tiến bộ đáng kể** so với outline thuần túy trước đó:
- Abstract **đã có nội dung thực** (không còn chỉ là outline note).
- Introduction **đã có 3 đoạn văn bản đầy đủ** + contributions + RQs.
- Related Work **đã có 2 đoạn văn + bảng taxonomy 8 rows**.
- Problem Formulation **đã có notation, equations, và pseudocode staged LEX-COS**.
- Encoding & Constraint sections **đã có nội dung kỹ thuật thực**.
- Experimental Design **đã có 2 subsections đầy đủ** với số liệu cụ thể.
- Threats to Validity **đã có nội dung đầy đủ**, không còn là outline note.

Phần còn là outline note/placeholder: Results (cả 3 subsections), Conclusion.

---

## 1. Tiêu đề, trang bìa và metadata

### 1.1 Tiêu đề — OK về nội dung, một vấn đề nhỏ

**Dòng 66–68:**
```latex
\title[Lexicographic Optimization for MaxSAT-Based HCORAP]{%
  Lexicographic Optimization for MaxSAT-Based Home-Care Resource Allocation:\
  Totalizer Encoding, Implied Constraints, and Symmetry Breaking}
```

- Running head dùng "HCORAP" (viết tắt) thay vì cụm từ đầy đủ—**hợp lệ** vì running head cần ngắn.
- Dấu `\` (line break) trong tiêu đề dài là cố ý—**OK** với sigconf, nhưng xem preview để đảm bảo không có hyphenation sai.

> [!NOTE]
> Dấu backslash `\` tại cuối dòng 67 tạo `\\` trong title (forced line break). Ở một số ACM template này có thể gây lỗi hoặc extra space ở PDF bìa. Nên dùng `{}` hay đơn giản bỏ dấu xuống dòng đó và để acmart tự wrap.

### 1.2 Tác giả — Thiếu affiliation \department cho Co-author 2 & 3

**Dòng 73–88:** Ba tác giả theo sau bởi một `\affiliation{}` chung. Cách này **đúng về acmart semantics** (shared affiliation group). Tuy nhiên:

- **Thiếu ORCID** cho cả 3 tác giả (comment TODO ở dòng 70–72 đã ghi nhận, deadline 2026-08-31—**giữ nguyên**).
- `\renewcommand{\shortauthors}{Kieu et al.}` ở dòng 88: nếu ≤3 tác giả thì ICIIT thường yêu cầu liệt kê đầy đủ ("Kieu, Do, and To"). Cần kiểm tra yêu cầu cụ thể của ICIIT.

### 1.3 Keywords — Đã cải thiện, một từ cần kiểm tra

**Dòng 110–111:**
```latex
\keywords{home health care scheduling, MaxSAT, lexicographic optimization,
Totalizer encoding, constraint strengthening}
```

- Thay đổi tốt: "home health care scheduling" (phổ biến hơn "home-care resource allocation" trong search).
- "Totalizer encoding" và "constraint strengthening" đã được thêm—**đúng**.
- `constraint strengthening` hơi mơ hồ; `implied constraints` và `symmetry breaking` sẽ specific hơn và khớp với terminology trong paper. Nên xem xét đổi thành `implied constraints, symmetry breaking` thay vì gộp thành `constraint strengthening`.

---

## 2. Abstract — Nội dung tốt, cần chỉnh một số điểm

**Dòng 90–108:**

```
Home-care resource allocation assigns qualified caregivers to requested
services and time slots under coverage, availability, and workload constraints.
The existing MaxSAT formulation uses a weighted aggregate whose scalar optima
can conceal different continuity, overtime, and similarity trade-offs.  We
formulate LEX-COS, an exact staged policy that minimizes continuity violations,
then overtime, and finally maximizes similarity, and study it with bidirectional
Totalizer threshold encodings, implied constraints, and exact symmetry breaking.
A predeclared paired evaluation uses the original and corrected-v2
benchmarks, cumulative timeouts, independent verification, and exact-objective
validation against Gurobi and CPLEX, with artifact-backed provenance.  The
design separates objective-policy effects from encoding interactions and treats
routing and uncertainty as separate model extensions.
```

**Điểm tốt:**
- 5-move structure: bài toán → hạn chế → contributions → protocol → design.
- Không có số liệu (đúng vì chưa data freeze).
- Không claim "state-of-the-art", "robust", "routing-aware"—**đúng**.

**Vấn đề:**

- **Câu thứ 3 quá dài và phức tạp:** "We formulate LEX-COS, an exact staged policy that minimizes continuity violations, then overtime, and finally maximizes similarity, and study it with bidirectional Totalizer threshold encodings, implied constraints, and exact symmetry breaking." → Một câu có 4 vế nối bằng "and". Nên tách thành 2 câu.

- **"bidirectional Totalizer threshold encodings"** — Thuật ngữ "threshold encodings" không xuất hiện ở nơi nào khác trong bài. Trong §4.1 bài dùng "reified threshold vectors" và "full threshold reification". Nên nhất quán; "bidirectional Totalizer encoding" là đủ.

- **"artifact-backed provenance"** — Cụm từ này khó hiểu cho độc giả không biết context. Nên đổi thành "a reproducibility artifact containing source, instances, raw results, and cryptographic hashes".

- **Outline note dòng 104–107** vẫn còn trong abstract và sẽ render màu xanh trong internal build. Đây là đúng design (chỉ xóa sau data freeze), nhưng cần đảm bảo `\ifinternaloutlinetrue` được đổi thành `\internaloutlinefalse` trước submission.

---

## 3. Introduction — Đã có nội dung đầy đủ, chất lượng tốt

**Dòng 123–163:**

### 3.1 Đoạn mở đầu (dòng 126–132)

```
Home-care planning must match requested services with qualified caregivers and
feasible time slots while respecting coverage, availability, concurrency, and
workload limits. ... We study the assignment-and-scheduling scope of the
Home-Care Optimization Resource Allocation Problem (HCORAP): locations may
inform compatibility rewards, but travel times, routes, and visit-to-visit
transitions are not decision variables.
```

**Rất tốt:** Định nghĩa scope sớm và rõ (không routing, không uncertainty). Tuy nhiên:
- "concurrency" — Đây không phải thuật ngữ chuẩn trong HCORAP; model dùng "at most one service per slot" constraint. Nên đổi thành "slot-conflict" hoặc xóa khỏi danh sách.
- Chưa có context xã hội (dân số già, nhu cầu home care tăng). Với ICIIT, một câu motivating context ở đầu sẽ giúp reader nhanh hơn.

### 3.2 Đoạn thứ 2 (dòng 134–143)

```
Unceta et al. cite{UncetaEtAl2024} formulate HCORAP as weighted MaxSAT.@ A
weighted scalar score is useful when its coefficients faithfully encode the
intended exchange rates.  It does not, however, express a strict managerial
priority unless the weights are chosen to dominate every possible change in
lower-priority criteria.
```

**Rất tốt:** Giải thích đúng hạn chế của weighted sum — đây là motivation khoa học chắc chắn nhất trong Introduction.

**Vấn đề nhỏ:** Dòng 134 có `\@` sau dấu chấm (thường dùng để báo LaTeX đây là dấu chấm kết câu, không phải kết từ viết tắt). Điều này **đúng về LaTeX typography** nhưng nếu xóa cũng không sai. Giữ nguyên.

### 3.3 Đoạn contributions và RQs (dòng 145–157)

```
This work makes four contributions.  First... Second... Third... Fourth...
We ask: RQ1... RQ2... RQ3...
```

**Rất tốt:** 4 contributions rõ ràng, 3 RQs được map rõ.

**Vấn đề:**
- "None of these questions presupposes a universal runtime improvement" ở cuối — Câu này hơi defensive và không phải claim khoa học. Nên đổi thành: "We report results in both directions without presupposing a universal runtime improvement."

- **LEX-OCS chưa được giải thích ở đây.** Contribution C1 nhắc đến "LEX-OCS" như một tên riêng, nhưng người đọc chưa biết OCS là gì. Nên thêm một cụm từ: "LEX-OCS (Overtime–Continuity–Similarity order) for sensitivity analysis".

- **Thiếu câu kết nối sang Related Work.** Introduction kết thúc abruptly sau RQs. Với ICIIT 5 trang không cần câu transition dài, nhưng ít nhất một câu "The rest of the paper is organized as follows" sẽ giúp.

---

## 4. Related Work — Cải thiện đáng kể, còn 2 vấn đề

**Dòng 165–212:**

### 4.1 Nội dung text (dòng 168–188)

**Rất tốt:**
- Đã có Demirovic et al. 2019 trong bảng và text (khắc phục vấn đề trong phiên bản cũ).
- Morgado et al. 2014 và Jahren & Asín 2018 đã được thêm vào bảng.
- Câu cuối đoạn Related Work rất chính xác: "Our contribution is not to introduce these general techniques, but to define their HCORAP semantics, verify their safety conditions, and evaluate their interactions..."

**Vấn đề:**
- **ErrarhoutEtAl2016** được cite nhưng BibTeX entry dùng journal "IFAC-PapersOnLine" — đây là venue của hội nghị IFAC (không phải journal thuần túy). ACM format sẽ render như journal. Nên dùng `@inproceedings` thay vì `@article` cho entry này.

### 4.2 Bảng Related Work (dòng 190–212)

```latex
\begin{table*}[t]
  \caption{Methodological scope of representative related-work streams. ...}
```

**Rất tốt:**
- Bảng đã có 7 streams + "This work" = 8 rows — đầy đủ hơn phiên bản cũ (5 rows).
- Demirovic 2019 và Morgado 2014 đã có mặt.
- Caption giải thích "Varies" và "IC/SB"—**tốt**.

**Vấn đề:**
- **Dòng 204:** `MaxSAT staff scheduling~\cite{DemirovicEtAl2019} & Yes & No & No & No & Yes & Varies & No`
  → Cột "A/S" = Yes, nhưng Demirovic et al. làm **staff scheduling**, không phải assignment theo nghĩa HCORAP. Tùy cách định nghĩa "A/S" trong bảng—nếu A/S = "assignment AND scheduling", thì Yes là OK. Nếu cần phân biệt, nên đổi thành cột riêng hoặc thêm footnote.

- **Cột "Cardinality":** Row "Boolean lexicographic and bi-objective MaxSAT [MarquesSilva, Jabs]" = "Varies". Jabs et al. 2024 dùng UNSAT-core approach, không phải cardinality encoding theo nghĩa Totalizer/Sorting network. Cột này có thể misleading—nên đổi thành "N/A" hoặc "Core-guided".

- **`\begin{table*}[t]`:** Bảng này sẽ span cả 2 column trong sigconf double-column. Với 7 columns và font `\footnotesize`, cần kiểm tra trong PDF xem các column có bị cắt không.

---

## 5. Problem Formulation — Chất lượng cao nhất trong bài

**Dòng 214–291:**

### 5.1 Notation (dòng 219–227)

```
Let A, S, H, U, Q denote caregivers, requested services, time slots, users,
and service sequences. Each service consumes one slot. A reward r_as in {0,...,4}
measures suitability...
```

**Rất tốt:**
- Ký hiệu nhất quán: $\mathcal{A}$, $\mathcal{S}$, $\mathcal{H}$, $\mathcal{U}$, $\mathcal{Q}$.
- Định nghĩa feasibility set $E$ và channeling $y_{as}$ rõ ràng.

**Vấn đề nhỏ:**
- "Each service consumes one slot" — Câu này cần ngữ cảnh hơn. Trong mô hình, một service cần chính xác 1 slot, nhưng điều này cần phân biệt với các mô hình khác có thể dùng nhiều slot. Thêm một cụm: "in this discrete-slot model" sẽ rõ hơn.

### 5.2 Hard model (dòng 229–234)

```
The hard model assigns every service exactly once.  At most one service may be
assigned to a caregiver in a slot, and at most one of a user's services may
occupy that slot.  If L_a = sum_s y_{as} is caregiver a's workload, then
L_a <= HN_a + HE_a, where HN_a and HE_a are the regular and extra-hour limits.
Thus full coverage is a hard requirement in every confirmatory experiment.
```

**Rất tốt:** 5 hard constraints được tóm tắt trong 3 câu—đúng scope cho bài 5 trang.

**Vấn đề:**
- **$HE_a$ chưa được định nghĩa ở đây**, chỉ được nhắc tới trong constraint. Mô hình (equations 1–3) không dùng $HE_a$ trong SIM/CONT/OT. Nếu $HE_a$ không xuất hiện trong objectives hay results, có thể bỏ và chỉ nói "workload capacity limit". Nếu giữ thì cần define rõ trong đoạn này.
- Availability constraint không được đề cập (**"$(a,s,h) \in E$"** đã ngầm định availability qua feasible triples $E$). Điều này là OK nhưng nên có footnote hoặc câu ngắn: "Availability and qualification constraints restrict $E$."

### 5.3 Equations (dòng 239–243)

```latex
\mathrm{SIM}  &= \sum_{a}\sum_{s} r_{as}y_{as},
\mathrm{CONT} &= \sum_{q}(n_q-1)^+,
\mathrm{OT}   &= \sum_{a}(L_a-HN_a)^+.
```

**Rất tốt:** Compact, đúng semantic.

**Vấn đề:**
- `\max(0, ...)` đã được dùng trong text (dòng 241–242) nhưng không phải notation chuẩn trong align. Trong LaTeX, nên dùng `(\cdot)^+` hoặc `\max(0,\cdot)` nhất quán.
- CONT định nghĩa ở dòng 241 dùng `n_q` nhưng $n_q$ được định nghĩa ở dòng 236–237: "let $n_q = |\{a : \exists s \in \mathcal{S}_q, y_{as}=1\}|$". **Đây là số agents duy nhất phục vụ sequence $q$.** CONT = số "extra agents" (ngoài 1 agent lý tưởng). Định nghĩa này đúng nhưng có thể gây confusion—reviewer có thể hỏi tại sao không tính theo edges của sequence. Nên thêm 1 câu giải thích: "CONT counts the number of additional caregivers beyond the first, summed over all sequences; a value of zero means each sequence has exactly one caregiver."

### 5.4 Weighted baseline (dòng 247–261)

```
The weighted policy B0 maximizes SIM - w_c*CONT - w_o*|P|*OT,
where P is the per-hour overtime penalty and the primary experiments use
(w_c, w_o)=(1,1).
```

**Vấn đề:**
- **$|P|$ chưa được định nghĩa.** $P$ được gọi là "per-hour overtime penalty" nhưng `|P|` (absolute value of P) là ký hiệu bất thường. Nếu P là một scalar constant, dùng `p` (chữ thường) và viết `p \cdot \mathrm{OT}`. Nếu P là một tập hợp (per-agent penalties), cần giải thích rõ hơn. Đây là **vấn đề ký hiệu quan trọng** mà reviewer sẽ bắt bẻ.

### 5.5 Pseudocode Staged LEX-COS (dòng 267–290)

**Đây là một trong những phần mạnh nhất của bài.** Pseudocode 8 bước rõ ràng, có invariant inductive proof ngắn.

**Vấn đề nhỏ:**
- Bước 2: "Encode $F_{i-1}$ with objective $\min f_i$" — Từ "encode" hơi mơ hồ. Nên dùng "Solve $\min f_i$ subject to $F_{i-1}$" hoặc "Formulate the WCNF for $\min f_i$ over $F_{i-1}$".
- Bước 4: "Verify the assignment and recompute its active objective value" — "active objective value" là gì? Nên nói rõ "Verify feasibility and compute $(f_1, \ldots, f_i)$ for the returned assignment."
- Dòng 284–290: Đoạn proof ngắn sau pseudocode ("By induction, $F_i$ contains exactly...") là tốt. Nhưng câu "Rebuilding the formula between stages can affect runtime but not this argument" có thể bị reviewer hỏi: nếu re-encode có thể thay đổi formula structure, thì có ảnh hưởng đến propagation và kết quả không? Nên clarify: "Rebuilding the formula resets learned clauses but preserves the constraint set; hence the optimality argument holds."

---

## 6. Encoding and Constraint Enhancements — Kỹ thuật tốt, 2 vấn đề ký hiệu

**Dòng 292–363:**

### 6.1 Sorting networks và Totalizer (dòng 295–315)

```
The alternative uses the standard Totalizer tree of Bailleux and Boufkhad
(2003), completed with both merge clause families from their full-CNF
treatment (2004).
```

**Rất tốt:** Cite rõ cả 2 paper (2003 và 2004) để phân biệt "efficient" (một chiều) vs "full CNF" (hai chiều). Đây là điểm kỹ thuật quan trọng.

**Vấn đề:**
- **Equations (dòng 306–308):**
  ```
  \neg L_alpha \lor \neg R_beta \lor O_{alpha+beta},
  L_{alpha+1} \lor R_{beta+1} \lor \neg O_{alpha+beta+1},
  ```
  Ký hiệu $\alpha$, $\beta$ chưa được định nghĩa—đây là threshold indices. Cần thêm "for all valid $\alpha \geq 0$, $\beta \geq 0$" hoặc giải thích rõ range của indices.

- **"using true zero-count and false out-of-range sentinels"** — Câu này kỹ thuật nhưng không giải thích sentinels là gì. Nên thêm: "where $L_0=\top$, $L_{|L|+1}=\bot$, and similarly for $R$ and $O$, to handle boundary cases."

- **Dòng 313:** "We make no general asymptotic-performance claim" — Câu này quan trọng về integrity nhưng nghe hơi defensive. Có thể tích hợp vào câu trước: "...a one-directional implication would not suffice. Formula size, memory, and runtime are measured empirically (RQ2), rather than assumed from asymptotic theory."

### 6.2 Implied constraints (dòng 317–341)

**Rất tốt:** Hai implied constraints được định nghĩa chính xác:
1. User-slot occupancy equality (dựa trên full coverage + user partition).
2. Slot capacity upper bound qua maximum matching $\nu(G_h)$.

**Vấn đề:**
- **$v_{uh}$ và $t_{sh}$ được định nghĩa ở đây (§4.2) nhưng không xuất hiện ở §3.** Nếu sau này cần reference chúng trong Results, độc giả phải quay lại §4.2. Thông thường việc introduce notation trong subsection của một section khác (§4 thay vì §3) là OK với paper 5 trang.

- **Equation (327):** `\sum_{h \in H} v_{uh} = |S_u|` — Cần chú thích "for all users $u \in \mathcal{U}$" để rõ ràng đây là constraint per user, không phải global sum.

- **`\nu(G_h)` (dòng 336):** Maximum matching không cần tính trong thời gian chạy vì chỉ dùng trong preprocessing. Nên clarify: "...computed once during formula construction as an upper bound on the number of services assignable in slot $h$."

- **"The broader implementation option `both-plus`..."** (dòng 340–341): Đây là câu quan trọng để phân biệt `both` vs `both-plus`. **Rất tốt.** Nhưng reviewer có thể hỏi: "what does `both-plus` add?" Nên thêm một câu ngắn: "...which bundles additional enhancements not yet fully evaluated at publication scale."

### 6.3 Exact symmetry breaking (dòng 343–362)

**Rất tốt:** Định nghĩa orbit bằng exact equality của candidate vectors—rõ ràng và mathematically sound.

**Vấn đề:**
- **Dòng 352–355:** "For consecutive members of an equivalence group, precedence clauses require the later member at a position to be preceded by the earlier member at a smaller position." — Câu này khó hiểu. "Preceded" theo nghĩa nào? Nếu đây là value-precedence constraints (v.p.c.), nên dùng thuật ngữ chuẩn: "Value-precedence constraints (VPCs) enforce that if a symmetrically equivalent element at index $j$ is assigned a value, then the element at index $j-1$ is assigned a no-larger value."

- **Cite BogaertsEtAl2022 ở cuối dòng 358–359:** Tốt, nhưng cần clarify: bài dùng chính xác framework nào của Bogaerts et al.? Nếu chỉ dùng ý tưởng "dominance breaking" mà không dùng certification methodology thì nên dùng "inspired by" thay vì cite như một dependency.

---

## 7. Experimental Design — Chất lượng tốt, 3 vấn đề số học

**Dòng 364–434:**

### 7.1 Configurations (dòng 369–401)

**Rất tốt:** Định nghĩa $B$ và $R$ (không phải "baseline" và "proposed" để tránh presuppose), giải thích 2×2×2 factorial design.

**Vấn đề — SỐ RUNS MÂU THUẪN:**
- **Dòng 397:** "The maximum manifest contains 4,996 measured runs"
- **submission_plan.md §4:** "Tổng measured: 4,896 runs"

Có sự sai lệch **100 runs** (4,996 vs 4,896). Đây là lỗi nghiêm trọng—một trong hai con số phải sai.

Kiểm tra theo submission_plan.md §4:
```
1.280 (factorial) + 128 + 128 (screens) + 320 (lex scalability)
+ 1.600 (original weighted primary) + 560 (LEX-COS primary)
+ 160 (LEX-OCS) + 320 (corrected-v2) + 400 (commercial)
= 4.896
```

Nhưng trong main.tex §5.1:
- "Screening records the exact selected total, between 3,856 and 4,996" — tức là 4,996 là **maximum**, không phải con số thực. Con số campaign thực bắt đầu từ 3,856 (nếu chỉ chạy screens) đến 4,996 (nếu chạy tất cả).

Vậy "4,896" trong submission_plan là tổng **measured** (sau khi bỏ smoke runs), còn "4,996" trong main.tex là "maximum manifest". Hai khái niệm khác nhau nhưng dùng cùng ngữ cảnh—dễ gây nhầm cho reviewer.

> [!CAUTION]
> Cần thống nhất một con số duy nhất trong bản thảo, hoặc nếu giữ hai khái niệm thì phải phân biệt rõ trong text: "The maximum manifest contains N runs (including all gated phases); the minimum confirmatory set, excluding screens, contains M runs."

### 7.2 Protocol (dòng 404–434)

**Rất tốt:** Mô tả đầy đủ GCP VM, solver commit, compiler flags, blocked order, timeouts.

**Vấn đề:**
- **Dòng 397–399:** "two 128-run exploratory screens, a 320-run lexicographic scalability screen" — 128×2 = 256, không phải con số trong submission_plan (256 screen rows). Không có vấn đề.

- **"The two exploratory corrected-v2 screens use 60 seconds"** (dòng 412–413): submission_plan §4 ghi timeout cho corrected multiobjective screen và weight screen là **60 s**—khớp.

- **Dòng 412–415:** "the factorial uses 120 seconds, and the lexicographic scalability and publication runs use 300 seconds. Each limit is cumulative across all stages of one lexicographic policy." — Đây là key sentence vì "cumulative" timeout cho LEX-COS là điểm kỹ thuật quan trọng. **Tốt.**

---

## 8. Results — Phần còn outline note

**Dòng 436–528:** Cả 3 subsections của Results vẫn là outline note + placeholder visuals.

**Vấn đề cụ thể với bảng Tab 2 (policy-validation):**

[main.tex L489–522]

Bảng đã có cấu trúc 3 panels (A, B, C)—đây là thiết kế tốt hơn phiên bản 3 cột cũ (từ phiên bản outline trước). Tuy nhiên:

- **Panel A (dòng 502–504):** "LEX-OCS vs. LEX-COS sensitivity: 160 pairs" — submission_plan ghi "LEX-OCS sensitivity trên 80 instances". Nếu 80 instances × 2 policies = 160 "pairs" (mỗi instance là 1 pair), thì OK. Nhưng cách viết "160 pairs" có thể bị hiểu nhầm là 160 instances × 2 = 320 runs. Nên clarify.

- **Panel B (dòng 509):** "Corrected-v2, weighted: 160 runs" — **Thiếu** cột encoding configuration. Corrected-v2 chỉ chạy dưới proposed configuration $R$, không chạy dưới $B$. Cần ghi rõ "proposed $R$" trong column hoặc caption.

- **Dòng 512:** "Gurobi, weighted / LEX-COS: 200 runs" — submission_plan §4 ghi "commercial original: 100 × Gurobi/CPLEX × weighted/LEX-COS = 400 runs". Vậy Gurobi = 200 runs (100 instances × 2 policies) và CPLEX = 200 runs—**khớp**. OK.

- **Outline note dòng 479–483:** "Run the evaluation set only if at least 50% of the 32 predeclared LEX-COS calibration runs prove optimum" — Con số 50% này **không xuất hiện** trong submission_plan.md. submission_plan §3.2 không nêu threshold cụ thể cho corrected-v2. Cần đồng bộ hai tài liệu.

---

## 9. Threats to Validity — Phần mạnh nhất trong bài

**Dòng 530–558:** Đây là section **hoàn chỉnh nhất** trong bản thảo.

**Điểm tốt:**
- Nêu rõ: overtime sparsity, synthetic corrected-v2, staged restart LEX-COS, one solver, small commercial subset.
- Dòng 554: "Finally, $B$ is intentionally an unstrengthened audit comparator, while $R$ bundles three changes." — Đây là câu quan trọng đáp lại vấn đề tôi nêu trong phiên bản trước (intentional weak baseline). **Rất tốt.**

**Vấn đề:**
- **EvalMaxSAT justification (dòng 544–545):** "EvalMaxSAT was excluded when the available candidate binary failed the official-WCNF smoke test" — Câu này hơi cụ thể quá (implementation detail). Nên abstract hóa: "A second MaxSAT solver was evaluated but excluded when it failed the WCNF compatibility check; this avoids cross-solver provenance issues but limits solver generality."

- **Routing và uncertainty** chỉ được nhắc trong 1 câu (dòng 537–539). Với ICIIT 5 trang, đây là đủ. **OK.**

---

## 10. Conclusion — Chỉ còn outline note

**Dòng 560–568:** Toàn bộ là outline note. Đây là section được xếp cuối cùng để viết sau data freeze—**đúng design**.

---

## 11. References.bib — Rà soát BibTeX

**13 entries tổng, tất cả đã compile không lỗi.**

| Entry | Vấn đề |
|---|---|
| `UncetaEtAl2024` | OK. DOI verified. |
| `CappaneraScutella2015` | OK. Transportation Science. |
| `ErrarhoutEtAl2016` | **Vấn đề:** Dùng `@article` với journal "IFAC-PapersOnLine" nhưng đây là conference proceedings. Nên đổi thành `@inproceedings` với booktitle "Proceedings of the 10th IFAC Symposium on Manufacturing Modelling, Management and Control (MIM 2016)". |
| `DemirovicEtAl2019` | **Vấn đề:** DOI `10.1007/s10479-017-2693-y` → paper published online 2017, volume 275 published 2019. Năm trong BibTeX là `2019` nhưng bài `DemirovicEtAl2019` cite từ main.tex là đúng về publication year. OK. |
| `MarquesSilvaEtAl2011` | OK. Author encoding: `Jo\~ao` render thành "João"—OK. |
| `JabsEtAl2024` | OK. JAIR. |
| `BailleuxBoufkhad2003` | OK. CP 2003. |
| `BailleuxBoufkhad2004` | **Vấn đề:** Không có `doi`. Chỉ có URL `https://hal.science/hal-00159899`. SAT 2004 không phải workshop thông thường—nên thêm booktitle đầy đủ: "Proceedings of the 7th International Conference on Theory and Applications of Satisfiability Testing (SAT 2004)". |
| `MorgadoEtAl2014` | OK. CP 2014. |
| `JahrenAsin2018` | **Vấn đề nhỏ:** Author render thành "Jahren and Achá" nhưng đúng là "Jahren and Asín Achá". Trong BBL dòng 175 hiện render "Jahren and Achá"—cần kiểm tra author field formatting. Field hiện tại: `{Eivind Jahren and Roberto Javier As{\'\i}n Ach\'a}` → sẽ render "Jahren and Javier Asín Achá" vì bigtex lấy last name. |
| `AsinEtAl2011` | OK. Constraints journal. |
| `BofillEtAl2022` | OK. IJCIS. |
| `BogaertsEtAl2022` | OK. JAIR. |

---

## 12. Kiểm tra LaTeX/build

| Hạng mục | Kết quả |
|---|---|
| Compile thành công | ✅ 5 trang |
| Citation undefined | ✅ Không có |
| Underfull \vbox | ⚠️ 3 cảnh báo (badness 7133, 10000, 10000) — do float placement tạo trang trắng cục bộ. Normal với table* và figure*. |
| Overfull \hbox | Cần kiểm tra thêm trong log (không scan hết 1211 dòng) |
| `\resultplaceholder` render | ✅ Không render vì `\ifinternaloutlinetrue` + bên trong `\ifinternaloutline` block |
| `\citationplaceholder` | ✅ Không được dùng trong bản này (tốt) |
| Author \affiliation | ✅ Shared group đúng acmart semantics |
| `table*` và `figure*` | ⚠️ Cần xem PDF để verify column span và overflow |

---

## 13. Tổng hợp: danh sách vấn đề cần sửa trước submission

### Ưu tiên cao (ảnh hưởng đến correctness/claim)

| # | Vị trí | Vấn đề | Hành động |
|---|---|---|---|
| H1 | L397 | Số 4,996 vs 4,896 mâu thuẫn với submission_plan | Thống nhất thành 1 con số, phân biệt "max manifest" vs "measured" |
| H2 | L248–252 | `|P|` chưa được định nghĩa | Định nghĩa rõ: $p$ là scalar overtime penalty per hour, đổi thành `p` |
| H3 | L502–504 | "160 pairs" cho LEX-OCS có thể nhầm là 160 instances | Làm rõ: "80 matched pairs (80 instances, each run under LEX-OCS and LEX-COS)" |
| H4 | L479–483 | Ngưỡng 50% corrected-v2 không có trong submission_plan | Đồng bộ hoặc giải thích nguồn gốc ngưỡng này |

### Ưu tiên vừa (ảnh hưởng đến clarity/rigor)

| # | Vị trí | Vấn đề | Hành động |
|---|---|---|---|
| M1 | L96–97 | "bidirectional Totalizer threshold encodings" không nhất quán | Đổi thành "bidirectional Totalizer encodings" |
| M2 | L101 | "artifact-backed provenance" mơ hồ | Đổi thành cụm cụ thể hơn |
| M3 | L128 | "concurrency" không phải thuật ngữ chuẩn trong HCORAP | Đổi thành "slot-conflict" hoặc xóa |
| M4 | L232 | `HE_a` định nghĩa nhưng không dùng trong objectives | Xóa hoặc giải thích vai trò |
| M5 | L306–308 | Indices $\alpha$, $\beta$ chưa định nghĩa range | Thêm "for all valid $\alpha \geq 0$, $\beta \geq 0$" |
| M6 | L327 | Thiếu "for all $u$" trong implied constraint equation | Thêm quantifier |
| M7 | L352–355 | "preceded...at a smaller position" khó hiểu | Dùng thuật ngữ "value-precedence constraint (VPC)" |
| M8 | BibTeX | `ErrarhoutEtAl2016` dùng `@article` nhưng là conference paper | Đổi thành `@inproceedings` |
| M9 | L156–157 | "None of these questions presupposes..." quá defensive | Reformulate thành positive statement |
| M10 | L88 | `{Kieu et al.}` có thể không đúng ICIIT format cho 3 tác giả | Kiểm tra yêu cầu và đổi thành "Kieu, Do, and To" nếu cần |

### Ưu tiên thấp (polish)

| # | Vị trí | Vấn đề |
|---|---|---|
| L1 | L67 | Dấu `\` trong title có thể tạo extra space |
| L2 | L110–111 | `constraint strengthening` quá broad; nên dùng `implied constraints, symmetry breaking` |
| L3 | L146 | Thiếu giải thích "OCS" trong LEX-OCS |
| L4 | L284 | Clarify "rebuilding formula resets learned clauses" |
| L5 | L340–341 | Thêm 1 câu giải thích `both-plus` thêm gì |
| L6 | L544–545 | EvalMaxSAT sentence quá specific—abstract hóa |
| L7 | BibTeX | `BailleuxBoufkhad2004` thiếu booktitle đầy đủ |
| L8 | BibTeX | `JahrenAsin2018` author last name render sai |

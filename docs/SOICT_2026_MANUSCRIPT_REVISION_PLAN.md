# Kế hoạch chỉnh sửa bản thảo SOICT 2026

Cập nhật: 04/09/2026

Bản thảo áp dụng: `LaTeX-Templates/paper/main_soict.tex`

Đích nộp bài: SOICT 2026, Springer CCIS

Quality gate đi kèm: `docs/SOICT_2026_A_STAR_QUALITY_RUBRIC.md`

## 1. Mục tiêu của đợt chỉnh sửa

Đợt chỉnh sửa này phải biến bản thảo từ một báo cáo về nhiều cấu hình kỹ
thuật thành một bài báo có một luận điểm khoa học duy nhất, có bằng chứng rõ
ràng và dễ đọc đối với cả độc giả vận trù học lẫn độc giả chưa quen SAT.

Luận điểm đề xuất là:

> Trong HCORAP, chính sách mục tiêu và mức tải của benchmark quyết định ý nghĩa
> vận hành của nghiệm, còn cardinality encoding quyết định hành vi tính toán
> của mô hình MaxSAT. Vì vậy, hai ảnh hưởng này phải được đánh giá tách biệt,
> trên các instances làm cho các mục tiêu thực sự hoạt động, và phải được kiểm
> chứng bằng các phương pháp tối ưu chính xác độc lập.

Kế hoạch không mặc định rằng Totalizer tốt hơn, MaxSAT nhanh hơn MIP, hoặc
lexicographic optimization là một thuật toán mới. Mỗi kết luận chỉ được giữ
nếu vượt claim gate đã định trước.

## 2. Chẩn đoán bản thảo hiện tại

### 2.1 Những bằng chứng đã đủ mạnh

1. **Ảnh hưởng của objective policy trên HCORAP-LC**
   - Có đủ 48 cặp Weighted và LEX-COS.
   - Gurobi chứng minh 96/96 optimum.
   - CPLEX audit độc lập trên 16 strata và hai policy khớp Gurobi.
   - LEX-COS đồng thời cải thiện continuity và overtime trên 42/48 instances.
   - Compatibility giảm với median tương đối khoảng 6,3%; đây là trade-off
     cần được trình bày, không phải lỗi của phương pháp.

2. **Tính đúng của nghiệm và objective vectors**
   - Các kết quả Corrected-v2 hiện có status nhất quán giữa hai commercial
     solvers.
   - Three-solver subset hiện có thể dùng làm kiểm tra bổ sung.
   - Independent verifier đã có vai trò rõ trong pipeline.

3. **Cơ sở chọn cấu hình MaxSAT tối giản**
   - Weighted factorial cũ cho thấy implied constraints và symmetry breaking
     không tạo lợi ích ổn định.
   - Kết quả này đủ để giải thích vì sao main experiment tắt hai tùy chọn đó.
   - Đây chỉ là configuration-selection evidence, không phải một RQ chính.

### 2.2 Những bằng chứng chưa đủ để kết luận

1. **Encoding effect dưới LEX-COS chưa có main evidence.** Pilot transfer có
   16 instances nhưng không có optimum và không đủ điều kiện dùng trong bài.
2. **RQ2 hiện chỉ dựa trên weighted runs ở 300 s.** Kết quả này không kiểm tra
   được encoding effect có chuyển sang staged lexicographic optimization hay
   không.
3. **Các bảng MaxSAT hiện tại trộn vai trò.** Chúng vừa báo policy, vừa báo
   solver, vừa báo timeout stage nên người đọc không biết bảng đang trả lời
   câu hỏi nào.
4. **Không có cơ sở để claim MaxSAT cạnh tranh với Gurobi.** Design A cho thấy
   Gurobi giải rất nhanh; commercial solvers hiện nên là exact references và
   validators, không phải đối thủ runtime nếu protocol chưa công bằng.

### 2.3 Vấn đề về nội dung và lập luận

1. Introduction chưa đi theo chuỗi `quyết định thực tế -> hạn chế của weighted
   trade-off -> câu hỏi nghiên cứu -> đóng góp -> bằng chứng`.
2. Related Work quá ngắn, chưa thừa nhận rõ lexicographic optimization và MIP
   đã được dùng trong home care.
3. Problem Formulation trộn ý nghĩa nghiệp vụ với chi tiết encode, làm độc giả
   khó thấy ba tiêu chí CONT, OT và SIM phục vụ quyết định nào.
4. Solution Design mô tả nhiều chi tiết cài đặt trước khi giải thích giá trị
   của staged policy và hai encodings.
5. Results thiên về kể bảng, chưa phân tích effect size, uncertainty, instance
   characteristics, trade-off hoặc failure behavior.
6. Discussion lặp kết quả thay vì trả lời: khi nào dùng policy nào, MaxSAT mang
   lại giá trị gì, và kết quả có thể khái quát đến đâu.
7. Một số diễn đạt còn mang tính nội bộ hoặc phòng thủ, chẳng hạn tên campaign,
   tên dataset lịch sử, submission gate và giải thích dài về những gì bài không
   làm.

### 2.4 Vấn đề về hình thức

PDF hiện có 12 trang tổng cộng nhưng phân bổ không hiệu quả:

- trang 4 và 5 bị ngắt mạnh quanh algorithm;
- trang 7 và 8 có nhiều khoảng trắng;
- trang 12 chỉ chứa một phần ngắn của references;
- `\raggedbottom`, minipage và float placement thủ công làm bố cục thiếu cân
  bằng;
- một số caption dài như đoạn văn nhưng vẫn chưa nêu takeaway chính.

SOICT cho phép tối đa 12 trang nội dung, không tính references. Vì vậy có thể
bổ sung khoảng hai trang nội dung có giá trị sau khi sửa flow của floats; chưa
cần cắt ý khoa học trong vòng viết lại đầu tiên.

### 2.5 Blocker về tài liệu tham khảo và artifact

1. Entry `UncetaEtAl2024` hiện sai tác giả, title, journal và DOI. Thông tin
   đúng của công trình gốc là:
   - Irene Unceta, Bernat Salbanya, Jordi Coll, Mateu Villaret, Jordi Nin;
   - *Optimizing resource allocation in home care services using MaxSAT*;
   - *Cognitive Systems Research*, 88 (2024), 101291;
   - DOI `10.1016/j.cogsys.2024.101291`.
2. Entry cardinality networks và Totalizer cần được đối chiếu lại với bản gốc,
   không kế thừa metadata đang có nếu chưa xác minh.
3. `literature_review.md` chứa metadata hỏng và các phát biểu về quy trình tìm
   kiếm không thể kiểm chứng; file này chỉ được dùng như danh sách gợi ý, không
   được dùng làm nguồn trích dẫn.
4. Bản thảo và các results chính đang bị `.gitignore` bỏ qua. Câu “code,
   generator and raw results are publicly available” chỉ được dùng sau khi đã
   có một release hoặc artifact có thể truy cập và kiểm tra được.

## 3. Định vị đóng góp

### 3.1 Những gì không nên claim

- Không claim đây là lần đầu dùng lexicographic optimization trong home care.
- Không claim staged lexicographic optimization là thuật toán mới.
- Không claim Totalizer là encoding tốt nhất một cách phổ quát.
- Không claim MaxSAT vượt Gurobi hoặc CPLEX về runtime.
- Không gọi HCORAP-LC là benchmark thực tế nếu mới chỉ là bộ sinh synthetic có
  mức tải được hiệu chỉnh.
- Không claim routing, uncertainty, Pareto frontier hoặc weight sensitivity là
  đóng góp khi chúng nằm ngoài main experiment.

### 3.2 Ba đóng góp nên dùng trong bản thảo

1. **Objective-policy contribution.** Xây dựng và kiểm chứng chính sách ưu
   tiên nghiêm ngặt cho HCORAP theo thứ tự continuity, overtime, compatibility,
   đồng thời định lượng lợi ích vận hành và chi phí compatibility so với
   weighted objective.
2. **Benchmark-and-evidence contribution.** Xây dựng HCORAP-LC, một bộ
   instances có tải được hiệu chỉnh để continuity và overtime có khả năng tác
   động đến nghiệm; dùng nó để chỉ ra vì sao benchmark quá nhẹ có thể che khuất
   khác biệt giữa các objective policies.
3. **Encoding contribution.** Thực hiện nghiên cứu có kiểm soát về sorting
   network và Totalizer dưới cả Weighted và LEX-COS, giữ nguyên solver,
   instances, resource limits và các tùy chọn còn lại; từ đó xác định encoding
   effect có độc lập với objective policy hay không.

Tính đúng xuyên suốt được hỗ trợ bằng exact MIP references và independent
solution checking. Đây là đặc tính của methodology, không cần tách thành đóng
góp thứ tư trừ khi artifact thực sự hoàn chỉnh.

### 3.3 Giá trị riêng của MaxSAT

MaxSAT không cần thắng MIP mới có giá trị khoa học. Vai trò hợp lý trong bài là:

- cung cấp một formulation khai báo thống nhất cho các ràng buộc Boolean và
  các mục tiêu đếm;
- cho phép nghiên cứu trực tiếp ảnh hưởng của cardinality encoding;
- hỗ trợ staged exact optimization mà không cần quy đổi ba tiêu chí sang một
  thang trọng số chung;
- tạo một điểm nối có thể kiểm chứng giữa objective semantics, encoding và
  solver behavior.

Nếu Design B vẫn cho completion thấp ở 3.600 s, bài phải định vị MaxSAT là một
formulation/evaluation study, không gọi đây là một exact method có khả năng mở
rộng tốt.

## 4. Title và narrative

### 4.1 Title khuyến nghị

Title hiện tại, sau khi Design B đạt claim gate dưới cả hai policies:

> **Exact Multi-Criteria Optimization for Home-Care Resource Allocation with
> MaxSAT**

Title này mô tả tổng quát bài toán và hướng giải chính xác bằng MaxSAT mà không
liệt kê từng thành phần đóng góp. Phần continuity-first, benchmark và encoding
được làm rõ trong abstract và introduction.

Nếu encoding effect yếu hoặc phụ thuộc policy:

> **Strict Objective Priorities for Home-Care Allocation: A MaxSAT Study under
> Calibrated Workload**

Không đưa “Totalizer” vào title trước khi có bằng chứng cross-policy. Không dùng
“novel”, “efficient”, “scalable” hoặc “state-of-the-art” trong title.

### 4.2 Narrative một câu cho từng phần

| Phần | Câu hỏi mà phần đó phải trả lời |
|---|---|
| Introduction | Quyết định nào đang khó và vì sao weighted objective chưa biểu diễn đúng ưu tiên? |
| Related Work | Literature đã giải những phần nào và khoảng trống cụ thể còn lại là gì? |
| Formulation | Một nghiệm hợp lệ là gì và ba tiêu chí được đo như thế nào? |
| Solution Design | Làm thế nào bảo toàn thứ tự ưu tiên và hai encodings khác nhau ở đâu? |
| Methodology | So sánh nào xác lập policy effect và so sánh nào xác lập encoding effect? |
| Results | Dữ liệu trả lời trực tiếp hai RQ như thế nào? |
| Discussion | Kết quả thay đổi lựa chọn mô hình và thực hành đánh giá ra sao? |
| Conclusion | Điều gì đã được chứng minh, trong phạm vi nào? |

## 5. Outline mục tiêu và page budget

Page budget dưới đây tính phần nội dung, không tính references.

| Section | Số trang mục tiêu | Vai trò |
|---|---:|---|
| Abstract | 0,25 | problem, gap, method, hai kết quả chính, scope |
| 1. Introduction | 1,50--1,75 | motivation, gap, RQs, contributions |
| 2. Related Work | 1,00--1,25 | home care, lexicographic/MO, MaxSAT encodings |
| 3. Problem and Objective Policies | 2,00--2,25 | feasible allocation, metrics, Weighted, LEX-COS |
| 4. MaxSAT Formulation and Encodings | 1,00--1,25 | mapping, staged solve, SN/TOT |
| 5. Experimental Methodology | 1,25--1,50 | datasets, two designs, metrics, validation |
| 6. Results | 3,00--3,50 | benchmark, policy, encoding, validation |
| 7. Discussion | 0,75--1,00 | implications, scope and limitations |
| 8. Conclusion | 0,35--0,50 | direct synthesis |
| **Tổng** | **11,10--13,00** | cắt về tối đa 12 sau content freeze |

Vòng viết đầu được phép vượt trang. Chỉ cắt sau khi các claims, tables và
figures đã cố định.

## 6. Kế hoạch chỉnh sửa theo section

### 6.1 Abstract: viết sau cùng

Cấu trúc đúng năm câu:

1. HCORAP cần phân bổ nhân viên trong khi bảo vệ continuity, hạn chế overtime
   và duy trì compatibility.
2. Weighted sums có thể cho phép một tiêu chí quan trọng bị đánh đổi; benchmark
   nhẹ cũng có thể làm tiêu chí đó không hoạt động.
3. Bài đánh giá LEX-COS trên HCORAP-LC và so sánh hai cardinality encodings
   trong một matrix cố định.
4. Báo đúng hai kết quả định lượng mạnh nhất sau data freeze.
5. Nêu kết luận có scope, không dùng lời quảng cáo.

Không đưa implementation detail, tên nội bộ của dataset, danh sách solver dài
hoặc số lượng runs vào abstract.

**Done khi:** mọi con số trong abstract được sinh từ cùng data freeze với
Results và không có claim rộng hơn evidence.

### 6.2 Introduction

Viết lại theo sáu đoạn:

1. Mô tả quyết định phân bổ và hậu quả của continuity, overtime, compatibility
   bằng ngôn ngữ nghiệp vụ.
2. Giải thích vì sao weighted objective không tương đương với ưu tiên tuyệt
   đối, dùng một ví dụ ngắn bằng lời, không cần ví dụ SAT.
3. Giải thích vấn đề benchmark: khi capacity dư thừa, continuity và overtime
   có thể gần như không tác động đến nghiệm.
4. Tóm tắt khoảng trống literature: lexicographic/MIP đã tồn tại, nhưng thiếu
   đánh giá HCORAP kết nối policy, benchmark load và cardinality encoding trong
   một protocol exact, paired và có validation.
5. Nêu RQ1 và RQ2 bằng câu tự nhiên.
6. Nêu ba contributions ở Mục 3.2, mỗi contribution gắn với một nguồn evidence.

**Cần xóa hoặc chuyển:** lịch sử chi tiết của solver, giải thích hard/soft
clause, tên config, caveat dài và các câu tự phòng thủ.

**Done khi:** một người chỉ đọc hai trang đầu có thể trả lời problem, gap, ba
contributions và hai RQs mà không cần biết SAT.

### 6.3 Related Work

Tổ chức theo vấn đề, không tổ chức theo từng paper:

1. **Home-care allocation, scheduling and routing.** Nêu rõ MIP, branch-price-
   and-cut và heuristics đã giải nhiều biến thể rộng hơn HCORAP, có routing,
   time windows hoặc synchronization.
2. **Multiple and lexicographic objectives in home care.** Thừa nhận các công
   trình đã dùng lexicographic evaluation, epsilon-constraint hoặc priority
   weights cho continuity, overtime và preference matching.
3. **MaxSAT and cardinality encodings.** Đặt công trình HCORAP-MaxSAT gốc,
   sorting/cardinality networks, Totalizer và multi-objective MaxSAT vào đúng
   bối cảnh.
4. **Gap paragraph.** Chốt khoảng trống rất hẹp: chưa có bằng chứng cho
   HCORAP về tương tác giữa strict policy, benchmark load và encoding choice
   dưới một evaluation matrix được kiểm soát và exact cross-check.

Không viết literature như bằng chứng rằng “không ai từng làm”. Viết nó như
ranh giới giữa cái đã biết và câu hỏi bài này kiểm tra.

**Done khi:** có bảng so sánh nội bộ tối thiểu 8--12 công trình theo problem,
objectives, method, data, timeout và evidence; main paper có thể chỉ cần một
đoạn tổng hợp hoặc bảng compact nếu còn chỗ.

### 6.4 Problem and Objective Policies

Đổi thứ tự trình bày:

1. Mở bằng một đoạn mô tả input và output bằng lời.
2. Định nghĩa sets, parameters và assignment variable trong một notation table
   nhỏ; không để notation xuất hiện trước định nghĩa.
3. Trình bày feasibility constraints theo nhóm nghiệp vụ: coverage,
   qualification/eligibility và capacity.
4. Định nghĩa ba đại lượng với chiều tối ưu rõ ràng:
   - `CONT`: số lần phá vỡ continuity, minimize;
   - `OT`: lượng overtime, minimize;
   - `SIM`: compatibility score, maximize.
5. Trình bày Weighted policy như baseline của nghiên cứu gốc.
6. Trình bày LEX-COS bằng objective vector
   `(CONT, OT, -SIM)` và một câu giải thích ý nghĩa của thứ tự.
7. Cho một proposition ngắn: nếu ba stages đều kết thúc optimum và mỗi stage
   cố định optimum trước đó, nghiệm cuối là lexicographic optimum.
8. Đưa LEX-OT vào một câu sensitivity; không phát triển thành policy chính.

Giải thích quan hệ STAB-CONT chỉ trong điều kiện full coverage. Không dùng hai
tên cho cùng đại lượng xuyên suốt bài; chọn `continuity violations` làm thuật
ngữ chính và chỉ nhắc stability khi đối chiếu bài gốc.

**Done khi:** mỗi phương trình có một câu giải thích nghiệp vụ và model trong
text khớp code/reference IR.

### 6.5 MaxSAT Formulation and Encodings

Giữ section ở mức đủ kiểm tra nhưng không dạy nhập môn SAT:

1. Một đoạn ánh xạ assignment và constraints sang Boolean formulation.
2. Một algorithm/pseudocode ngắn cho staged solve; đưa cumulative 3.600 s
   budget vào Methodology, không nhét vào algorithm.
3. Một đoạn cho sorting network và một đoạn cho Totalizer, tập trung vào phần
   cardinality constraints mà chúng encode.
4. Một câu nói mọi yếu tố khác được giữ cố định trong RQ2.
5. Một đoạn rất ngắn giải thích ablation cũ dẫn đến tắt implied constraints và
   symmetry breaking trong matrix chính.

Không mô tả mọi auxiliary variable nếu không dùng trong proof hoặc analysis.
Không gọi encoding là solver. Không dùng “more compact” nếu chỉ giảm variables
nhưng tăng clauses; báo hai số riêng.

**Done khi:** chuyên gia SAT có thể kiểm tra semantics, còn độc giả OR vẫn hiểu
vì sao section này liên quan RQ2.

### 6.6 Experimental Methodology

Chia rõ hai designs:

#### Design A: objective-policy effect

- 48 HCORAP-LC instances, 16 size strata x 3 evaluation seeds;
- Weighted và LEX-COS;
- Gurobi exact primary results, CPLEX stratified audit;
- 300 s một thread;
- paired comparison của objective vectors, không so runtime với MaxSAT.

#### Design B: policy-by-encoding effect

- 48 Original instances;
- 2 policies x 2 encodings = 192 EvalMaxSAT runs;
- Weighted-SN, Weighted-TOT, LEX-COS-SN, LEX-COS-TOT;
- 3.600 s cumulative limit, một thread;
- implied constraints và symmetry breaking đều tắt;
- cùng VM, binary, commit, instance hash và randomized instance blocks;
- 96 Gurobi exact references ở cùng 3.600 s để kiểm tra objective vectors;
- commercial solver không được diễn giải như runtime baseline.

#### Metrics và uncertainty

- status counts: optimum, infeasible, timeout;
- completion và PAR-2 cho dữ liệu có timeout;
- median runtime và paired speedup trên domain được định nghĩa rõ;
- bootstrap 95% CI, resample theo instance, không coi stages là observations
  độc lập;
- variables, clauses, peak RSS;
- stage nơi LEX-COS timeout;
- objective agreement với exact reference;
- hardware: GCP C4 high-cpu, 8 vCPU, 16 GB RAM, nhưng mỗi run một thread.

**Done khi:** người khác có thể dựng lại matrix và metric chỉ từ section này và
artifact README.

### 6.7 Results

Mỗi subsection phải bắt đầu bằng câu trả lời trực tiếp, sau đó mới chỉ bảng và
phân tích.

#### 6.7.1 Why workload calibration matters

- So sánh objective activity giữa Original và HCORAP-LC.
- Báo tỷ lệ instances mà CONT hoặc OT là hằng/không hoạt động dưới Original.
- Giải thích HCORAP-LC dùng để đánh giá policy semantics, không dùng để chứng
  minh solver scalability.
- Không dùng tên `Corrected-v2` trong main paper; dùng `HCORAP-LC` và định
  nghĩa rõ đây là load-calibrated synthetic benchmark.

#### 6.7.2 Effect of strict priorities

- Báo 48 paired deltas Weighted -> LEX-COS.
- Báo số instances improved/equal/worse cho từng criterion.
- Chuẩn hóa CONT và OT theo services hoặc capacity khi so giữa kích thước.
- Báo compatibility loss cả absolute lẫn relative.
- Phân tích joint improvement: bao nhiêu instances cùng giảm CONT và OT.
- Phân tích 3 cases mà LEX-COS và LEX-OT khác nhau để cho thấy priority order
  thực sự có ý nghĩa ở biên, nhưng không thổi phồng từ ba observations.

#### 6.7.3 Effect of cardinality encoding

- Dùng đúng bốn cells của Design B.
- Báo completion, PAR-2, median runtime, peak RSS, variables và clauses.
- Báo paired SN/TOT speedup và 95% CI riêng cho Weighted, LEX-COS.
- Phân tích timeout theo lexicographic stage.
- Kiểm tra encoding effect theo instance size/load, nhưng chỉ diễn giải như
  pattern nếu mỗi stratum có quá ít seeds.
- Báo Totalizer broadly better chỉ khi đạt gate ở Mục 9.2.

#### 6.7.4 Exact validation

- Một đoạn hoặc bảng compact cho CPLEX-Gurobi agreement ở Design A.
- Một đoạn cho MaxSAT-Gurobi objective agreement ở Design B.
- Một câu về independent solution checker.
- Không giữ bảng completion 300 s cũ trong main results.

**Done khi:** mỗi RQ có một kết luận một câu, một main table/figure trực tiếp và
một đoạn giải thích nguyên nhân hoặc phạm vi.

### 6.8 Discussion

Tích hợp limitations và threats, không tạo section riêng:

1. **Decision implication.** LEX-COS phù hợp khi continuity và overtime là
   service commitments; Weighted phù hợp khi người ra quyết định chấp nhận
   explicit exchange rates.
2. **Benchmark implication.** Objective policy chỉ đánh giá được trên instance
   khiến các tiêu chí hoạt động; benchmark nhẹ phù hợp hơn cho encoding stress
   hoặc comparability với nghiên cứu gốc.
3. **Encoding implication.** Kết luận Totalizer transfer, policy-dependent hoặc
   neutral dựa đúng claim gate.
4. **Why MaxSAT.** Nêu giá trị formulation/encoding study; thừa nhận MIP giải
   nhanh Design A mà không viết phòng thủ.
5. **Scope.** Synthetic data, một MaxSAT solver, một VM class, ba seeds mỗi
   size stratum, và không xét routing/uncertainty. Gắn mỗi giới hạn với điều
   không được khái quát, không tạo danh sách dài.

### 6.9 Conclusion

Ba đoạn ngắn:

1. Nhắc problem và hai RQs.
2. Tóm tắt hai kết quả mạnh nhất với số liệu đã freeze.
3. Nêu implication và một hướng tiếp theo hợp lý, ưu tiên evaluation trên dữ
   liệu vận hành hoặc solver-diverse validation; không liệt kê mọi extension.

## 7. Kế hoạch phân tích sâu từ dữ liệu đã có

Các phân tích sau không cần chạy solver mới:

| Phân tích | Input | Output dự kiến | Vai trò |
|---|---|---|---|
| Objective activation | Original + HCORAP-LC exact results | tỷ lệ CONT/OT thay đổi được | biện minh benchmark |
| Paired policy deltas | 48 Gurobi pairs | median, IQR/CI, win/equal/loss | RQ1 chính |
| Normalized deltas | instance metadata | delta/service, delta/capacity | so sánh qua sizes |
| Joint trade-off | paired objective vectors | CONT-OT gains vs SIM loss | hình chính RQ1 |
| Priority sensitivity | LEX-COS vs LEX-OT | 45 equal + 3 differing cases | scope của ordering |
| Load/size stratification | metadata + deltas | pattern theo U, A, V/load | giải thích effect |
| IC/SB ablation | weighted factorial cũ | compact selection table | chọn config |
| Cross-solver audit | exact validation files | agreement counts | correctness |

Mọi CI phải ghi rõ resampling unit. Với ba seeds mỗi stratum, không dùng kiểm
định nhiều nhóm hoặc claim causal theo size; chỉ báo descriptive pattern và
uncertainty.

## 8. Kế hoạch thực nghiệm còn lại

Chỉ chạy Design B đã khóa trong `submission_plan.md`:

| Policy | Encoding | Instances | Timeout | Runs |
|---|---|---:|---:|---:|
| Weighted | Sorting network | 48 Original | 3.600 s | 48 |
| Weighted | Totalizer | 48 Original | 3.600 s | 48 |
| LEX-COS | Sorting network | 48 Original | 3.600 s cumulative | 48 |
| LEX-COS | Totalizer | 48 Original | 3.600 s cumulative | 48 |
| **EvalMaxSAT total** |  |  |  | **192** |
| Gurobi references, two policies | MIP-E | 48 Original | 3.600 s | **96** |

Không chạy lại Design A. Không mở lại epsilon-constraint/Pareto, routing,
uncertainty, weight sensitivity, IC/SB factorial hoặc LEX-OT full MaxSAT trong
main campaign.

### 8.1 Pre-run gate

- frozen source commit đã push lên remote;
- worktree sạch trên GCP;
- EvalMaxSAT binary hash, Gurobi version và license check được ghi;
- dry-run resolve đúng 288 records;
- generated instances có checksums;
- status parser phân biệt optimum/infeasible/timeout;
- lexicographic runner dùng cumulative budget và đúng thứ tự CONT -> OT -> SIM;
- independent verifier chạy trong smoke test;
- output và backup path có đủ dung lượng.

### 8.2 Claim gate cho Totalizer

Chỉ dùng câu “Totalizer improves computational performance” cho một policy
nếu đồng thời:

1. số optimum/infeasible được chứng minh không thấp hơn sorting network;
2. PAR-2 thấp hơn;
3. median paired `runtime_SN/runtime_TOT > 1`;
4. bootstrap 95% CI của median speedup nằm hoàn toàn trên 1;
5. không có objective mismatch hoặc invalid assignment.

Chỉ gọi hiệu ứng là cross-policy nếu gate đạt riêng cho cả Weighted và LEX-COS.
Nếu không đạt, title và discussion phải dùng kết luận neutral hoặc policy-
dependent.

### 8.3 Stop rule

- Không thay instance subset, timeout hoặc metrics sau khi xem kết quả.
- Nếu pipeline fail do bug, sửa code, tăng experiment version và chạy lại toàn
  bộ cells bị ảnh hưởng; không trộn trước và sau fix.
- Nếu một solver crash có hệ thống, giữ logs và báo invalid campaign trước khi
  quyết định rerun.
- Không dùng partial results trong abstract hoặc main tables.

## 9. Bốn visual chính

Không tăng số bảng/hình chỉ để làm bài dài hơn.

1. **Table 1: Problem and benchmark summary.** Sets/scale, Original versus
   HCORAP-LC, và objective activation; không phải bảng liệt kê mọi config.
2. **Figure 1: Policy trade-off.** Paired CONT/OT improvement và SIM loss trên
   48 HCORAP-LC instances; ưu tiên một figure có small multiples hoặc joint
   distribution thay vì ba biểu đồ rời.
3. **Table 2: Policy-by-encoding results.** Bốn rows, completion, PAR-2, median
   runtime, RSS, variables, clauses.
4. **Figure 2: Encoding effect.** Hai paired speedup estimates với 95% CI,
   một cho Weighted và một cho LEX-COS; có thể thêm stage-timeout inset nếu
   cần.

Exact validation nên là một phần compact trong Table 2 hoặc một đoạn có số cụ
thể. LEX-OT sensitivity và IC/SB ablation đưa vào appendix/artifact nếu page
limit căng.

Caption phải tự trả lời “độc giả cần nhìn thấy điều gì”, không chỉ mô tả axes.
Mọi visual được sinh bằng script từ raw data và có macro/source table nhất
quán với text.

## 10. Kế hoạch tìm kiếm và sửa Related Work

### 10.1 Search protocol

Tìm trên Google Scholar, Crossref, Scopus/Web of Science nếu có quyền truy cập,
và trang chính thức của publisher. Dùng các query:

- `home health care resource allocation lexicographic optimization`;
- `home care continuity overtime lexicographic objective`;
- `home care allocation MaxSAT`;
- `home health care scheduling multiobjective exact method`;
- `home care epsilon constraint Pareto continuity preference`;
- `MaxSAT lexicographic multiobjective optimization`;
- `cardinality encoding Totalizer sorting network MaxSAT`;
- `ordered objectives maximum satisfiability`.

Backward snowball từ bài HCORAP-MaxSAT gốc và các survey home health care;
forward snowball trên Google Scholar tới 2026. Chỉ dùng nguồn primary để phát
biểu technical claims. Survey dùng để tìm nguồn và mô tả taxonomy.

### 10.2 Inclusion criteria

- peer-reviewed journal/conference hoặc official solver-evaluation report;
- liên quan trực tiếp allocation/scheduling/resource planning trong home care,
  multiple/lexicographic objectives hoặc MaxSAT cardinality encoding;
- ghi được đầy đủ DOI, venue, year và stable URL;
- ưu tiên bài có formulation, exact method, benchmark và timeout rõ.

### 10.3 Các nguồn bắt buộc phải đối chiếu

1. Unceta et al. (2024), HCORAP bằng MaxSAT, *Cognitive Systems Research*,
   DOI `10.1016/j.cogsys.2024.101291`.
2. Mosquera, Smet, and Vanden Berghe (2019), flexible home care scheduling,
   *Omega*, DOI `10.1016/j.omega.2018.02.005`.
3. Lanzarone and Matta (2014), robust nurse-to-patient assignment, continuity
   và lexicographic overtime, DOI `10.1016/j.orhc.2014.01.003`.
4. Malagodi et al. (2021), home-care vehicle routing, overtime và preference
   matching, MILP, DOI `10.1007/s10729-020-09532-2`.
5. Cappanera and Scutellà (2015), integrated assignment, scheduling and
   routing, DOI `10.1287/trsc.2014.0548`.
6. Các exact home-health-care methods dùng branch-price-and-cut hoặc MIP để
   đặt đúng vai trò của commercial baseline.
7. Jabs et al. (2024), bi-objective MaxSAT, DOI `10.1613/jair.1.15333`.
8. Berg, Schidler, and Järvisalo (2026), ordered objectives in MaxSAT; phải
   phân biệt “ordered objective variables” với thứ tự từ điển giữa nhiều
   objective functions của bài này.
9. Bailleux and Boufkhad, Totalizer; Asín et al., cardinality networks; kiểm
   tra metadata từ Springer/DOI trước khi sửa `.bib`.
10. MaxSAT Evaluation official reports để biện minh solver và 3.600 s timeout,
    không dùng chúng để claim EvalMaxSAT tốt nhất hiện nay nếu chưa có đúng
    track/year evidence.

### 10.4 Literature comparison matrix nội bộ

Tạo CSV/Markdown với các cột:

`citation | decision scope | objectives | priority mechanism | solver/method |
data | instance scale | timeout | exactness | relation to this paper`.

Mọi câu novelty trong Introduction phải truy ngược được tới matrix này. Nếu
không đủ bằng chứng cho “first”, thay bằng phát biểu gap có scope như “we are
not aware of an evaluation that jointly isolates ...”. Tốt hơn nữa là tránh
“first” hoàn toàn.

## 11. Bibliography và provenance audit

1. Sửa entry bài gốc trước mọi chỉnh sửa nội dung.
2. Với mỗi `.bib` entry đang được cite:
   - mở DOI hoặc publisher page;
   - đối chiếu authors, title, year, venue, volume, issue, pages/article number;
   - chuẩn hóa capitalization của MaxSAT, HCORAP, Totalizer;
   - xóa duplicate và entry không được cite sau content freeze.
3. Chạy script kiểm tra DOI HTTP status và duplicate DOI/key.
4. Build sạch để phát hiện undefined citations/references.
5. Ghi commit hash của source, generator, raw results và analysis scripts trong
   artifact manifest.
6. Chỉ phát biểu public availability khi URL release/DOI artifact hoạt động từ
   một máy không đăng nhập.

## 12. Quy tắc viết dễ hiểu và nhất quán

### 12.1 Từ vựng khóa

| Dùng nhất quán | Tránh hoặc chỉ dùng khi định nghĩa |
|---|---|
| home-care resource allocation | roster/schedule/route nếu model không tạo chúng |
| caregiver | worker, agent, nurse dùng lẫn nhau |
| service request | visit/task dùng lẫn nhau |
| continuity violation | stability cost trừ khi đối chiếu bài gốc |
| overtime | excess load, overflow dùng không định nghĩa |
| compatibility score | similarity/affinity dùng lẫn nhau |
| objective policy | regime, scheme, design dùng mơ hồ |
| sorting network, Totalizer | SN/TOT trước khi định nghĩa |
| exact reference/validator | commercial baseline nếu không so runtime |
| load-calibrated benchmark | corrected/fixed benchmark |

### 12.2 Quy tắc câu và paragraph

- Mỗi câu có chủ ngữ và động từ chính rõ.
- Mỗi paragraph mở bằng một claim, theo sau bởi evidence/logic, kết bằng
  implication hoặc bridge.
- Một câu không mang quá hai ý chính.
- Không dùng em dash; dùng dấu phẩy, dấu chấm phẩy hoặc tách câu.
- Không in đậm từ giữa paragraph nếu không phải term trong định nghĩa.
- Không dùng “clearly”, “obviously”, “significantly” khi chưa có nghĩa thống
  kê hoặc operational.
- Không viết “to the best of our knowledge” để thay thế literature search.
- Không lặp số giống nhau trong text, caption và nhiều tables nếu không cần.
- Không gọi một result là “strong” hoặc “promising”; mô tả effect trực tiếp.

## 13. Claim-evidence matrix phải khóa trước data freeze

| Claim dự kiến | Evidence bắt buộc | Trạng thái hiện tại | Quyết định |
|---|---|---|---|
| Weighted policy che khuất strict priorities | definition + paired Design A | đủ | giữ |
| HCORAP-LC làm mục tiêu hoạt động hơn Original | activation analysis | đủ: overtime hoạt động ở 47/48 Weighted optima, so với 1/33 jointly solved Original optima | giữ |
| LEX-COS cải thiện CONT và OT | 48 exact paired results | đủ | giữ |
| LEX-COS có compatibility cost | paired absolute/relative deltas | đủ | giữ |
| Priority order đôi khi thay đổi nghiệm | LEX-COS vs LEX-OT 48 pairs | đủ nhưng chỉ 3 cases | sensitivity only |
| Totalizer tốt hơn weighted SN | Design B weighted cell + gate | đủ: median speedup 1,14; CI [1,08; 1,23] | giữ |
| Totalizer effect chuyển sang LEX-COS | Design B LEX-COS cell + gate | đủ: median speedup 1,04; CI [1,03; 1,05] | giữ |
| MaxSAT results đúng | Gurobi references + verifier | đủ cho 192 runs; 0 mismatch | giữ |
| MaxSAT cạnh tranh với MIP | fair runtime protocol | không có | xóa claim |
| Artifact reproducible | public frozen release + clean test | chưa xác minh | chờ |

## 14. Trình tự triển khai

### Phase 0: khóa khoa học, 04/09

- duyệt kế hoạch này và rubric;
- khóa hai RQs, ba contributions và Design B;
- sửa bibliography blocker và tạo literature matrix;
- không sửa Abstract/Conclusion ở phase này.

### Phase 1: viết phần không phụ thuộc Design B, 05--06/09

- viết lại Introduction và Related Work;
- tái cấu trúc Problem and Objective Policies;
- rút gọn MaxSAT section;
- viết Methodology theo hai designs;
- tạo analysis scripts cho Design A và visual templates;
- chạy build và visual QA mỗi ngày.

### Phase 2: chạy và kiểm tra Design B, 05--10/09

- preflight, smoke test và freeze commit;
- chạy Gurobi references và 192 EvalMaxSAT records;
- resume theo runner, không thay matrix;
- verify completeness, objective agreement và assignments;
- khóa raw data và checksums.

### Phase 3: analysis và Results, 10--12/09

- sinh bốn visuals;
- áp claim gates;
- viết Results theo RQ, không theo tên output folder;
- viết Discussion, tích hợp scope/limitations;
- chốt title dựa trên Design B.

### Phase 4: front/back matter, 12--13/09

- viết lại Abstract và Conclusion từ claim-evidence matrix;
- cập nhật keywords theo title và scope;
- kiểm tra author/affiliation single-blind đúng template;
- tạo artifact release candidate.

### Phase 5: review và submission, 14--16/09

- ít nhất hai người chấm rubric độc lập;
- sửa mọi gate fail và mọi tiêu chí B/C/D/E dưới 3;
- content freeze rồi mới cắt về 12 trang;
- clean build, kiểm tra PDF từng trang, citations, links và metadata;
- thử artifact trên môi trường sạch;
- nộp trước deadline, kiểm tra timezone của hệ thống submission.

Lưu ý: abstract deadline chính thức là 09/09/2026 và full-paper deadline là
16/09/2026. Design B đã hoàn tất ngày 07/09; title, abstract, Results,
Discussion và Conclusion đã được đồng bộ theo claim gate đạt ở cả hai policies.

## 15. Definition of Done

Bản thảo chỉ được coi là sẵn sàng khi:

1. tất cả gate G0--G5 trong rubric là `PASS`;
2. Design B đủ 288 records và qua evidence validation;
3. title, abstract, contributions, Results và Conclusion dùng cùng claims;
4. Related Work thừa nhận đúng prior lexicographic/MIP/MaxSAT work;
5. bibliography không có DOI hỏng hoặc metadata sai;
6. mọi table/figure được sinh từ scripts và có source checksum;
7. không còn thuật ngữ dùng trước định nghĩa, tên nội bộ hoặc em dash;
8. không có claim MaxSAT superiority nếu không có fair evidence;
9. body không quá 12 trang và references nằm ngoài page count;
10. PDF qua visual QA, không có khoảng trắng bất thường, orphan heading, bảng
    tràn hoặc hình khó đọc;
11. artifact URL hoạt động và một người khác chạy được smoke test;
12. rubric đạt ít nhất 85/100, không gate fail, và các mục Novelty, Technical
    correctness, Experimental design, Results đều đạt tối thiểu 3/4.

## 16. Quyết định cần giữ cố định trong quá trình sửa

- Venue là SOICT 2026, không còn ICIIT.
- Main paper có hai RQs và hai experimental designs.
- HCORAP-LC dùng cho policy semantics; Original dùng cho encoding study và khả
  năng so sánh với nghiên cứu gốc.
- Gurobi/CPLEX là exact references hoặc validators trừ khi có fair runtime
  protocol mới được thiết kế trước.
- Timeout 300 s của Design A và 3.600 s của Design B không được trộn trong cùng
  runtime comparison.
- Limitations được tích hợp trong Discussion và Conclusion.
- Vòng đầu ưu tiên đúng, rõ và đủ evidence; page trimming thực hiện sau cùng.

## 17. Các trang nguồn đã xác minh cho kế hoạch

- [SOICT 2026 paper submission](https://soict.org/submission/paper-submission/)
  cho page limit, template và review model.
- [SOICT 2026](https://soict.org/) cho timeline và conference scope.
- [Unceta et al., HCORAP-MaxSAT](https://www.sciencedirect.com/science/article/pii/S1389041724000858)
  cho nguồn mô hình gốc và metadata chính xác.
- [Mosquera et al., flexible home-care scheduling](https://www.sciencedirect.com/science/article/abs/pii/S0305048317305996)
  cho bối cảnh flexible scheduling và lexicographic evaluation.
- [Lanzarone and Matta, robust home-care assignment](https://www.sciencedirect.com/science/article/pii/S2211692314000046)
  cho continuity và lexicographic overtime objectives.
- [Malagodi et al., preference matching and overtime](https://pmc.ncbi.nlm.nih.gov/articles/PMC8184733/)
  cho MILP, strict/soft preferences và 3.600 s experimental limit.
- [Jabs et al., bi-objective MaxSAT](https://www.cs.helsinki.fi/u/mjarvisa/papers/jbnj.jair24.pdf)
  cho bối cảnh multi-objective MaxSAT.
- [Berg et al., ordered objectives in MaxSAT](https://ojs.aaai.org/index.php/AAAI/article/download/38429/42391)
  cho nghiên cứu MaxSAT gần nhất; thuật ngữ của bài này phải được phân biệt với
  lexicographic ordering giữa ba HCORAP criteria.
- [MaxSAT Evaluations](https://maxsat-evaluations.github.io/) cho nguồn chính
  thức về solver-evaluation protocols.

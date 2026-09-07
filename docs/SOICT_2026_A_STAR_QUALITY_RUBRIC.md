# Rubric chất lượng A* cho bản thảo SOICT 2026

Cập nhật: 04/09/2026.

Phạm vi áp dụng: bản thảo đang hoạt động
[`LaTeX-Templates/paper/main_soict.tex`](../LaTeX-Templates/paper/main_soict.tex).

## 1. Mục đích và cách sử dụng

Rubric này dùng chuẩn chất lượng của các hội nghị chọn lọc cao làm mức kiểm
tra nội bộ. Nó không khẳng định SOICT là hội nghị A* và không dự đoán quyết
định accept. Mục tiêu là buộc bản thảo đạt bốn yêu cầu:

1. đóng góp có ý nghĩa và được định vị chính xác;
2. lập luận và phương pháp đúng;
3. thực nghiệm đủ mạnh để hỗ trợ từng claim;
4. bài viết và artifact cho phép reviewer kiểm tra kết quả.

Quy trình đánh giá:

1. kiểm tra các gate bắt buộc ở Mục 3;
2. chấm từng nhóm tiêu chí từ 0 đến 4 ở Mục 4;
3. ghi bằng chứng bằng section, bảng, hình hoặc đường dẫn artifact;
4. mở action item cho mọi mục dưới 3 điểm;
5. nhờ ít nhất hai người chấm độc lập trước khi data freeze.

Không cộng điểm để bù cho một gate bị trượt. Một bản thảo có tổng điểm cao
nhưng còn claim không có bằng chứng vẫn là `NOT READY`.

## 2. Cơ sở của rubric

Rubric kết hợp yêu cầu chính thức của SOICT 2026 với các tiêu chí phản biện và
reproducibility checklist của những hội nghị chọn lọc cao:

- [SOICT 2026 paper submission](https://soict.org/submission/paper-submission/)
  quy định Springer CCIS, tối đa 12 trang không tính references, single-blind,
  PDF không có số trang và ít nhất ba reviewer.
- [SOICT 2026](https://soict.org/) liệt kê Applied Operations Research and
  Optimization trong phạm vi hội nghị.
- [AAAI-26 reviewer instructions](https://aaai.org/conference/aaai/aaai-26/instructions-for-aaai-26-reviewers/)
  yêu cầu câu chuyện rõ, đóng góp kỹ thuật xác định được, phương pháp sound,
  related work đúng ngữ cảnh, baseline phù hợp, metric hợp lý, benchmark phù
  hợp và thực nghiệm có thể tái lập.
- [AAAI-26 review criteria](https://aaai.org/conference/aaai/aaai-26/main-technical-track-call/)
  nhấn mạnh significance, novelty, soundness, relevance, clarity và
  reproducibility.
- [NeurIPS paper checklist](https://neurips.cc/public/guides/PaperChecklist)
  yêu cầu claim khớp bằng chứng, nêu assumptions và scope, mô tả experimental
  settings, confidence intervals, compute resources, code và dữ liệu.
- [ACM SIGMOD reproducibility criteria](https://reproducibility.sigmodconf.hosting.acm.org/)
  yêu cầu source, configuration, input data, execution protocol và script tạo
  lại bảng, hình từ raw results.

## 3. Gate bắt buộc

Đánh dấu `PASS`, `FAIL` hoặc `N/A`. Tất cả gate áp dụng phải là `PASS`.

### G0. Venue và submission

- [ ] `PASS` Chủ đề được đặt trong Applied Operations Research and
  Optimization, decision support hoặc healthcare applications của SOICT.
- [ ] `PASS` Bản thảo dùng đúng Springer CCIS/LNCS template do SOICT cung cấp.
- [ ] `PASS` Phần nội dung chính không vượt 12 trang; references không tính
  trong giới hạn này.
- [ ] `PASS` PDF không có số trang.
- [ ] `PASS` Bản single-blind giữ tên và affiliation của tác giả.
- [ ] `PASS` Bản thảo là tiếng Anh và được nộp ở định dạng PDF.
- [ ] `PASS` Công trình là nguyên bản, chưa xuất bản và không đồng thời được
  review ở nơi khác.
- [ ] `PASS` Abstract được nộp trước 09/09/2026.
- [ ] `PASS` Full paper được nộp trước 16/09/2026.

### G1. Claim integrity

- [ ] `PASS` Mỗi claim trong title, abstract, introduction và conclusion có
  một bảng, hình, định lý hoặc phân tích trực tiếp hỗ trợ.
- [ ] `PASS` Không gọi staged lexicographic optimization là thuật toán tối ưu
  mới nếu phần thuật toán đã được biết trong literature.
- [ ] `PASS` Không claim MaxSAT nhanh hơn Gurobi hoặc CPLEX khi chưa có
  protocol so sánh runtime công bằng và kết quả tương ứng.
- [ ] `PASS` Không suy rộng kết quả weighted sang LEX-COS khi Design B chưa có.
- [ ] `PASS` Không dùng pilot, incomplete campaign hoặc số nhập tay làm main
  evidence.
- [ ] `PASS` Abstract, Results và Conclusion dùng cùng một data freeze.

### G2. Technical correctness

- [ ] `PASS` Tất cả tập, tham số, biến, metric và acronym được định nghĩa trước
  khi dùng.
- [ ] `PASS` Hard constraints trong bản thảo tương đương model trong code.
- [ ] `PASS` Quan hệ giữa `STAB` và `CONT` được phát biểu với đúng điều kiện
  full coverage.
- [ ] `PASS` Ba stage của LEX-COS cố định chính xác optimum của stage trước.
- [ ] `PASS` Một lexicographic run chỉ là `OPTIMUM` khi cả ba stage chứng minh
  optimum trong cumulative budget.
- [ ] `PASS` Sorting network và Totalizer encode cùng cardinality semantics.
- [ ] `PASS` Implied constraints và symmetry breaking không làm mất objective
  vector tối ưu cần so sánh.
- [ ] `PASS` `OPTIMUM`, `INFEASIBLE` và `TIMEOUT` được phân biệt trong code,
  analysis và manuscript.
- [ ] `PASS` Mọi assignment được báo cáo đều qua independent solution checker.

### G3. Main evidence

- [ ] `PASS` Design A có đủ 48 cặp Weighted và LEX-COS trên HCORAP-LC.
- [ ] `PASS` Gurobi chứng minh đủ 96 optimum của hai policy chính.
- [ ] `PASS` CPLEX audit đủ 16 strata x 2 policy và khớp objective với Gurobi.
- [ ] `PASS` Design B có đủ 48 instance x 2 policy x 2 encoding ở 3600 s.
- [ ] `PASS` Design B dùng cùng solver binary, thread, VM, timeout và instance
  hashes cho các paired comparisons.
- [ ] `PASS` Gurobi reference của Design B đầy đủ và mọi MaxSAT optimum khớp
  objective vector tương ứng.
- [ ] `PASS` Không trộn runtime 300 s với runtime 3600 s trong một PAR-2,
  speedup hoặc cactus plot.
- [ ] `PASS` Claim Totalizer nhanh hơn chỉ được dùng khi completion không giảm,
  PAR-2 giảm, median paired speedup lớn hơn 1 và 95% CI nằm trên 1.
- [ ] `PASS` Kết quả neutral hoặc policy-dependent vẫn được báo đúng nếu claim
  gate về Totalizer không đạt.

### G4. Reproducibility

- [ ] `PASS` Commit dùng chạy measured campaign tồn tại trên remote và được
  ghi trong provenance.
- [ ] `PASS` Artifact chứa code, license, exact instances, configs, solver
  versions hoặc hashes, raw JSON/logs và environment.
- [ ] `PASS` Có một lệnh preflight và một lệnh end-to-end để tạo lại main
  results trên một VM sạch.
- [ ] `PASS` Mọi bảng và hình định lượng được sinh từ script, không chép số
  thủ công.
- [ ] `PASS` Checksums kết nối raw result, analysis output và LaTeX output.
- [ ] `PASS` Một người không viết runner đã thực hiện smoke test theo README.

### G5. Presentation

- [ ] `PASS` Reviewer có thể nêu problem, gap và ba contributions sau khi đọc
  title, abstract và hai trang đầu.
- [ ] `PASS` Không có thuật ngữ nội bộ như `Corrected-v2`, submission gate,
  campaign phase hoặc tên config khó hiểu trong main text.
- [ ] `PASS` Không có placeholder, TODO, citation thiếu, reference hỏng hoặc số
  liệu mâu thuẫn.
- [ ] `PASS` PDF không có text bị cắt, bảng tràn, hình mờ, trang trắng hoặc
  section heading đứng một mình cuối trang.

## 4. Thang điểm 100

### 4.1 Mức điểm chung

| Điểm | Diễn giải |
|---:|---|
| 0 | Thiếu, sai hoặc không thể kiểm tra. |
| 1 | Có đề cập nhưng còn lỗ hổng có thể dẫn đến reject. |
| 2 | Đủ cho một bài hội nghị thông thường nhưng chưa thuyết phục reviewer khó tính. |
| 3 | Mạnh, đúng, rõ và chỉ còn chỉnh sửa cục bộ. |
| 4 | A*-ready: reviewer có thể kiểm tra và khó đưa ra phản biện nghiêm trọng. |

Điểm quy đổi của một nhóm:

`weighted score = weight x rating / 4`.

### 4.2 Rubric theo nhóm

| ID | Nhóm tiêu chí | Trọng số | Điều kiện để đạt 4/4 |
|---|---|---:|---|
| A | Ý nghĩa bài toán và venue fit | 8 | Vấn đề quan trọng, tác động vận hành cụ thể, fit SOICT rõ và có một thesis xuyên suốt. |
| B | Novelty và định vị literature | 16 | So sánh đúng với các công trình gần nhất; novelty không dựa vào thuật ngữ mới; trả lời thuyết phục vì sao MaxSAT có giá trị khi MIP đã giải được benchmark. |
| C | Tính đúng đắn kỹ thuật | 14 | Formulation, equivalence, staged optimization, encodings và constraints đều đúng, đủ assumptions và khớp code. |
| D | Thiết kế thực nghiệm và baseline | 16 | Hai design tách bạch, paired protocol công bằng, commercial references đúng vai trò, không chọn cấu hình hoặc dataset sau khi xem kết quả. |
| E | Độ mạnh của kết quả và phân tích | 16 | Main claims có effect size, completion, timeout-aware metric, uncertainty và phân tích nguyên nhân hoặc failure cases. |
| F | Giá trị của benchmark và khả năng khái quát | 8 | HCORAP-LC có lý do thiết kế rõ, không tạo lợi thế nhân tạo, có seed separation, comparison với Original và được phát hành đầy đủ. |
| G | Reproducibility và artifact | 10 | Có thể chạy lại từ môi trường sạch và tạo đúng main tables/figures từ raw data với provenance. |
| H | Mạch lập luận và độ dễ đọc | 8 | Thuật ngữ nhất quán, câu trọn ý, contribution-evidence mapping rõ, caption tự giải thích và không viết phòng thủ. |
| I | Hình thức và submission compliance | 4 | Đúng CCIS, trong page limit, single-blind metadata đúng và PDF sạch về thị giác. |
|  | **Tổng** | **100** |  |

### 4.3 Ngưỡng quyết định nội bộ

| Tổng điểm | Trạng thái |
|---:|---|
| 85-100 | `A*-READY`, nếu không fail gate và B, C, D, E đều từ 3 trở lên. |
| 75-84 | `STRONG BUT NOT FROZEN`, cần xử lý các điểm dưới 3. |
| 60-74 | `MAJOR REVISION`, câu chuyện hoặc evidence chưa đủ mạnh. |
| 0-59 | `NOT READY`, cần tái cấu trúc hoặc bổ sung bằng chứng chính. |

## 5. Checklist chi tiết theo góc nhìn reviewer

### A. Problem, significance và thesis

- [ ] Câu đầu giải thích quyết định thực tế cần đưa ra, không bắt đầu bằng SAT.
- [ ] Ba tiêu chí continuity, overtime và compatibility được giải thích bằng
  ý nghĩa vận hành.
- [ ] Gap không chỉ là "chưa dùng Totalizer".
- [ ] Có một câu thesis duy nhất nối objective policy, benchmark load và
  Boolean encoding.
- [ ] Motivation không hứa routing, uncertainty hoặc deployment nếu bài không
  đánh giá chúng.
- [ ] Relevance với optimization và healthcare decision support tại SOICT
  được thấy ngay trong Introduction.

### B. Novelty và related work

- [ ] Literature search bao phủ home-care allocation bằng ILP, CP, SAT/MaxSAT,
  multi-objective và lexicographic optimization.
- [ ] Bản thảo nói rõ lexicographic optimization là kỹ thuật đã biết.
- [ ] Novelty được đặt ở formulation HCORAP, load-calibrated evaluation và
  controlled encoding evidence.
- [ ] Có so sánh với ít nhất 3-5 công trình gần nhất theo problem, objective,
  solver, benchmark và evidence.
- [ ] Bản thảo trả lời trực tiếp câu hỏi: "Nếu Gurobi giải tất cả HCORAP-LC
  instances rất nhanh, tại sao cần MaxSAT?"
- [ ] Vai trò MaxSAT được mô tả bằng giá trị có thể kiểm tra, không bằng claim
  chung về flexibility hoặc scalability.
- [ ] Title không đưa Totalizer thành đóng góp trung tâm nếu hiệu ứng không
  ổn định dưới cả Weighted và LEX-COS.
- [ ] Không dùng "novel", "first", "state of the art" nếu chưa có literature
  evidence đầy đủ.

### C. Formulation và solution design

- [ ] `allocation`, `service`, `continuity group`, `regular capacity` và
  `overtime capacity` được dùng nhất quán.
- [ ] Mỗi equation được giải thích bằng một câu có ý nghĩa nghiệp vụ.
- [ ] Direction của `CONT`, `OT`, `SIM` luôn rõ.
- [ ] Weighted objective và LEX-COS được so sánh trên cùng feasible set.
- [ ] Có lập luận ngắn chứng minh staged procedure trả lexicographic optimum
  khi cả ba stage hoàn thành.
- [ ] Cumulative timeout được định nghĩa một lần và dùng nhất quán.
- [ ] Encoding section đủ để một chuyên gia SAT kiểm tra semantics.
- [ ] Optional constraints được mô tả theo tác dụng logic, không theo tên hàm
  hoặc tên config trong source code.

### D. Experimental design và baselines

- [ ] RQ1 chỉ đo effect của objective policy trên HCORAP-LC.
- [ ] RQ2 chỉ đo effect của cardinality encoding khi các yếu tố khác cố định.
- [ ] Gurobi và CPLEX được gọi là exact MIP references hoặc validators khi
  không có runtime comparison công bằng.
- [ ] Nếu paper đóng góp một exact solution method cạnh tranh, phải báo runtime
  với MIP/CP baseline trên cùng protocol.
- [ ] Nếu paper không claim competitiveness, phải nêu chính xác giá trị khoa
  học của MaxSAT formulation.
- [ ] Instance selection, seeds, configuration selection và stopping rule được
  quyết định trước main campaign.
- [ ] Timeout phù hợp với nghiên cứu trước và áp dụng như nhau trong từng
  direct comparison.
- [ ] Task order được randomized hoặc blocked để giảm ảnh hưởng VM drift.
- [ ] Mỗi run dùng một thread hoặc resource allocation được ghi rõ.
- [ ] Infeasible instances không bị bỏ khỏi completion và PAR-2.

### E. Results và statistical analysis

- [ ] Mỗi subsection mở đầu bằng câu trả lời trực tiếp cho RQ.
- [ ] Báo số optimum, infeasible, timeout và completed runs.
- [ ] Báo PAR-2 cho dữ liệu có timeout.
- [ ] Paired speedup chỉ dùng các cặp có định nghĩa hợp lệ và nêu rõ sample.
- [ ] 95% confidence interval ghi rõ bootstrap method và đơn vị resampling.
- [ ] Báo cả effect size và uncertainty, không chỉ p-value.
- [ ] Phân tích theo instance size hoặc load để giải thích khi nào hiệu ứng
  mạnh hoặc yếu.
- [ ] Có ít nhất một phân tích timeout hoặc hard-instance behavior.
- [ ] Compatibility loss được diễn giải như chi phí của priority policy.
- [ ] LEX-OT được dùng đúng vai trò sensitivity, không thành RQ thứ ba.
- [ ] IC và SB được dùng để giải thích configuration selection, không chiếm
  vị trí ngang với hai main studies.
- [ ] Kết quả neutral hoặc bất lợi được giải thích trung thực.
- [ ] Không so sánh trực tiếp runtime giữa solver, hardware hoặc timeout khác
  nhau.

### F. Benchmark validity

- [ ] Giải thích vì sao demand-to-capacity ratio gần 0.85 có ý nghĩa.
- [ ] Cho thấy HCORAP-LC kích hoạt trade-off mà Original không kích hoạt.
- [ ] Chứng minh generator không thay đổi hard model hoặc objective semantics.
- [ ] Mỗi instance có feasible witness được checker xác nhận.
- [ ] Generation seeds của development và evaluation được tách.
- [ ] Không giảm dataset sau khi quan sát solver outcome.
- [ ] Artifact chứa generator command và exact instance hashes.
- [ ] Discussion nói rõ phạm vi synthetic và hướng đánh giá trên operational
  data.

### G. Reproducibility

- [ ] README bắt đầu bằng một command chạy smoke test.
- [ ] README có command chạy Design A, Design B và analysis.
- [ ] Có version hoặc hash của EvalMaxSAT, Gurobi, CPLEX, compiler và OS.
- [ ] License của code, benchmark và third-party solver được ghi rõ.
- [ ] Raw logs giữ status, runtime, memory, objective và timeout metadata.
- [ ] Analysis scripts kiểm tra duplicate, missing rows và hash mismatch.
- [ ] Manuscript values được sinh từ validated CSV/JSON.
- [ ] Figure và table scripts chạy không cần sửa đường dẫn bằng tay.
- [ ] Artifact có một small test hoàn thành trong thời gian hợp lý.
- [ ] Full campaign có ước lượng CPU-hour, disk và memory.

### H. Writing và visual presentation

- [ ] Title phản ánh contribution mạnh nhất, không liệt kê feature.
- [ ] Abstract theo mạch problem -> gap -> approach -> protocol -> main result
  -> implication.
- [ ] Introduction kết thúc bằng 3 contributions có evidence tương ứng.
- [ ] Related Work tổng hợp khoảng trống, không chỉ tóm tắt từng paper.
- [ ] Không dùng thuật ngữ trước khi định nghĩa.
- [ ] Không đổi qua lại giữa schedule, assignment và allocation cho cùng một
  khái niệm.
- [ ] Mỗi câu có chủ ngữ và động từ chính rõ.
- [ ] Không có câu mang tính quy trình nội bộ hoặc biện hộ.
- [ ] Hạn chế chữ in đậm trong paragraph và không dùng em dash.
- [ ] Caption nêu sample, metric, hướng tốt và điều kiện so sánh.
- [ ] Bảng dùng số chữ số thập phân nhất quán.
- [ ] Hình đọc được khi in grayscale và ở kích thước CCIS thực tế.
- [ ] Discussion trả lời "kết quả có ý nghĩa gì" thay vì lặp lại bảng.
- [ ] Scope và next steps được tích hợp tự nhiên trong Discussion.
- [ ] Conclusion không thêm claim hoặc con số mới.

## 6. Claim-evidence matrix bắt buộc

Điền matrix này trước mỗi data freeze.

| Claim | Evidence bắt buộc | Vị trí dự kiến | Trạng thái hiện tại |
|---|---|---|---|
| LEX-COS làm giảm continuity theo đúng priority | 48 paired Gurobi optima, delta distribution | Table 1, Fig. 1 | `PASS` |
| LEX-COS ảnh hưởng overtime và compatibility | Paired deltas, median, IQR, count improved/worsened | Table 1, Results | `PASS` |
| Priority order có hoặc không tạo khác biệt | LEX-COS và LEX-OT trên cùng 48 instances | Results sensitivity paragraph | `PASS` |
| HCORAP-LC làm objective trade-off quan sát được | Original vs HCORAP-LC objective activity và generation protocol | Methodology, Discussion | `PASS` |
| Totalizer tốt hơn sorting network dưới Weighted | 3600 s paired SN/TOT result và 95% CI | RQ2 table/figure | `PASS`: 1,14; CI [1,08; 1,23] |
| Encoding effect chuyển sang LEX-COS | 3600 s LEX-COS SN/TOT result và 95% CI | RQ2 table/figure | `PASS`: 1,04; CI [1,03; 1,05] |
| Objective values đúng | Gurobi/CPLEX agreement và independent checker | Independent Validation | `PASS` |
| Artifact tái tạo được main evidence | Clean-environment rerun và generated manuscript files | Artifact README/provenance | `PARTIAL` |

## 7. Sáu câu hỏi phản biện khó phải trả lời được

Mỗi câu phải có câu trả lời 2-4 câu và một pointer đến evidence.

1. Đóng góp mới là gì nếu lexicographic optimization và Totalizer đều đã được
   biết?
2. Vì sao dùng MaxSAT khi Gurobi giải toàn bộ policy study nhanh hơn rõ rệt?
3. HCORAP-LC phản ánh workload hợp lý hay được thiết kế để tạo kết quả thuận
   lợi cho LEX-COS?
4. Totalizer chỉ tốt dưới weighted objective hay hiệu ứng còn giữ dưới
   LEX-COS?
5. Kết quả trên 48 synthetic instances có thể hỗ trợ claim nào và không hỗ
   trợ claim nào?
6. Một reviewer độc lập có thể tái tạo Table 1, Table 2 và các figure từ raw
   data bằng những command nào?

### Câu trả lời dựa trên evidence, cập nhật 07/09/2026

1. Lexicographic optimization và Totalizer không phải novelty độc lập. Đóng góp
   là chính sách ưu tiên dành riêng cho HCORAP, benchmark HCORAP-LC làm các ưu
   tiên đó đo được, và thí nghiệm kiểm soát cho biết lựa chọn encoding có chuyển
   từ Weighted sang staged LEX-COS hay không. Evidence nằm ở Introduction,
   Problem and Objective Policies, Table 1 và Table 2.
2. Gurobi được dùng để tạo objective-quality evidence vì nó giải Study A nhanh
   và chứng minh tối ưu. MaxSAT được giữ để kế thừa mô hình Boolean của nghiên
   cứu gốc, thực hiện cùng policy bằng staged optimization và nghiên cứu trực
   tiếp cardinality encoding. Bài không claim MaxSAT nhanh hơn MIP; xem đoạn
   `Role of MaxSAT and exact MIP` trong Discussion.
3. HCORAP-LC cố ý kích hoạt áp lực tài nguyên, nhưng không ưu ái một policy: mỗi
   cặp policy nhận đúng cùng instance, feasible witness được tạo trước, và cả
   hai nghiệm đều được chứng minh tối ưu. LEX-COS cũng trả giá 6,3% compatibility,
   nên dữ liệu không chỉ tạo kết quả thuận lợi; xem Methodology và RQ1.
4. Hiệu ứng giữ dưới cả hai policies. Weighted có median SN/TOT là 1,14 với CI
   [1,08; 1,23], còn LEX-COS là 1,04 với CI [1,03; 1,05]; completion không giảm
   và PAR-2 giảm ở cả hai. Evidence nằm ở Table 2, Figure 2 và
   `policy_encoding_contrasts.csv`.
5. 48 synthetic instances hỗ trợ kết luận nội bộ về ba objective policies, hai
   encodings và EvalMaxSAT trên hai benchmark đã định nghĩa. Chúng không hỗ trợ
   kết luận về dữ liệu bệnh viện thực, routing, uncertainty hoặc ưu thế phổ quát
   của MaxSAT/Totalizer. Phạm vi này được tích hợp trong Discussion.
6. Từ repository root, reviewer chạy:

   ```bash
   python3 experiments/generate_compact_manuscript_results.py \
     --policy-analysis results_v2/gcp_corrected_exact_analysis \
     --encoding-analysis \
       hcorap_compact_policy_encoding_3600_20260907_082501/analysis \
     --output LaTeX-Templates/paper/generated_compact
   cd LaTeX-Templates/paper
   latexmk -pdf -interaction=nonstopmode -halt-on-error main_soict.tex
   ```

   Generator kiểm tra hai evidence gates, sinh hai bảng, các số và tọa độ dùng
   cho hai figures, đồng thời ghi SHA-256 của mọi input vào provenance.

Nếu một câu chưa có câu trả lời dựa trên evidence, bản thảo chưa đạt
`A*-READY`.

## 8. Phiếu chấm

Sao chép bảng này cho mỗi vòng review.

| Trường | Giá trị |
|---|---|
| Ngày review | |
| Reviewer | |
| Source commit | |
| Result freeze ID | |
| PDF SHA-256 | |

| ID | Trọng số | Điểm 0-4 | Điểm quy đổi | Evidence | Action |
|---|---:|---:|---:|---|---|
| A | 8 | | | | |
| B | 16 | | | | |
| C | 14 | | | | |
| D | 16 | | | | |
| E | 16 | | | | |
| F | 8 | | | | |
| G | 10 | | | | |
| H | 8 | | | | |
| I | 4 | | | | |
| **Tổng** | **100** | | | | |

Quy tắc review độc lập:

- hai reviewer không xem điểm của nhau trước khi hoàn thành phiếu;
- nếu một tiêu chí lệch từ 2 điểm trở lên, hai reviewer phải đối chiếu evidence;
- điểm cuối không lấy trung bình máy móc nếu một reviewer phát hiện lỗi
  correctness hoặc claim integrity;
- freeze chỉ được ký khi mọi action mức `BLOCKER` và `MAJOR` đã đóng.

## 9. Baseline audit ngày 04/09/2026

Đây là chấm sơ bộ để định hướng công việc, không phải điểm cố định.

| ID | Điểm | Nhận định ngắn |
|---|---:|---|
| A | 3/4 | Bài toán và ba tiêu chí đã rõ; venue fit tốt. |
| B | 1/4 | Related Work còn mỏng và chưa trả lời đủ giá trị của MaxSAT khi MIP rất mạnh. |
| C | 3/4 | Formulation và policy rõ; phần equivalence của encoding/optional constraints còn ngắn. |
| D | 2/4 | Design A mạnh; Design B 3600 s chưa có trong manuscript evidence. |
| E | 2/4 | Policy analysis tốt; encoding analysis mới có Weighted 300 s và còn ít hardness analysis. |
| F | 2/4 | HCORAP-LC kích hoạt trade-off nhưng cần biện minh sâu hơn cho ratio và external validity. |
| G | 3/4 | Runner, validation và provenance đã có cấu trúc tốt; cần clean-machine reproduction. |
| H | 3/4 | Bản thảo đã dễ đọc hơn; novelty narrative và result depth vẫn cần tăng. |
| I | 3/4 | Đang dùng LLNCS/CCIS và không có page number; cần final template/font/compliance check. |
| **Tổng** | **58/100** | `NOT READY` theo chuẩn A*. |

Gate hiện tại:

- `G0`: `PARTIAL`, cần xác nhận final CCIS package và submission metadata;
- `G1`: `PARTIAL`, MaxSAT value proposition chưa đủ mạnh;
- `G2`: `PARTIAL`, cần equivalence audit từ code đến paper;
- `G3`: `FAIL`, thiếu Design B 3600 s;
- `G4`: `PARTIAL`, chưa có clean-machine reproduction report;
- `G5`: `PARTIAL`, văn phong tốt hơn nhưng evidence và novelty story chưa đủ.

### Cập nhật triển khai ngày 07/09/2026

- `G3`: `PASS`. Design B có đủ 192 EvalMaxSAT và 96 Gurobi records ở 3600 s;
  `evidence_valid=true`, 0 objective mismatch và 0 status contradiction.
- Hai Totalizer claim gates đều đạt; Table 2 và Figure 2 đã thay toàn bộ evidence
  sơ bộ 300 s.
- Title, Abstract, contributions, Results, Discussion và Conclusion đã được
  đồng bộ với cùng claim-evidence matrix.
- Table 1, Table 2 và dữ liệu của cả hai figures được sinh từ validated CSV;
  build dừng nếu thiếu generated macros.
- PDF có 13 trang, trong đó phần thân kết thúc trong trang 12 và references bắt
  đầu ở trang 12; không có overflow và mọi font đều embedded.
- Không chấm lại tổng điểm cho tới vòng review độc lập; `G0`, `G2` và `G4` vẫn
  cần final template check, code-to-paper audit và clean-machine reproduction.

## 10. Thứ tự ưu tiên đến submission

### Blocker

- [x] Hoàn thành và validate Design B 3600 s.
- [x] Sinh lại RQ2 table/figure từ frozen results, không sửa số bằng tay.
- [x] Khóa định vị: đây là HCORAP-specific MaxSAT formulation và evaluation,
  không phải thuật toán lexicographic tổng quát mới.
- [x] Hoàn thiện literature review về ILP/CP/MaxSAT và lexicographic home-care
  optimization.
- [x] Viết câu trả lời dựa trên evidence cho sáu câu hỏi ở Mục 7.

### Major

- [ ] Bổ sung phân tích theo size/load/hardness và timeout stage.
- [x] Chứng minh rõ benchmark construction không tạo ưu thế cho một solver
  hoặc policy.
- [ ] Chạy artifact trên VM sạch và lưu reproduction report.
- [x] Đồng bộ title, abstract, contributions, Results, Discussion và Conclusion
  với cùng claim-evidence matrix.

### Final polish

- [x] Kiểm tra PDF ở 100%; bản in grayscale vẫn cần reviewer độc lập xác nhận.
- [x] Kiểm tra BibTeX, DOI, author order, affiliation và corresponding author.
- [ ] Xác nhận page count không tính references.
- [ ] Xóa auxiliary files và ảnh preview khỏi submission bundle.
- [ ] Tạo source archive tối thiểu có thể biên dịch trên môi trường sạch.

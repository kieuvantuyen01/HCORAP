# Hướng dẫn biên soạn bản thảo SOICT 2026

Quality gate và rubric kiểm tra nội bộ nằm trong
[`docs/SOICT_2026_A_STAR_QUALITY_RUBRIC.md`](docs/SOICT_2026_A_STAR_QUALITY_RUBRIC.md).

Tài liệu này quy định cách trình bày. Thiết kế nghiên cứu, ma trận thực nghiệm,
lịch chạy và deadline nằm trong [`submission_plan.md`](submission_plan.md).
Protocol thực thi chi tiết nằm trong
[`docs/GCP_EXPERIMENT_RUNBOOK.md`](docs/GCP_EXPERIMENT_RUNBOOK.md).

## 1. Phạm vi và định dạng

- Venue: SOICT 2026.
- Phạm vi phù hợp: Applied Operations Research and Optimization, healthcare
  applications và decision support.
- Ngôn ngữ: tiếng Anh học thuật.
- Bản thảo dùng Springer CCIS/LNCS template, tối đa 12 trang không tính phần
  tài liệu tham khảo.
- Quy trình review là single-blind, vì vậy bản nộp giữ tên và affiliation của
  tác giả.
- Bản nộp là PDF và không có số trang.
- Abstract deadline: 09/09/2026; full-paper deadline: 16/09/2026. Phải kiểm
  tra lại giờ đóng hệ thống submission trước ngày nộp.
- Source đang hoạt động là `LaTeX-Templates/paper/main_soict.tex`.
- Không tự điền DOI, ISBN hoặc copyright metadata trước khi hội nghị cung cấp.
- Chỉ dùng kết quả đã qua validation và data freeze; không đưa pilot hoặc raw
  runtime thiếu provenance vào bảng chính.

### Source và bản build chính thức

- Chỉ cập nhật `main_soict.tex`. Các file `main.tex`, `review.tex` và
  `submission.tex` thuộc pipeline cũ và không phải source của bản SOICT.
- Build bằng lệnh được ghi trong `LaTeX-Templates/paper/README.md`.
- Các bảng, hình và macro định lượng phải được sinh từ frozen results. Không
  điền tay con số vào Abstract, Results, Discussion hoặc Conclusion.
- Trước khi nộp, lưu source commit, result freeze ID và SHA-256 của PDF.

Page budget định hướng cho phần nội dung chính: khoảng 2 trang cho Abstract và
Introduction; 1 trang cho Related Work; 3 trang cho formulation và solution
design; 1,5 trang cho methodology; 3 trang cho Results; 1,5 trang cho Discussion
và Conclusion. Đây là ngân sách biên tập, không phải lý do cắt bỏ định nghĩa,
bằng chứng hoặc phân tích cần thiết. References nằm ngoài giới hạn 12 trang.

## 2. Tác giả và đơn vị

Thứ tự tác giả dự kiến:

1. Tuyen Van Kieu: `tuyenkv@vnu.edu.vn`;
2. Khanh Ngoc Do: `2302061@vnu.edu.vn`;
3. Khanh Van To: `khanhtv@vnu.edu.vn`.

Affiliation thống nhất:

```text
Faculty of Information Technology,
VNU University of Engineering and Technology, Vietnam
```

Phải xác nhận lại thứ tự tác giả, corresponding author, ORCID và cách viết tên
tiếng Anh với cả ba đồng tác giả trước khi đăng ký abstract.

Với `llncs`, liệt kê các tác giả trong một khối `\author` và ngăn cách bằng
`\and`. Vì cả ba tác giả cùng một đơn vị, dùng một khối `\institute` chung.
Dùng nhất quán tên không dấu ở PDF, submission system, ORCID và artifact
metadata; `\authorrunning` là `Kieu et al.`.

## 3. Cấu trúc bản thảo

### Abstract

Nêu ngắn gọn bài toán, hạn chế của weighted objective/encoding gốc, bốn thành
phần đóng góp, protocol và các kết quả định lượng chính. Không đưa số liệu chưa
được sinh từ frozen tables. Tránh claim “state of the art” nếu không có benchmark
và baseline tương ứng. Dùng abstract draft không số trong `main_soict.tex` ngay từ
giai đoạn chạy; sau data freeze chỉ thay một câu bằng 2--3 kết quả truy vết được.

### Introduction

Mạch lập luận nên theo thứ tự:

1. HCORAP và ý nghĩa vận hành;
2. cách tiếp cận MaxSAT hiện có;
3. sự mơ hồ của weighted optimum và chi phí encoding;
4. khoảng trống về đánh giá objective policy và cardinality encoding trong
   HCORAP;
5. contributions và research questions.

### Related Work

Dựa trên [`literature_review.md`](literature_review.md). Phải có một bảng phân
loại nghiên cứu liên quan theo các trục: home-care assignment/scheduling,
routing, uncertainty, multiobjective/lexicographic optimization, MaxSAT
encoding, implied constraints và symmetry breaking. Mỗi citation phải hỗ trợ
trực tiếp cho câu chứa citation; kiểm tra DOI/BibTeX trước data freeze.

### Problem and Methods

- định nghĩa tập, tham số, biến và hard constraints nhất quán với model/code;
- định nghĩa rõ assignment-suitability reward `SIM`, caregiver-fragmentation
  penalty `CONT`, excess-workload metric `OT` và per-unit penalty `P`;
- nêu quan hệ hằng số giữa stability reward của bài gốc và `CONT` trong điều
  kiện full coverage;
- không gọi `OT` là số giờ overtime nếu mô hình chưa định nghĩa mỗi service có
  duration đúng một giờ;
- phân biệt weighted B0 với `LEX-COS = CONT -> OT -> SIM`;
- trình bày Totalizer, implied constraints và symmetry-breaking constraints cho
  các lớp tương đương được phát hiện;
- nêu điều kiện bảo toàn nghiệm/optimum, không chỉ mô tả trực giác;
- dùng pseudocode ngắn cho staged lexicographic optimization nếu cần.

### Experimental Setup

Ghi rõ benchmark split, baseline/composite configurations, EvalMaxSAT SHA-256, GCP
machine, một worker/thread, timeout, task ordering, verifier, Gurobi/CPLEX
validation settings và exclusion rules. Phân biệt rõ screen, development,
calibration, evaluation và primary data. Chỉ dùng `confirmatory` khi phạm vi,
hypotheses và analysis đã được khóa trước primary runs.

### Results

Ưu tiên hai nghiên cứu chính và một lớp validation:

1. ảnh hưởng của weighted và LEX-COS trên 48 HCORAP-LC instances;
2. ảnh hưởng của sorting network và Totalizer dưới cả weighted và LEX-COS trên
   48 Original instances;
3. Gurobi, CPLEX và independent solution checker chỉ đóng vai trò xác nhận tính
   đúng của objective values và nghiệm.

Các nhánh epsilon, weight, uncertainty và routing nằm ngoài measured compact
campaign. Báo cả timeout, PAR-2, peak RSS và số cặp cùng optimum; không chỉ báo
trung bình trên các run giải được.

Thứ tự trình bày theo claim thay vì liệt kê log chạy: (i) LEX-COS thay đổi ba
tiêu chí như thế nào, (ii) Totalizer thay đổi hiệu năng và kích thước công thức
như thế nào dưới từng policy, và (iii) các kiểm tra độc lập xác nhận objective
values. Implied constraints, symmetry breaking và LEX-OT chỉ là supporting
evidence.

Main paper dự kiến giữ bốn visual: policy-effect table, CONT/OT scatter plot,
four-cell encoding table và paired speedup plot có 95% confidence intervals.
Cactus plot chỉ thay thế paired speedup plot nếu nó giải thích timeout tốt hơn;
không thêm visual chỉ để tăng số lượng. Các số và phần prose định lượng phải
được sinh từ frozen results và có provenance.

### Discussion and Conclusion

Discussion giải thích ý nghĩa vận hành của policy trade-off, điều kiện encoding
effect chuyển giữa hai policies và vai trò khác nhau của Original với HCORAP-LC.
Các giới hạn về synthetic instances, solver/hardware dependence, routing,
uncertainty và operational data được tích hợp vào Discussion, không tạo một
section `Limitations and Threats to Validity` riêng. Conclusion chỉ nhắc lại
claim đã có bảng hoặc hình hỗ trợ.

## 4. Quy tắc văn phong

- dùng thuật ngữ nhất quán xuyên suốt;
- câu phải trọn ý và có chủ ngữ–vị ngữ;
- ưu tiên câu ngắn, tránh ghép quá nhiều mệnh đề;
- hạn chế in đậm trong paragraph và tránh em dash nếu không cần thiết;
- không dùng “obviously”, “clearly”, “always” hoặc “significantly” khi chưa có
  lập luận/kiểm định hỗ trợ;
- phân biệt *solver-reported optimum*, *best incumbent*, *timeout* và
  *infeasible*;
- gọi verifier hiện tại là *independent solution verification*, không phải
  independent optimality certification hoặc proof checking;
- không gọi availability stress test là robust optimization;
- không gọi coordinate-based similarity là routing.

## 5. Hình, bảng và khả năng tái lập

- giữ tối đa bốn visual chính đã nêu trong phần Results; mỗi visual phải trả lời
  một câu hỏi nghiên cứu hoặc hỗ trợ một claim;
- tránh các nhãn nội bộ khó hiểu như `cross-paradigm`, `treatment bundle` hoặc
  `reference configuration` khi có thể nói trực tiếp solver/cấu hình nào được
  so sánh;
- mọi bảng/hình phải được sinh từ script và frozen raw data;
- caption phải tự giải thích được metric, sample và hướng tốt/xấu;
- biểu đồ dùng font và kích thước đọc được trong bố cục CCIS;
- dùng bảng thay cho hình khi cần đối chiếu nhiều giá trị chính xác;
- color palette phải phân biệt được khi in grayscale;
- không chỉnh tay số trong LaTeX sau khi collector đã sinh bảng;
- artifact phải chứa commit, configs, instances, raw JSON/logs, environment và
  SHA-256 checksum.

## 6. Checklist biên tập

- [ ] Title phản ánh lexicographic objective và MaxSAT enhancements.
- [ ] Abstract không chứa số chưa data-freeze.
- [ ] Contributions khớp trực tiếp với RQs và result tables.
- [ ] Ký hiệu toán học khớp code và dùng nhất quán.
- [ ] Related-work table có citation đầy đủ, không biến thành danh sách tóm tắt.
- [ ] Methods đủ chi tiết để tái triển khai.
- [ ] Experimental setup khớp hai thiết kế đã khóa: Design A gồm 48 cặp policy;
      Design B gồm 48 instances x 2 policies x 2 encodings ở 3.600 s.
- [ ] Mọi dòng `OPTIMUM` trong bảng có nghiệm qua independent solution verifier;
      không gọi đó là independently certified optimum nếu chưa kiểm tra proof
      trace.
- [ ] Discussion tích hợp phạm vi áp dụng và các giới hạn về routing,
      uncertainty, synthetic instances và solver/hardware dependence.
- [ ] BibTeX/DOI, author metadata, page limit và PDF fonts đã được kiểm tra.

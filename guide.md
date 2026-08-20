# Hướng dẫn biên soạn bản thảo ICIIT 2027

Tài liệu này quy định cách trình bày. Thiết kế nghiên cứu, ma trận thực nghiệm,
lịch chạy và deadline nằm trong [`submission_plan.md`](submission_plan.md).
Protocol thực thi chi tiết nằm trong
[`docs/GCP_EXPERIMENT_RUNBOOK.md`](docs/GCP_EXPERIMENT_RUNBOOK.md).

## 1. Phạm vi và định dạng

- Track: ICIIT 2027 Conference Proceedings.
- Ngôn ngữ: tiếng Anh học thuật.
- Mục tiêu: 5 trang double-column, bao gồm hình, bảng và tài liệu tham khảo.
- Dùng LaTeX template do hội nghị cung cấp trong `LaTeX-Templates/`.
- Không tự điền DOI, ISBN hoặc copyright metadata trước khi hội nghị cung cấp.
- Chỉ dùng kết quả đã qua validation và data freeze; không đưa pilot hoặc raw
  runtime thiếu provenance vào bảng chính.

### Ba chế độ build bắt buộc

- `main.tex` tạo PDF sạch để đánh giá hình thức trước data freeze. PDF này không
  hiển thị ghi chú biên tập hoặc placeholder nhưng chưa phải bản submission.
- `review.tex` chỉ dùng nội bộ để kiểm tra cấu trúc bằng chứng đang chờ. Không
  gửi PDF này cho hội nghị.
- `submission.tex` là release build duy nhất được phép gửi. Build phải dừng với
  lỗi nếu thiếu bất kỳ frozen file bắt buộc nào.
- Abstract findings, Results và Conclusion chỉ được nạp vào `main.tex` khi bộ ba
  file trong `LaTeX-Templates/paper/generated/` đã được sinh đầy đủ từ frozen
  artifacts. Không điền số thủ công để làm bản thảo trông hoàn chỉnh.

Page budget mục tiêu: khoảng 0,8 trang cho title/abstract/Introduction; 0,5
trang cho Related Work; 1,5--1,7 trang cho model và phương pháp; 0,6--0,8 trang
cho experimental setup; tối thiểu 1,2 trang cho Results/Conclusion; phần còn lại
cho limitations và references. Trước data freeze, khoảng trống trong PDF sạch
là ngân sách dành cho Results, không phải phần cần lấp bằng background.

## 2. Tác giả và đơn vị

Thứ tự tác giả dự kiến:

1. Tuyen Van Kieu — `tuyenkv@vnu.edu.vn`;
2. Khanh Ngoc Do — `2302061@vnu.edu.vn`;
3. Khanh Van To — `khanhtv@vnu.edu.vn`.

Affiliation thống nhất:

```text
Faculty of Information Technology,
VNU University of Engineering and Technology, Vietnam
```

Phải xác nhận lại thứ tự tác giả, corresponding author, ORCID và cách viết tên
tiếng Anh với cả ba đồng tác giả, rồi khóa metadata nội bộ trước 31/08/2026.

Với `acmart`, giữ mỗi người trong một lệnh `\author` riêng để metadata và chỉ mục
tác giả đúng. Vì cả ba cùng một đơn vị, đặt ba khối `\author`/`\email` liên tiếp
rồi khai báo **một** `\affiliation` chung sau tác giả cuối; đây là cách shared
affiliation trong sample chính thức của class. Không gộp ba tên vào một lệnh
`\author`. Dùng nhất quán tên không dấu ở PDF, submission system, ORCID và
artifact metadata; `\shortauthors` là `Kieu et al.`.

## 3. Cấu trúc bản thảo

### Abstract

Nêu ngắn gọn bài toán, hạn chế của weighted objective/encoding gốc, bốn thành
phần đóng góp, protocol và các kết quả định lượng chính. Không đưa số liệu chưa
được sinh từ frozen tables. Tránh claim “state of the art” nếu không có benchmark
và baseline tương ứng. Dùng abstract draft không số trong `main.tex` ngay từ
giai đoạn chạy; sau data freeze chỉ thay một câu bằng 2--3 kết quả truy vết được.

### Introduction

Mạch lập luận nên theo thứ tự:

1. HCORAP và ý nghĩa vận hành;
2. cách tiếp cận MaxSAT hiện có;
3. sự mơ hồ của weighted optimum và chi phí encoding;
4. khoảng trống về lexicographic policy, Totalizer và constraint strengthening;
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

Ưu tiên ba nhóm bằng chứng:

1. paired factorial/encoding-size performance;
2. weighted so với LEX-COS và LEX-OCS sensitivity;
3. corrected-v2 và Gurobi/CPLEX validation.

Các nhánh epsilon, weight, uncertainty và routing nằm ngoài measured compact
campaign. Báo cả timeout, PAR-2, peak RSS và số cặp cùng optimum; không chỉ báo
trung bình trên các run giải được.

Thứ tự trình bày bắt buộc là RQ1 (objective policy), RQ2 (Totalizer), RQ3
(implied/symmetry và interactions), rồi validation/scope. Bản 5 trang dùng đúng
hai visual kết quả, đều là bảng full-width:

1. bảng factorial hai panel: đủ tám cell; bốn direct-factor contrasts tiêu biểu
   kèm full set 12 contrasts trong artifact; và một dòng end-to-end B--R tái sử
   dụng đúng 80 instances của factorial;
2. bảng policy/validation ba panel: proved count, both-optimum, objective deltas
   và PAR-2; corrected-v2; agreement EvalMaxSAT/Gurobi/CPLEX.

Không thêm biểu đồ Pareto, cactus/scatter hoặc weight-sensitivity vào main paper
trừ khi thay thế một trong ba visual hiện có và chứng minh được lượng thông tin
tăng lên. Các số và cả đoạn prose định lượng phải do
`experiments/generate_manuscript_results.py` sinh; data-freeze phải kiểm tra
`manuscript-provenance.json`, không chỉ kiểm tra file LaTeX tồn tại.

### Threats and Conclusion

Nêu thẳng các giới hạn: benchmark gốc ít excess workload, corrected-v2 là
synthetic, LEX chạy theo staged solving, một solver MaxSAT/version, tập
cross-solver validation nhỏ, không có routing và uncertainty campaign xác nhận.
Conclusion chỉ nhắc lại claim đã có bảng/hình hỗ trợ.

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

- tối đa ba visual trong bản 5 trang: related-work taxonomy, factorial summary,
  và một bảng ba panel gộp policy với corrected/commercial validation;
- mọi bảng/hình phải được sinh từ script và frozen raw data;
- caption phải tự giải thích được metric, sample và hướng tốt/xấu;
- biểu đồ dùng font/kích thước đọc được ở double-column;
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
- [ ] Experimental setup khớp exact scope trong `screening_decision.json`
  (đúng 1.270 measured runs).
- [ ] Mọi dòng `OPTIMUM` trong bảng có nghiệm qua independent solution verifier;
      không gọi đó là independently certified optimum nếu chưa kiểm tra proof
      trace.
- [ ] Threats đề cập routing, uncertainty và synthetic corrected-v2.
- [ ] BibTeX/DOI, author metadata, page limit và PDF fonts đã được kiểm tra.

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

## 2. Tác giả và đơn vị

Thứ tự tác giả dự kiến:

1. Tuyên Văn Kiều — `tuyenkv@vnu.edu.vn`;
2. Khánh Ngọc Đỗ — `2302061@vnu.edu.vn`;
3. Khánh Văn Tô — `khanhtv@vnu.edu.vn`.

Affiliation thống nhất:

```text
Faculty of Information Technology,
VNU University of Engineering and Technology, Vietnam
```

Phải xác nhận lại thứ tự tác giả, corresponding author, ORCID và cách viết tên
tiếng Anh trước khi upload.

## 3. Cấu trúc bản thảo

### Abstract

Nêu ngắn gọn bài toán, hạn chế của weighted objective/encoding gốc, bốn thành
phần đóng góp, protocol và các kết quả định lượng chính. Không đưa số liệu chưa
được sinh từ frozen tables. Tránh claim “state of the art” nếu không có benchmark
và baseline tương ứng.

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
- định nghĩa rõ `SIM`, `CONT`, `OT`, overtime penalty `P`;
- phân biệt weighted B0 với `LEX-COS = CONT -> OT -> SIM`;
- trình bày Totalizer, implied constraints và exact symmetry breaking;
- nêu điều kiện bảo toàn nghiệm/optimum, không chỉ mô tả trực giác;
- dùng pseudocode ngắn cho staged lexicographic optimization nếu cần.

### Experimental Setup

Ghi rõ benchmark split, baseline/proposed configurations, Open-WBO commit,
GCP machine, một worker/thread, timeout, task ordering, verifier, commercial
settings và exclusion rules. Phân biệt rõ screen, development, calibration,
evaluation và confirmatory data.

### Results

Ưu tiên ba nhóm bằng chứng:

1. paired factorial/encoding-size performance;
2. weighted so với LEX-COS và LEX-OCS sensitivity;
3. corrected-v2 và Gurobi/CPLEX validation.

Epsilon và weight screens chỉ được gọi là exploratory. Báo cả timeout, PAR-2,
peak RSS và số cặp cùng optimum; không chỉ báo trung bình trên các run giải được.

### Threats and Conclusion

Nêu thẳng các giới hạn: benchmark gốc ít overtime, corrected-v2 là synthetic,
LEX chạy theo staged solving, một solver MaxSAT/version, commercial subset nhỏ,
không có routing và uncertainty campaign xác nhận. Conclusion chỉ nhắc lại claim
đã có bảng/hình hỗ trợ.

## 4. Quy tắc văn phong

- dùng thuật ngữ nhất quán xuyên suốt;
- câu phải trọn ý và có chủ ngữ–vị ngữ;
- ưu tiên câu ngắn, tránh ghép quá nhiều mệnh đề;
- hạn chế in đậm trong paragraph và tránh em dash nếu không cần thiết;
- không dùng “obviously”, “clearly”, “always” hoặc “significantly” khi chưa có
  lập luận/kiểm định hỗ trợ;
- phân biệt *optimal solution*, *best incumbent*, *timeout* và *infeasible*;
- không gọi availability stress test là robust optimization;
- không gọi coordinate-based similarity là routing.

## 5. Hình, bảng và khả năng tái lập

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
- [ ] Experimental setup khớp frozen manifest 4.896 runs.
- [ ] Mọi optimum trong bảng đã verified.
- [ ] Threats đề cập routing, uncertainty và synthetic corrected-v2.
- [ ] BibTeX/DOI, author metadata, page limit và PDF fonts đã được kiểm tra.

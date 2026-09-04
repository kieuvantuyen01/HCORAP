# Audit: corrected-v2 lexicographic encoding-transfer pilot

Ngày rà soát: 24/08/2026.

## Kết luận

Pilot hợp lệ về cấu trúc và cho quyết định **STOP**. Không chạy chiến dịch
confirmation 96 rows. Kết quả này đóng nhánh thực nghiệm chuyển giao; nó không
được dùng để tuyên bố speedup hoặc hiệu quả tổng quát trong bản thảo vì toàn bộ
32 runs đều chạm timeout 300 giây và protocol chỉ cho phép claim chuyển giao sau
confirmation.

## Phạm vi và provenance

- Raw results:
  results_v2/gcp_corrected_lex_encoding_transfer_pilot/.
- Analysis:
  results_v2/gcp_corrected_lex_encoding_transfer_pilot_analysis/.
- Ma trận: 16 corrected-v2 critical-load strata của seed 1002, mỗi instance có
  hai cấu hình, tổng 32 runs.
- Policy: LEX-COS (CONT → OT → SIM).
- Cấu hình T0: totalizer / implied none / symmetry none.
- Cấu hình R: totalizer / implied both / symmetry slot-service.
- Timeout: 300 giây; một worker, một CPU affinity.
- Collector validation: complete 32/32, không thiếu, không thừa và không có
  run ID bất hợp lệ.
- Source commit trên GCP:
  bdf0a9b24463f70704ec0116d52d5727ac9f75f3, clean worktree.
- HCORAP binary SHA-256:
  ec8e11fc609f8a920a9ecf58d585e08806b22066a1c3d2f17be7ceb84c613202.
- EvalMaxSAT SHA-256:
  97614c996e1173ca0672ec46da153656046db1d84b9362a8561161ee750779f7.

Tái chạy experiments/analyze_lex_encoding_transfer.py tạo lại đúng báo cáo đã
cung cấp. Analyzer hiện còn kiểm tra rằng hai cấu hình phải đồng ý trên mọi
objective value của phần stage prefix đã hoàn tất.

## Kết quả ghép cặp

| Chỉ tiêu | T0 | R | Diễn giải |
|---|---:|---:|---|
| Runs | 16 | 16 | đủ 16 cặp |
| OPTIMUM ba giai đoạn | 0 | 0 | không có run hoàn tất SIM |
| TIMEOUT | 16 | 16 | toàn bộ chạm 300 giây |
| Stage đã hoàn tất | 2/3 trên mọi run | 2/3 trên mọi run | đều vào stage SIM |
| PAR-2 trung bình | 600 s | 600 s | không phân biệt được hai cấu hình |

Hai cấu hình đồng ý về CONT và OT optimum trên 16/16 cặp. T0 không tiến thêm
giai đoạn ở bất kỳ cặp nào (stage_wins = 0). Trên hai stage đã hoàn tất, R
chậm hơn ở 13/16 cặp nhưng median ratio R/T0 chỉ là 1,003; đây là mô tả pilot,
không phải speedup estimate.

R có formula footprint lớn hơn T0 trên mọi cặp:

- tăng trung vị 4.355,5 variables;
- tăng trung vị 22.813,5 hard clauses;
- tăng trung vị 15,4 MB peak RSS; RSS cao hơn ở 15/16 cặp.

Các con số footprint giải thích vì sao R không tạo tín hiệu để mở rộng pilot,
nhưng không chứng minh T0 nhanh hơn trên corrected-v2 LEX-COS.

## Gate và quyết định

| Gate đã khóa trước | Quan sát | Đạt |
|---|---:|:---:|
| T0 thêm ròng ít nhất 2 optimum | 0 | Không |
| T0 tiến thêm một criterion trên ít nhất 4/16 cặp | 0/16 | Không |
| T0 giảm PAR-2 ít nhất 10% | 0% | Không |

Quyết định của analyzer là **STOP**. Không đặt CONFIRM_LEX_TRANSFER=YES và
không chạy gcp_corrected_lex_encoding_transfer_full.

## Cách sử dụng

1. Giữ raw results, logs, environment, manifest và analysis trong artifact.
2. Không cộng 32 pilot runs vào ma trận 924 measured runs.
3. Không đưa runtime ratio, PAR-2 hay footprint của pilot vào Abstract,
   Results hoặc Conclusion.
4. Trong kế hoạch thực nghiệm, đánh dấu nhánh transfer đã đóng bằng STOP.
5. Nếu sau này cần một claim chuyển giao, phải thiết kế một confirmatory
   experiment mới có đủ 48 paired instances và một protocol không phụ thuộc
   vào việc chọn kết quả thuận lợi từ pilot.

## Vị trí artifact

Hai thư mục đã được đặt dưới results_v2/, cùng results root với chiến dịch chính
đang dùng để sinh bản thảo. Khi đóng gói bộ kết quả cục bộ, đặt
HCORAP_RESULTS_ROOT=results_v2. Trên GCP, pipeline tiếp tục dùng
experiments/results/ mặc định. Script publication audit coi pilot là
conditional-pilot: kiểm tra đủ 32 raw runs và quyết định STOP nhưng không cộng
nó vào 924 measured runs. Không đổi tên raw files hoặc sửa các JSON đã thu thập;
dùng SHA-256 và instance_sha256 để duy trì truy vết.

## Preflight đóng gói

Gate riêng của pilot đã pass trong publication audit. Audit toàn cục tại clone
cục bộ vẫn trả về publication_ready = false vì hai source commit của các
campaign cũ (0a264adc... và a4d810b...) không tồn tại trong object database hiện
tại; origin cũng không quảng bá hai commit này. Đây là provenance blocker đã
được ghi trước trong docs/EXPERIMENT_SUPPLEMENT_MATRIX_20260822.md, không phải
lỗi của pilot. Trước khi freeze artifact cuối, cần khôi phục hai commit từ GCP
clone/bundle hoặc một remote lưu trữ chúng. Không sửa tay commit trong
environment.json để vượt gate.

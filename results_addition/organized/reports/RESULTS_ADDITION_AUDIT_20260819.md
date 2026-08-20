# Audit kết quả bổ sung

Thời điểm tạo (UTC): `2026-08-19T00:00:00+00:00`.

> Kết luận sử dụng: **không có campaign nào trong `results_addition/` đủ điều kiện đưa trực tiếp vào bảng kết quả chính của bản thảo theo submission protocol hiện tại.** Dữ liệu gốc được giữ nguyên để làm bằng chứng phát triển, smoke test và chẩn đoán.

## Tóm tắt kiểm kê

- 9,236 file nguồn, tổng dung lượng 29,923,192 byte.
- 9,163 JSON hợp lệ về cú pháp; 0 trạng thái ngoài vocabulary đã biết.
- 11 bảng `results_per_instance.csv`, 11 sổ `.done_runs`, 2 manifest TSV.
- 1 nhóm file trùng nội dung SHA-256; đây là trùng vật lý, tách biệt với dòng manifest/done-runs bị append lặp.
- 3,929 JSON mang trạng thái `OPTIMUM`; 0 trong số đó thiếu cờ kiểm chứng nghiệm.
- 0 nghiệm có weighted-reference formula không khớp; 0 lượt vượt timeout trên 5%.
- 5 ô thí nghiệm có tập instance không cân bằng; 0 nhóm all-optimum bất đồng objective signature; 0 nhóm bất đồng trạng thái đã chứng minh.

## Phân loại campaign

| Campaign | JSON / dự kiến | Trạng thái | Phạm vi có thể tái sử dụng |
|---|---:|---|---|
| `commercial_30_15_4_40_25_5` | 284 / 400 | `exclude-primary` | development diagnostics only |
| `commercial_all_modes_30_15_4_40_25_5` | 3,149 / 3,200 | `exclude-primary` | development and solver-consistency diagnostics only |
| `commercial_main` | 160 / 160 | `exclude-primary` | software correctness smoke tests only |
| `epsilon_8cfg_evalmaxsat` | 1,848 / 10,000 | `exclude-primary` | historical exploratory diagnostics only |
| `gcp_primary_analysis` | 0 | `no-evidence` | none |
| `iciit2027_all_solvers` | 20 / 2,400 | `exclude-primary` | first-20-instance EvalMaxSAT smoke evidence only |
| `lex_8cfg_evalmaxsat` | 1,689 / 4,000 | `exclude-primary` | historical exploratory diagnostics only |
| `main_8cfg_evalmaxsat` | 1,996 / 2,000 | `exclude-primary` | historical encoding diagnostics only |
| `paper_test` | 17 / 17 | `exclude-primary` | software smoke test only |

## Phát hiện bắt buộc xử lý

- **EvalMaxSAT lịch sử:** tổng cộng 5,553 JSON chỉ được giữ làm diagnostic vì khác commit, sampling và provenance. Publication campaign mới vẫn dùng EvalMaxSAT, nhưng phải chạy lại với binary hash và smoke contract đã khóa; không nhập trực tiếp runtime, optimum rate hay Pareto points cũ vào claim chính.
- **Commercial parser/raw mismatch:** `commercial_30_15_4_40_25_5` có 400 dòng CSV, gồm đúng 116 `PARSE_ERROR`, và 400 done markers nhưng chỉ 284 JSON. Không thể truy vết 116 dòng lỗi về raw result; đây chính là commercial dataset lịch sử bị parser loại sai.
- **Commercial campaign bị ngắt:** `commercial_all_modes_30_15_4_40_25_5` dự kiến 3.200 JSON nhưng có 3.149; `.done_runs` bị append lặp và CSV không đồng nhất với số JSON.
- **Manifest bị nhân đôi:** `commercial_main/manifest.tsv` có 320 dòng nhưng chỉ 160 run ID/result path duy nhất. Các JSON tồn tại, song đây là correctness test trên `tests/instances` và source tree được ghi nhận là dirty.
- **Censoring theo thứ tự file:** các campaign lex/epsilon dừng từng cấu hình theo time budget, nên tập instance có thể khác giữa cấu hình và bị phụ thuộc thứ tự tên file. Chỉ phân tích paired intersection nếu dùng cho chẩn đoán; không diễn giải như benchmark confirmatory.
- **Provenance không đồng nhất:** epsilon `delta_0` dùng commit/binary khác bốn delta còn lại; toàn bộ dữ liệu chạy trên Ubuntu 22.04 và không chứng minh đúng máy `c4-highcpu-8`, trong khi protocol hiện tại khóa Ubuntu 24.04, EvalMaxSAT SHA-256 và publication tag sạch.
- **Tên thư mục gây hiểu nhầm:** `iciit2027_all_solvers` mới có 20 EvalMaxSAT run của mode weighted; Gurobi và CPLEX chưa bắt đầu. Không được mô tả là all-solvers result.
- **Epsilon delta_0 có schema thư mục lệch:** xuất hiện cả `cfg1_ORIGINAL` và `cfg5_ORIGINAL`, khác thứ tự cấu hình của các delta còn lại. Việc gộp theo `cfg_id` sẽ sai nếu không chuẩn hóa bằng ba thuộc tính encoding/implied/symmetry.

## Quy tắc sử dụng cho bản thảo

1. Không merge các CSV/XLSX hiện có vào `experiments/results/` của publication campaign.
2. Chỉ trích xuất từ raw JSON khi làm chẩn đoán; luôn lọc theo semantic configuration, không theo `cfg_id`.
3. Không dùng các aggregate epsilon (`epsilon_pareto_frontier.csv`, `epsilon_unique_points.csv`, ...) cho claim Pareto vì nguồn EvalMaxSAT bị loại và campaign bị censor.
4. Kết quả Gurobi/CPLEX cũ có thể dùng để kiểm tra code nội bộ, nhưng bảng chính vẫn phải đến từ publication runner mới với raw JSON, manifest, validation và checksum đầy đủ.
5. Giữ nguyên toàn bộ file nguồn. Mọi bảng dẫn xuất mới phải ghi input SHA-256 từ `checksums/SHA256SUMS` và script tạo bảng.

## Cấu trúc đã tổ chức

- `catalog/file_inventory.jsonl`: mọi file nguồn, kích thước và SHA-256.
- `catalog/json_runs.jsonl`: chỉ mục chuẩn hóa cho từng JSON result.
- `catalog/campaign_summary.json`: phân loại, completeness và provenance theo campaign.
- `quality/*.json`: đối chiếu CSV/JSON, duplicate markers, balance và objective agreement.
- `checksums/SHA256SUMS`: checksum chỉ cho dữ liệu nguồn, không tự bao gồm `organized/`.

## Ghi chú spreadsheet

Ba workbook XLSX trong `main_8cfg_evalmaxsat` được giữ nguyên như output lịch sử. Audit không tạo hoặc sửa workbook; catalog chuẩn được xuất JSON/JSONL để tránh hợp thức hóa các bảng dẫn xuất từ campaign không đủ điều kiện publication.

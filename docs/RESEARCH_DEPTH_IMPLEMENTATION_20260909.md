**Triển khai tăng chiều sâu nghiên cứu HCORAP — 09/09/2026**

Đã bổ sung công cụ để trả lời ba câu hỏi: khác biệt giữa COS và Weighted có còn tồn tại khi xét toàn bộ tập nghiệm đồng tối ưu; cần đánh đổi bao nhiêu CONT để giảm OT; và lợi ích của COS thay đổi thế nào khi năng lực giờ làm thay đổi. Phần MaxSAT kiểm tra một hệ quả cấu trúc của CONT = 0. Chưa đưa kết luận thực nghiệm mới vào bản thảo khi chưa có kết quả được chứng nhận.

**Đã chạy và tận dụng được gì**

- Kiểm tra lại độc lập 144 lịch Gurobi của 48 instance HCORAP-LC bằng Python, đối chiếu coverage, CONT, OT và SIM với JSON gốc. Xuất thêm cấu trúc nhóm chăm sóc, tải của từng nhân viên và khác biệt giữa lịch.
- Xác nhận 3/48 trường hợp COS khác OT → CONT → SIM. Ba trường hợp này đều có thêm một đơn vị CONT khi chuyển sang ưu tiên OT, đổi lại giảm một giờ OT. Thay đổi SIM lần lượt là 0, −3 và +8; không diễn giải thứ tự COS là tốt hơn trên mọi tiêu chí.
- Phân tích 192 lần chạy Original đã căn chỉnh cardinality: 164 OPTIMUM, 24 UNSATISFIABLE, 4 TIMEOUT_FEASIBLE. Có 40 instance với cả bốn tổ hợp policy × encoding được chứng minh tối ưu. Các tỷ số thời gian chính xác chỉ dùng nhóm này; timeout vẫn được giữ riêng và trong PAR2.
- Tạo 432 biến thể năng lực từ 48 instance, với rho ∈ {0.55, 0.85, 0.98} và tỷ lệ giờ thường ∈ {0.70, 0.85, 1.00}. Mọi biến thể giữ nguyên dữ liệu ngoài HN/HE, giữ được lịch khả thi làm chứng. 48 điểm gốc (0.85, 0.85) tái tạo đúng SHA-256 của instance cha.
- Chạy trọn vẹn hai campaign chức năng bằng reference-enumerator: 20 lần cho policy diagnostics và 18 lần cho trọng số, tất cả OPTIMUM. Đây là kiểm thử trên instance nhỏ, không phải bằng chứng thực nghiệm trên bộ đánh giá.

Kết quả phân tích dữ liệu cũ nằm trong `experiments/results/research_depth_existing/`. Dữ liệu và kết quả được sinh vẫn ở các thư mục bỏ qua bởi Git; các chương trình sinh, cấu hình và tài liệu được giữ trong mã nguồn.

**Các thay đổi trong bộ giải**

`hcorap_commercial --method weighted-face --probe-objective continuity|overtime --probe-sense min|max` thực hiện hai bước trong cùng ngân sách thời gian: chứng minh W*, sau đó cố định SIM − wc·CONT − wo·|P|·OT = W* và tìm cực trị được yêu cầu. Chỉ chuyển bước khi bước trước OPTIMUM. Bốn phép khảo sát cho hai khoảng CONT và OT riêng biệt; các đầu mút không nhất thiết thuộc cùng một lịch. Nếu CONT nhỏ nhất trên mặt tối ưu Weighted vẫn lớn hơn CONT của COS, sự mất liên tục không thể giải thích chỉ bằng cách chọn nghiệm đồng tối ưu.

`hcorap_commercial --method continuity-budget --continuity-slack k` chứng minh C*, đặt CONT ≤ C* + k, tối thiểu OT rồi tối đa SIM. k = 0 phải khôi phục vector COS. OT tối ưu không tăng khi k tăng; SIM chỉ được bảo đảm không giảm giữa hai mức k có cùng OT tối ưu. k là số đơn vị phạt CONT tổng cộng được nới, không phải số bệnh nhân. Hai chế độ mới dành cho full coverage và MIP/reference; CP bị từ chối rõ ràng.

`hcorap_multi --zero-continuity-local` thay giới hạn tổng CONT bằng AMO cho từng nhóm khi giới hạn CONT bằng 0. Liên kết hiện có bảo đảm một nhóm hoạt động dùng ít nhất một nhân viên, nhóm không hoạt động không dùng nhân viên nào. Vì từng phần phạt không âm, tổng bằng 0 tương đương mỗi nhóm dùng tối đa một nhân viên. Khi C* > 0 vẫn dùng ràng buộc tổng cũ. Tùy chọn mặc định tắt; SN/TOT, thời gian, IC/SB và các thiết lập khác được giữ giống nhau trong đối chứng.

Runner đưa các tham số mới vào định danh và kiểm tra JSON; collector tách các mức k, chiều khảo sát và trọng số. So sánh backend dùng đúng đại lượng được tối ưu: hai nghiệm continuity-budget có thể cùng OT/SIM nhưng khác CONT trong giới hạn cho phép. Công cụ trọng số đã bỏ cách gọi các cặp cùng wc:wo là “tương đương tỉ lệ”: hệ số SIM luôn bằng 1 nên (1,1) và (8,8) là hai hàm mục tiêu khác nhau.

**Lần kiểm tra EvalMaxSAT trên macOS**

Đã chạy `instance_30_10_4_1.txt` với SN/TOT × global/local, ngân sách 120 giây, một worker. Cả bốn lần chứng minh được C* = 0 và OT* = 0, sau đó binary EvalMaxSAT macOS bị signal 11 ở bước SIM. Bản đối chứng global cũng lỗi. Kết quả này chưa xác định được nguyên nhân trong solver và không chứng minh tăng tốc toàn bộ.

| Mã hóa | Mệnh đề cứng ở bước OT, global | Local |
|---|---:|---:|
| Sorting network | 261.704 | 63.604 |
| Totalizer | 795.466 | 65.782 |

Chi tiết: `experiments/results/research_depth_zero_continuity_local_check/analysis/`. `validation.json` của runner ghi `complete` khi các tác vụ đã kết thúc; cần xem thêm status. Analyzer mới báo `ERROR: 4`, `both_optimum_pairs: 0` và không tính speedup cho những lần này. Kiểm thử tương đương bằng RC2 trên instance nhỏ đã qua. Đo hiệu năng chính cần dùng bản EvalMaxSAT Linux đã dùng trong campaign của bài báo.

**Chuẩn bị trên máy chạy thí nghiệm**

Script điều phối đầy đủ cho máy ảo GCP là
`experiments/run_research_depth_gcp.sh`. Script kiểm tra Linux x86-64, cấu hình
CPU/RAM/ổ đĩa, giấy phép Gurobi/CPLEX, binary EvalMaxSAT, hash đầu vào, build
sạch, kiểm thử, dry-run toàn bộ ma trận và chạy từng campaign theo cơ chế
resume. Mỗi thành phần được collect và phân tích trước khi chuyển tiếp.

Chạy từ thư mục gốc repository. Cần có bộ HCORAP-LC và archive `results_v2/gcp_commercial_corrected_primary` để kiểm tra hash nguồn khi sinh lại biến thể. Nếu thư mục sweep đã có, giữ nguyên hoặc dùng một thư mục mới; chương trình không ghi đè sweep đang tồn tại.

```bash
python3 experiments/generate_load_sweep.py
python3 experiments/prepare_research_depth_campaigns.py
```

Hai lệnh trên vẫn có thể chạy riêng. Script GCP sẽ tự sinh load sweep nếu chưa
có; các cấu hình campaign đã được lưu trong repository nên không sinh lại trong
một measured run.

Thiết lập một lần trên VM sau khi clone đúng commit và chép bộ instance cùng
archive corrected 144-run vào đúng vị trí:

```bash
export GUROBI_HOME=/opt/gurobi/linux64
export CPLEX_STUDIO_DIR=/opt/ibm/ILOG/CPLEX_Studio2211
export EVALMAXSAT_BIN=/opt/evalmaxsat/EvalMaxSAT_bin
export HCORAP_EXPECTED_COMMIT=$(git rev-parse HEAD)
export HCORAP_BACKUP_DIR=/mnt/disks/hcorap-backup

experiments/run_research_depth_gcp.sh preflight
```

Sau khi đọc lại ma trận 960 lần chạy được in bởi preflight:

```bash
export CONFIRM_RESEARCH_DEPTH_PILOT=YES
nohup experiments/run_research_depth_gcp.sh pilot \
  > research-depth-pilot.log 2>&1 &

experiments/run_research_depth_gcp.sh status
tail -f research-depth-pilot.log
```

Nếu phiên SSH ngắt hoặc VM khởi động lại, chạy lại đúng lệnh `pilot`; runner chỉ
chạy các run ID còn thiếu hoặc không hợp lệ. Có thể tách tài nguyên thành
`pilot-commercial` và `pilot-maxsat`. Sau khi xem kết quả pilot và quyết định
giữ nguyên thiết kế 2.208 lần chạy full:

```bash
export CONFIRM_RESEARCH_DEPTH_FULL=YES
nohup experiments/run_research_depth_gcp.sh full \
  > research-depth-full.log 2>&1 &
```

`all` chạy preflight, pilot rồi full và đòi cả hai biến xác nhận. Các pha
`analyze-pilot` và `analyze-full` không gọi solver. Script từ chối measured run
nếu worktree bẩn hoặc HEAD khác `HCORAP_EXPECTED_COMMIT`; vì vậy cần commit/tag
phiên bản thực nghiệm trước khi đưa lên GCP. Nếu cấu hình VM không có đúng 8
vCPU, đặt `HCORAP_EXPECTED_VCPUS` bằng số vCPU thực tế và ghi giá trị đó khi báo
cáo môi trường.

Cấu hình đã được sinh sẵn trong `experiments/configs/research_depth_*.json`. Pilot chọn 16 instance trải đủ các lớp U/A/V và chứa cả ba trường hợp xung đột COS/OCS đã biết. Đây là pilot thiết kế có chủ đích, không phải mẫu ngẫu nhiên để suy rộng. Full dùng 48 instance gốc.

| Campaign | Pilot | Full | Câu hỏi |
|---|---:|---:|---|
| diagnostics | 160 | 480 | 3 policy tham chiếu + 4 cực trị Weighted + k = 0,1,2 |
| weights | 144 | 432 | wc, wo ∈ {1,4,8}, hệ số SIM = 1 |
| load | 432 | 1.296 | 9 mức năng lực × 3 policy |
| diagnostics_cplex_audit | 160 | — | Kiểm tra độc lập cùng pilot bằng CPLEX MIP |
| zero_continuity | 64 | — | 16 Original × SN/TOT × global/local |

Các con số là số lần gọi driver, mỗi lần có thể gồm nhiều bước solver. Full chứa lại các ô pilot và dùng thư mục kết quả riêng; không gộp hai bộ như các quan sát độc lập. Các policy tham chiếu được chạy lại trong campaign mới để giữ cùng binary/môi trường. Dữ liệu cũ vẫn dùng trực tiếp cho phân tích cấu trúc và đối chiếu lịch sử.

Trên Linux/GCP đã cài SDK và giấy phép tương ứng:

```bash
make -B -j4 YICES=0 GUROBI=1 CPLEX=1 hcorap_commercial
bin/release/hcorap_commercial --list-backends
python3 experiments/run_commercial_campaign.py experiments/configs/research_depth_diagnostics_pilot.json --preflight-only
python3 experiments/run_commercial_campaign.py experiments/configs/research_depth_diagnostics_pilot.json --resume
python3 experiments/analyze_policy_diagnostics.py experiments/results/research_depth_diagnostics_pilot experiments/results/research_depth_diagnostics_pilot/analysis
```

`-B` cần thiết khi biên dịch lại vì cấu trúc dữ liệu trong header đã đổi và Makefile hiện tại không theo dõi đầy đủ phụ thuộc header. Máy macOS hiện tại chỉ có reference-enumerator trong binary commercial; chưa chạy Gurobi/CPLEX mới tại đây.

Đối chiếu CPLEX:

```bash
python3 experiments/run_commercial_campaign.py experiments/configs/research_depth_diagnostics_cplex_audit.json --resume
python3 experiments/compare_policy_diagnostics.py experiments/results/research_depth_diagnostics_pilot experiments/results/research_depth_diagnostics_cplex_audit experiments/results/research_depth_diagnostic_audit
```

Độ nhạy trọng số:

```bash
python3 experiments/run_commercial_campaign.py experiments/configs/research_depth_weights_pilot.json --resume
python3 experiments/collect_commercial_campaign.py experiments/results/research_depth_weights_pilot
python3 experiments/analyze_weight_sensitivity.py --results experiments/results/research_depth_weights_pilot --output-dir experiments/results/research_depth_weights_pilot/analysis
```

Độ nhạy năng lực:

```bash
python3 experiments/run_commercial_campaign.py experiments/configs/research_depth_load_pilot.json --resume
python3 experiments/analyze_load_sweep.py experiments/results/research_depth_load_pilot instances/research_depth_load_sweep/load_sweep_manifest.json experiments/results/research_depth_load_pilot/analysis
```

Thay `pilot` bằng `full` sau khi kiểm tra pilot. Không tự động mở rộng vì các kết quả sẽ quyết định phép khảo sát nào đáng đưa vào nghiên cứu chính. Grid năng lực này giữ khả thi; không kiểm tra chuyển tiếp sang bất khả thi. Dùng rho thực tế trong diễn giải. Các instance lồng nhau chia sẻ họ U/seed; bảng hiện tại là thống kê mô tả, chưa phải kiểm định với 432 mẫu độc lập.

Phần MaxSAT, đặt `EVALMAXSAT_BIN` tới binary Linux đã kiểm chứng:

```bash
make -B -j4 YICES=0 hcorap_multi
python3 experiments/run_reproducible_campaign.py experiments/configs/research_depth_zero_continuity_pilot.json --resume
python3 experiments/collect_reproducible_campaign.py experiments/results/research_depth_zero_continuity_pilot
python3 experiments/analyze_zero_continuity.py experiments/results/research_depth_zero_continuity_pilot experiments/results/research_depth_zero_continuity_pilot/analysis
```

Chỉ mở rộng phép thử mã hóa nếu vector tối ưu khớp, không có lỗi solver, và phần giảm công thức mang lại lợi ích thời gian trên các cặp chạy hoàn tất. Thay đổi shared commercial source sẽ khiến cơ chế xác minh tái sử dụng cũ từ chối binary khác nguồn; giữ nguyên cơ chế đó. Không chuyển các kết quả mới vào thư mục campaign cũ hoặc dùng reference lịch sử như chứng nhận cho code đã đổi.

**Tái tạo phần phân tích không cần chạy solver**

```bash
python3 experiments/analyze_policy_structure.py results_v2/gcp_commercial_corrected_primary experiments/results/research_depth_existing/corrected
python3 experiments/analyze_stage_encoding_effect.py results_cardinality_aligned_3600/gcp_original_policy_encoding_cardinality_aligned_3600 experiments/results/research_depth_existing/original
```

**Kiểm thử**

Lượt kiểm thử kết hợp gần nhất tại máy phát triển: **216 passed**. Thời gian
phụ thuộc môi trường; GCP preflight sẽ chạy lại cùng nhóm kiểm thử trước khi đo.

```bash
make -B -j4 YICES=0 hcorap_multi hcorap_commercial
python3 -m pytest tests/test_research_depth.py tests/test_cpp_multiobjective.py tests/test_commercial_backends.py tests/test_commercial_campaign.py tests/test_reproducible_campaign.py tests/test_weight_analysis.py -q
```

Các phép thử mới đối chiếu các cực trị với vét cạn độc lập, gồm trường hợp đồng tối ưu, trọng số bằng 0 và W* âm; kiểm tra k = 0 và đường nới CONT; kiểm tra full/partial coverage và trường hợp C* > 0 cho mã hóa local; kiểm tra hash, anchor, định danh campaign, lỗi chứng nhận và việc không tính speedup khi solver lỗi. Không suy rộng các kết quả chức năng này thành claim về hiệu năng hay tính mới.

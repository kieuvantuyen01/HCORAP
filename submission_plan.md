# Kế hoạch hoàn thiện và nộp bài ICIIT 2027

Cập nhật ngày 09/08/2026. Phạm vi thực nghiệm trong tài liệu này được khóa theo
`experiments/configs/reduced_campaign_manifest.json` tại commit
`6394e6e952cd07f1f7b0ac7b8c0bf46f4743e3dc`.

## 1. Thông điệp khoa học và hàm mục tiêu

Kết quả hiện có cho thấy weighted MaxSAT có thể trả về các vector chất lượng
khác nhau dù cùng scalar optimum. Trên benchmark gốc, overtime lại xuất hiện rất
hiếm, nên chỉ thay trọng số hoặc đặt overtime ở tầng đầu không tạo đủ tín hiệu.
Các kết quả commercial cũ cũng cho thấy policy `CONT -> SIM -> OT` có thể chấp
nhận thêm overtime chỉ để đổi lấy một lượng similarity nhỏ.

Policy chính của bài được khóa là:

\[
\boxed{\min CONT\;\rightarrow\;\min OT\;\rightarrow\;\max SIM}
\]

Tên sử dụng trong code và bản thảo: **LEX-COS**
(Continuity–Overtime–Similarity). Full coverage vẫn là hard constraint nên
không nằm trong vector lexicographic.

Policy sensitivity là:

\[
\min OT\;\rightarrow\;\min CONT\;\rightarrow\;\max SIM,
\]

được gọi là **LEX-OCS**. Policy cũ `CONT -> SIM -> OT` không còn là policy chính
và không nằm trong campaign mặc định.

Thông điệp bài báo nên giữ ở mức sau:

> Một policy lexicographic tường minh loại bỏ sự mơ hồ của weighted optimum;
> Totalizer làm thay đổi kích thước và hiệu năng encoding; tác động của implied
> constraints và exact symmetry breaking cần được đánh giá bằng paired
> factorial ablation thay vì giả định luôn có lợi.

Không viết claim “mọi cải tiến đều làm solver nhanh hơn”. Kết quả âm hoặc phụ
thuộc lớp instance vẫn là kết quả hợp lệ.

## 2. Câu hỏi nghiên cứu

### RQ1 — Weighted sum và lexicographic optimization

So với weighted B0, LEX-COS thay đổi continuity, overtime, similarity,
time-to-proof và solved count như thế nào? Đổi thứ tự hai ưu tiên đầu sang
LEX-OCS có tạo khác biệt thực chất không?

### RQ2 — Totalizer

Totalizer ảnh hưởng thế nào đến số biến, số hard/soft clauses, peak RSS, solved
count, PAR-2 và time-to-proof so với sorting network?

### RQ3 — Constraint strengthening

Implied constraints và exact symmetry breaking có cải thiện hiệu năng ổn định
hay tương tác với cardinality encoding và cấu trúc instance?

### Validation — corrected-v2 và commercial baseline

Các kết luận chính có còn quan sát được trên corrected-v2 critical benchmark
không? MaxSAT có khớp objective với exact MIP của Gurobi và CPLEX trên tập
commercial baseline đã khai báo trước không?

## 3. Phạm vi thực nghiệm đã khóa

### 3.1 Phần bắt buộc để đưa vào bản thảo

- paired 8-cell factorial ablation trên 160 original instances;
- weighted baseline/proposed trên đủ 800 original instances;
- LEX-COS confirmatory trên 280 held-out instances thuộc 14 lớp;
- LEX-OCS sensitivity trên 80 instances;
- weighted/LEX-COS trên 160 corrected-v2 evaluation-critical instances;
- Gurobi MIP và CPLEX MIP trên 100 original instances;
- small epsilon và weight screens chỉ dùng làm exploratory evidence.

Baseline và proposed được khóa:

```text
baseline = sorting-network / implied none / symmetry none
proposed = totalizer / implied both / symmetry slot-service
```

Hai lớp `30_15_4` và `40_25_5` đã được xem trong commercial development nên bị
loại khỏi LEX-COS confirmatory set. Corrected-v2 calibration dùng seed 1–10;
evaluation dùng seed 1001–1010; hai tập không giao nhau.

### 3.2 Phần hoãn hoặc loại khỏi bài hiện tại

| Nhánh | Quyết định | Cách trình bày |
|---|---|---|
| full Pareto/epsilon confirmation | hoãn | chỉ báo cáo screen ba delta như exploratory |
| full weight confirmation | hoãn | chỉ báo cáo screen bốn weight pairs như exploratory |
| corrected relaxed/saturated load stress | hoãn | limitation hoặc supplementary future run |
| availability uncertainty | hoãn | không đưa claim robust optimization |
| commercial epsilon/corrected/CP | loại khỏi campaign chính | chỉ giữ code và correctness tests |
| routing | không chạy | limitation/future work; model chưa có routing semantics |

Không mở rộng lại các nhánh này trước khi hoàn tất toàn bộ ma trận bắt buộc và
khóa draft v1.

## 4. Ma trận thực nghiệm rút gọn

| Campaign | Thiết kế | Runs | Timeout | Worst-case core-hour |
|---|---|---:|---:|---:|
| original factorial ablation | 160 instances × 8 configs × weighted | 1.280 | 120 s | 42,67 |
| corrected multiobjective screen | 32 × LEX-COS/epsilon `0,.05,.10` | 128 | 60 s | 2,13 |
| corrected weight screen | 32 × `(1,1),(1,4),(4,1),(8,8)` | 128 | 60 s | 2,13 |
| original lex scalability | 80 × 2 configs × weighted/LEX-COS | 320 | 300 s | 26,67 |
| original weighted primary | 800 × baseline/proposed | 1.600 | 300 s | 133,33 |
| original LEX-COS primary | 280 held-out × baseline/proposed | 560 | 300 s | 46,67 |
| LEX-OCS sensitivity | 80 × baseline/proposed | 160 | 300 s | 13,33 |
| corrected-v2 primary | 160 × weighted/LEX-COS, proposed config | 320 | 300 s | 26,67 |
| commercial original | 100 × Gurobi/CPLEX × weighted/LEX-COS | 400 | 300 s | 33,33 |
| **Tổng measured** |  | **4.896** |  | **326,93** |

Ngoài ma trận measured có 36 commercial correctness-smoke runs, timeout 30
giây, không đưa vào bảng runtime. So với thiết kế cũ 16.040 runs và khoảng 1.171
core-hour, thiết kế mới giảm khoảng 69,5%. Worst case là 13,62 ngày tuần tự;
thời gian thực tế thường thấp hơn vì nhiều run kết thúc trước timeout.

## 5. Protocol GCP và cách chạy

Máy publication: GCP `c4-highcpu-8`, 8 vCPU, 16 GB RAM, non-Spot, Ubuntu 24.04
LTS. Mỗi measured run dùng một solver process, một solver thread và một vCPU
được pin. Không chạy workload khác trong campaign.

Các yếu tố phải cố định:

- Open-WBO commit `80f3073e41028b219b0b0ad7c61fba28351f88e6`;
- compiler flags `-O3 -DNDEBUG -std=c++11`;
- `WORKERS=1`, `HCORAP_CPU_CORE=0`;
- cùng VM family, OS image, binary và solver hash;
- cumulative timeout cho toàn bộ stages của một policy;
- 10 warm-up instances không thuộc tập đo;
- blocked-instance randomized order với seed được lưu;
- không update OS/compiler/solver hoặc sửa config giữa campaign;
- Gurobi/CPLEX: một thread, seed 0, relative/absolute gap bằng 0.

Runner đã triển khai instance-major blocked execution, resume, hard timeout,
peak RSS, raw JSON, stderr/native logs, instance/binary/solver hashes, exact run
ID, completeness validation và environment snapshot. Đây không còn là hạng mục
“cần sửa” mà là điều kiện phải được preflight xác nhận.

Sau khi cấu hình Open-WBO, Gurobi và CPLEX trên VM:

```bash
export OPEN_WBO_SOURCE_DIR=/opt/hcorap-open-wbo
export OPEN_WBO_BIN=/opt/hcorap-open-wbo/open-wbo
export OPEN_WBO_COMMIT=80f3073e41028b219b0b0ad7c61fba28351f88e6
export GUROBI_HOME=/absolute/path/to/gurobi/platform
export CPLEX_STUDIO_DIR=/absolute/path/to/CPLEX_Studio
export HCORAP_CPU_CORE=0

bash experiments/gcp_prepare_and_run.sh preflight
bash experiments/gcp_prepare_and_run.sh commercial-preflight

export CONFIRM_REDUCED_CAMPAIGN=YES
bash experiments/run_iciit2027_reduced_campaign.sh
```

Script một lệnh tự chạy:

```text
build/test -> benchmark verification -> budget/task-count validation -> warm-up
-> screening GO/NO-GO -> original primary -> corrected primary
-> commercial preflight/primary -> reproducibility package
```

Chi tiết vận hành và resume nằm trong
[`docs/GCP_EXPERIMENT_RUNBOOK.md`](docs/GCP_EXPERIMENT_RUNBOOK.md).

## 6. Các gate bắt buộc

### G1 — Correctness và reproducibility preflight

- toàn bộ tests pass;
- C++ binaries build thành công;
- Open-WBO source/binary đúng pinned commit;
- 320/320 corrected-v2 instances qua witness/hash/matrix verification;
- manifest ngân sách đúng 4.896 measured runs;
- mọi MaxSAT config resolve đúng expected instance/run count;
- commercial license preflight và 36/36 smoke runs đạt verified optimum;
- Gurobi, CPLEX và reference enumerator không mismatch.

### G2 — Screening GO/NO-GO

- không có technical hoặc validation error;
- zero weighted-objective mismatch giữa baseline và proposed khi cùng optimum;
- proposed optimum count ít nhất 95% baseline trong factorial screen;
- LEX-COS và epsilon optimum rate tối thiểu 10% trong short screen;
- weight screen tạo ít nhất một instance với hai objective vectors khác nhau;
- trên lex scalability screen, LEX-COS hoàn tất ít nhất 60% B0-optimal
  instances ở ít nhất một config;
- peak RSS không quá 12 GB.

Nếu G2 là `NO-GO`, dừng publication campaign. Không lách gate bằng cách sửa biến
môi trường. Đọc raw stage logs, tối ưu implementation hoặc thu hẹp claim.

### G3 — Data freeze

- đúng 4.896 measured rows, không duplicate và không thiếu file;
- không `PARSE_ERROR`, runner error hoặc invalid result;
- mọi `OPTIMUM` được verifier chấp nhận;
- collector tái tạo toàn bộ bảng từ raw JSON;
- weighted scalar objective khớp trên các paired exact optima;
- LEX objective vector khớp giữa Gurobi và CPLEX khi cả hai exact optimum;
- không thay code, solver, timeout hoặc benchmark sau thời điểm freeze;
- artifact `.tar.gz` và `.sha256` được tạo từ clean commit.

## 7. Ánh xạ kết quả vào bản thảo

| Claim/bảng | Nguồn được phép dùng |
|---|---|
| Totalizer/implied/symmetry ablation | `gcp_primary_analysis/factorial_*` |
| baseline/proposed trên 800 instances | `gcp_original_weighted_primary` |
| LEX-COS quality và runtime | `gcp_primary_analysis/lex_confirmatory_*` |
| LEX-OCS sensitivity | `lex_policy_sensitivity_pairs.csv` |
| corrected-v2 validation | `gcp_corrected_primary` |
| Gurobi/CPLEX agreement | `gcp_commercial_original` |
| epsilon/weight exploratory | hai `gcp_*_screen_analysis` directories |
| threats/reproducibility | environment, manifests, hashes và validation files |

Không dùng runtime lịch sử thiếu provenance trong bảng chính. Không gộp screen,
confirmatory và development data mà không ghi nhãn. Không loại timeout sau khi
xem kết quả chất lượng.

## 8. Lịch thực hiện cập nhật

| Ngày | Công việc và đầu ra |
|---|---|
| 09/08 | Khóa code/config ở commit `6394e6e`; chốt reduced matrix và runbook |
| 10–11/08 | Tạo fresh GCP VM; cài solver/SDK/license; chạy hai preflight |
| 12–15/08 | Chạy screen 1.856 rows; đọc và lưu quyết định GO/NO-GO |
| 16/08 | Gate review; chỉ tiếp tục nếu GO; lập issue cho mọi anomaly |
| 17–23/08 | Chạy 1.600 original weighted primary rows |
| 24–26/08 | Chạy 560 LEX-COS confirmatory + 160 LEX-OCS sensitivity rows |
| 27–29/08 | Chạy 320 corrected-v2 + 400 commercial rows |
| 30–31/08 | Resume lỗi kỹ thuật hợp lệ, chạy collectors, G3 data freeze |
| 01–05/09 | Phân tích paired statistics, PAR-2, cactus/size/trade-off plots |
| 06–10/09 | Viết Methods, Experimental Setup và Results từ generated tables |
| 11–13/09 | Viết Introduction, Related Work, Threats và Conclusion; draft v1 |
| 14–18/09 | Technical review nội bộ; kiểm tra claim–evidence và sửa bản thảo |
| 19–22/09 | Clean-room reproduction; kiểm artifact và checksum |
| 23–25/09 | English editing, rút đúng page limit, kiểm LaTeX/BibTeX |
| 26–27/09 | Upload thử, kiểm metadata, author order và PDF compliance |
| 28/09 | Mốc submit nội bộ |
| 29–30/09 | Chỉ dùng làm buffer sự cố; không bổ sung experiment mới |

Nguồn chính thức hiện ghi full-paper deadline **30/09/2026**, notification
**20/10/2026**, registration **10/11/2026** và hội nghị diễn ra
**04–07/03/2027** tại TP. Hồ Chí Minh. Cần kiểm tra lại trang Important Dates
trước ngày upload vì lịch hội nghị có thể được cập nhật:
[ICIIT 2027 Important Dates](https://www.iciit.org/date.html).

## 9. Kế hoạch trình bày cho conference proceedings

Trang submission chính thức yêu cầu tối thiểu 4 trang double-column; một regular
registration bao gồm tối đa 5 trang double-column. Mục tiêu an toàn là đúng 5
trang, gồm cả hình, bảng và tài liệu tham khảo:
[ICIIT 2027 Paper Submission](https://www.iciit.org/sub.html).

Phân bổ dự kiến:

| Phần | Ngân sách |
|---|---:|
| Abstract + Introduction + contributions | 0,65 trang |
| Related Work | 0,45 trang |
| Model và lexicographic objective | 0,75 trang |
| Totalizer, implied constraints, symmetry breaking | 0,85 trang |
| Experimental protocol | 0,55 trang |
| Results và discussion | 1,25 trang |
| Threats, conclusion và references | 0,50 trang |
| **Tổng** | **5,00 trang** |

Ưu tiên tối đa ba bảng/hình chính:

1. factorial/encoding size và performance summary;
2. weighted–LEX-COS paired quality/runtime summary;
3. corrected-v2/commercial validation summary hoặc một compact cactus plot.

Pareto/weight screen chỉ đưa vào một đoạn hoặc supplementary artifact nếu không
còn chỗ. Không hy sinh threats to validity để nhét thêm exploratory plots.

## 10. Definition of done trước khi submit

- [ ] Commit/tag publication code và lưu commit ID trong manuscript.
- [ ] G1 và G2 đều pass; `screening_decision.json` là `GO`.
- [ ] G3 xác nhận đủ 4.896 measured rows.
- [ ] Tất cả bảng/hình được sinh bằng script từ frozen raw data.
- [ ] Mỗi con số trong Abstract/Conclusion truy ngược được tới generated table.
- [ ] Tách rõ calibration, evaluation, development và confirmatory sets.
- [ ] Routing/robust/Pareto claims không vượt quá implementation thực tế.
- [ ] Artifact có source, configs, instances, raw logs, environment và SHA-256.
- [ ] Clean-room reproduction chạy được trên fresh checkout.
- [ ] PDF đúng template/page limit, font embedded và không có warning nghiêm trọng.
- [ ] Author names, affiliations, email, title và submission metadata khớp nhau.
- [ ] Internal submit hoàn tất trước 28/09/2026.

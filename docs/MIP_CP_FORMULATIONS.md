# Brainstorming mô hình MIP và CP cho HCORAP

Tài liệu này là bản nháp có thể chuyển thành một mục độc lập trong kế hoạch
nghiên cứu hoặc bài báo. Phạm vi giữ nguyên HCORAP hiện tại: mỗi service chiếm
một time slot; coverage là hard constraint trong benchmark chính; chưa xét
travel time, routing, uncertainty hoặc dynamic rescheduling.

## 1. Khuyến nghị tổng thể

Nên tách rõ ba backend và ít nhất hai formulation:

| Mã | Backend | Formulation |
|---|---|---|
| MIP-E/Gurobi | Gurobi MIP | MIP time-indexed, chỉ tạo biến trên candidate khả thi |
| MIP-E/CPLEX | CPLEX MIP | đúng cùng hệ phương trình MIP-E |
| CP-T | IBM ILOG CP Optimizer | biến nguyên caregiver/slot và global constraints |
| CP-I | IBM ILOG CP Optimizer | optional interval, `alternative`, `noOverlap` |

MIP-E/Gurobi và MIP-E/CPLEX phải dùng cùng dữ liệu tiền xử lý, cùng biến, cùng
ràng buộc và cùng objective. So sánh này chủ yếu đo khác biệt giữa hai MIP
solver. CP-T nên là CP baseline chính vì HCORAP hiện có duration bằng một slot.
CP-I là formulation thứ hai có giá trị khoa học và là nền tảng tự nhiên cho
duration, calendar, travel time hoặc transition time trong nghiên cứu tiếp theo.

Tên sản phẩm nên viết chính xác là **Gurobi Optimizer**, **CPLEX Optimizer**
cho MIP, và **IBM ILOG CP Optimizer** cho CP. Có thể dùng “CPLEX CP” trong ghi
chú nội bộ, nhưng trong bài báo nên dùng “CP Optimizer” để tránh nhầm đây là
một chế độ của CPLEX MIP.

## 2. Ký hiệu và tiền xử lý dùng chung

Các tập:

- \(\mathcal A\): caregivers/agents;
- \(\mathcal S\): services;
- \(\mathcal T\): time slots;
- \(\mathcal U\): users;
- \(\mathcal Q\): các continuity sequences;
- \(\mathcal S_u\subseteq\mathcal S\): services của user \(u\);
- \(\mathcal S_q\subseteq\mathcal S\): services thuộc sequence \(q\).

Dữ liệu:

- \(R_{as}\geq 0\): similarity reward; \(R_{as}=0\) nghĩa là không đủ điều kiện;
- \(A_{at}\in\{0,1\}\): agent \(a\) khả dụng tại slot \(t\);
- \(B_{st}\in\{0,1\}\): service \(s\) được phép thực hiện tại slot \(t\);
- \(H_a^N\): số slot làm việc bình thường;
- \(H_a^E\): số slot overtime tối đa;
- \(p=|P|\): chi phí dương trên mỗi overtime slot.

Định nghĩa tập candidate khả thi:

\[
\mathcal E=
\{(a,s,t)\mid R_{as}>0,\ A_{at}=1,\ B_{st}=1\}.
\]

Với mỗi service, đặt

\[
\mathcal E_s=\{(a,t)\mid(a,s,t)\in\mathcal E\}.
\]

Nếu \(\mathcal E_s=\varnothing\) trong mô hình full coverage thì instance
infeasible ngay ở bước tiền xử lý. Chỉ tạo biến trên \(\mathcal E\) giúp giảm
số biến từ \(A\,S\,T\) xuống \(|\mathcal E|\), đồng thời loại bỏ hoàn toàn các
ràng buộc kiểu \(x_{ast}=0\).

Lưu ý về đơn vị: \(H_a^N,H_a^E\) đang đếm số service/time slot, không nhất
thiết là số giờ thực nếu độ dài một slot khác một giờ. Tài liệu nên dùng nhất
quán “workload slots” hoặc nêu rõ một slot tương ứng một giờ.

## 3. Mô hình MIP-E

### 3.1. Biến quyết định

\[
\begin{aligned}
x_{ast}&\in\{0,1\}
&&\forall(a,s,t)\in\mathcal E,\\
y_{as}&\in\{0,1\}
&&\forall(a,s):\exists t,\ (a,s,t)\in\mathcal E,\\
v_{aq}&\in\{0,1\}
&&\forall(a,q):\exists s\in\mathcal S_q,\ y_{as}\text{ tồn tại},\\
W_a&\in\mathbb Z_{\geq0}
&&\forall a\in\mathcal A,\\
e_a&\in\mathbb Z_{\geq0}
&&\forall a\in\mathcal A.
\end{aligned}
\]

Trong đó:

- \(x_{ast}=1\) iff agent \(a\) thực hiện service \(s\) ở slot \(t\);
- \(y_{as}=1\) iff agent \(a\) thực hiện service \(s\);
- \(v_{aq}=1\) iff agent \(a\) thực hiện ít nhất một service trong sequence \(q\);
- \(W_a\) là workload của agent \(a\);
- \(e_a\) là số overtime slot của agent \(a\).

Có thể loại \(y,W,e\) bằng phép thế, nhưng giữ các biến có tên giúp mô hình dễ
đọc, dễ trích xuất metric và dễ đối chiếu với MaxSAT/verifier.

### 3.2. Coverage

Vì coverage là hard constraint trong benchmark chính:

\[
\sum_{(a,t)\in\mathcal E_s}x_{ast}=1
\qquad \forall s\in\mathcal S.
\tag{M1}
\]

Đẳng thức (M1) thay cho cặp “at most one assignment” và “service phải được
phục vụ”. Cách viết này ngắn hơn và làm rõ mỗi service được gán đúng một lần.

### 3.3. Không xung đột caregiver

\[
\sum_{s:(a,s,t)\in\mathcal E}x_{ast}\leq1
\qquad \forall a\in\mathcal A,\ t\in\mathcal T.
\tag{M2}
\]

### 3.4. Không phục vụ đồng thời cùng một user

\[
\sum_{s\in\mathcal S_u}
\sum_{a:(a,s,t)\in\mathcal E}x_{ast}\leq1
\qquad \forall u\in\mathcal U,\ t\in\mathcal T.
\tag{M3}
\]

### 3.5. Liên kết assignment và workload

\[
y_{as}=\sum_{t:(a,s,t)\in\mathcal E}x_{ast}
\qquad \forall(a,s)\text{ có candidate},
\tag{M4}
\]

\[
W_a=\sum_{s:(a,s)\text{ có candidate}}y_{as}
\qquad \forall a\in\mathcal A,
\tag{M5}
\]

\[
W_a\leq H_a^N+H_a^E
\qquad \forall a\in\mathcal A.
\tag{M6}
\]

### 3.6. Continuity of care

Biến \(v_{aq}\) là phép OR chính xác của các \(y_{as}\) trong sequence:

\[
v_{aq}\geq y_{as}
\qquad
\forall a,q,\ s\in\mathcal S_q,
\tag{M7}
\]

\[
v_{aq}\leq
\sum_{s\in\mathcal S_q:(a,s)\text{ có candidate}}y_{as}
\qquad \forall a,q.
\tag{M8}
\]

Đặt

\[
D_q=\sum_{a\in\mathcal A}v_{aq}.
\tag{M9}
\]

Với full coverage, mọi sequence không rỗng đều có \(D_q\geq1\). Continuity
penalty vì thế là

\[
\operatorname{CONT}
=\sum_{q:|\mathcal S_q|>1}(D_q-1).
\tag{M10}
\]

Không cần toán tử \((D_q-1)^+\) trong mô hình full coverage. Các sequence đơn
có penalty bằng 0 và có thể bỏ khỏi objective.

### 3.7. Overtime

Formulation gọn là

\[
e_a\geq W_a-H_a^N,\quad
0\leq e_a\leq H_a^E.
\tag{M11}
\]

(M11) cho \(e_a=\max(0,W_a-H_a^N)\) tại optimum nếu overtime có hệ số phạt
dương hoặc được tối thiểu hóa ở một tầng lexicographic. Tuy nhiên, \(e_a\)
không có ngữ nghĩa chính xác ở mọi nghiệm trung gian nếu objective hiện tại
không chứa overtime.

Nếu cần \(e_a\) chính xác trong mọi nghiệm, dùng các biến threshold
\(o_{ak}\in\{0,1\}\), \(k=1,\ldots,H_a^E\), với ý nghĩa
\(o_{ak}=1\Longleftrightarrow W_a\geq H_a^N+k\):

\[
W_a\geq(H_a^N+k)o_{ak},
\tag{M12}
\]

\[
W_a\leq H_a^N+k-1+(H_a^E-k+1)o_{ak},
\tag{M13}
\]

\[
e_a=\sum_{k=1}^{H_a^E}o_{ak}.
\tag{M14}
\]

Có thể thêm \(o_{a,k}\geq o_{a,k+1}\) như một implied constraint. Formulation
threshold gần với semantics của các sorting-network/Totalizer output trong
MaxSAT và thuận tiện cho kiểm thử chéo.

Một lựa chọn khác ở tầng API là dùng general constraint
\(e_a=\max(0,W_a-H_a^N)\). Nếu mục tiêu là so sánh Gurobi MIP và CPLEX MIP
trên đúng cùng algebraic model, nên dùng (M12)--(M14) hoặc (M11) ở cả hai
backend thay vì dùng hai general-constraint implementation khác nhau.

### 3.8. Các hàm mục tiêu

\[
\operatorname{SIM}
=\sum_{(a,s)}R_{as}y_{as},
\tag{M15}
\]

\[
\operatorname{OT}=\sum_{a\in\mathcal A}e_a.
\tag{M16}
\]

Weighted baseline:

\[
\max\quad
\operatorname{SIM}
-w_c\operatorname{CONT}
-w_o\,p\,\operatorname{OT}.
\tag{M17}
\]

Hai chính sách lexicographic:

\[
\min\operatorname{CONT}
\succ
\max\operatorname{SIM}
\succ
\min\operatorname{OT},
\tag{M18}
\]

\[
\min\operatorname{OT}
\succ
\min\operatorname{CONT}
\succ
\max\operatorname{SIM}.
\tag{M19}
\]

Similarity-budget \(\varepsilon\)-constraint:

1. tính \(\operatorname{SIM}^*=\max\operatorname{SIM}\);
2. thêm
   \[
   \operatorname{SIM}\geq
   L_\delta=\left\lceil(1-\delta)\operatorname{SIM}^*\right\rceil;
   \]
3. tối ưu theo thứ tự
   \[
   \min\operatorname{CONT}
   \succ\min\operatorname{OT}
   \succ\max\operatorname{SIM}.
   \]

Để so sánh backend minh bạch, nên triển khai B1/B2 bằng các lần solve tuần tự
và thêm bound tối ưu sau mỗi tầng. Native multiobjective của Gurobi/CPLEX và
static lexicographic objective của CP Optimizer có thể là một ablation riêng.

## 4. Mở rộng MIP khi coverage mềm

Tạo \(z_s\in\{0,1\}\):

\[
z_s=\sum_{(a,t)\in\mathcal E_s}x_{ast}
\qquad\forall s\in\mathcal S.
\tag{MS1}
\]

Vì \(z_s\) là biến nhị phân và mọi \(x_{ast}\) đều không âm, (MS1) đồng thời
định nghĩa trạng thái coverage của service \(s\) và bảo đảm service đó có nhiều
nhất một assignment. Do đó không cần viết thêm một bất đẳng thức
\(\sum_{(a,t)\in\mathcal E_s}x_{ast}\leq1\) riêng.

Coverage:

\[
\operatorname{COV}=\sum_sz_s.
\tag{MS2}
\]

Tạo \(h_q\in\{0,1\}\), là OR của các \(z_s\) trong sequence:

\[
h_q\geq z_s\quad\forall s\in\mathcal S_q,\qquad
h_q\leq\sum_{s\in\mathcal S_q}z_s.
\tag{MS3}
\]

Continuity penalty đúng khi sequence có thể không được phục vụ là

\[
\operatorname{CONT}^{\mathrm{serv}}
=\sum_q(D_q-h_q).
\tag{MS4}
\]

Nếu sequence không có service nào được phục vụ thì \(D_q=h_q=0\); nếu active
thì \(h_q=1\) và penalty là \(D_q-1\). Không được dùng trực tiếp \(D_q-1\)
trong soft coverage vì nó cho giá trị âm trên sequence không được phục vụ.

Coverage phải là tầng ưu tiên cao nhất:

\[
\max\operatorname{COV}\succ\text{các objective còn lại}.
\]

## 5. Mô hình CP-T: biến nguyên và global constraints

CP-T khai thác việc mỗi service hiện có duration đúng một slot.

### 5.1. Biến

\[
\alpha_s\in\mathcal A
\quad\text{(agent thực hiện service \(s\))},
\]

\[
\tau_s\in\mathcal T
\quad\text{(slot của service \(s\))}.
\]

Tập cặp hợp lệ cho service \(s\):

\[
\mathcal P_s=\{(a,t)\mid(a,s,t)\in\mathcal E\}.
\]

### 5.2. Candidate table

\[
(\alpha_s,\tau_s)\in\mathcal P_s
\qquad\forall s\in\mathcal S.
\tag{CPT1}
\]

Trong CP Optimizer, đây là một compatibility/table constraint
(`allowedAssignments`). Nó đồng thời biểu diễn qualification, agent
availability và service time window.

### 5.3. Không xung đột agent-slot

Mã hóa một cặp agent-slot bằng

\[
\kappa_s=T\alpha_s+\tau_s
\]

với chỉ số bắt đầu từ 0. Khi đó:

\[
\operatorname{allDifferent}(\kappa_s:s\in\mathcal S).
\tag{CPT2}
\]

(CPT2) cấm hai service dùng cùng một agent ở cùng một slot.

### 5.4. Không xung đột user-slot

\[
\operatorname{allDifferent}(\tau_s:s\in\mathcal S_u)
\qquad\forall u\in\mathcal U.
\tag{CPT3}
\]

### 5.5. Workload, continuity và overtime

\[
W_a=\operatorname{count}([\alpha_s]_{s\in\mathcal S},a),
\qquad
W_a\leq H_a^N+H_a^E.
\tag{CPT4}
\]

Đặt \(v_{aq}=1\) iff có \(s\in\mathcal S_q\) với \(\alpha_s=a\). Có thể dùng
reified count:

\[
v_{aq}\Longleftrightarrow
\operatorname{count}([\alpha_s]_{s\in\mathcal S_q},a)>0.
\tag{CPT5}
\]

Sau đó \(D_q,\operatorname{CONT}\) được tính như (M9)--(M10), và

\[
e_a=\max(0,W_a-H_a^N).
\tag{CPT6}
\]

Similarity được tính bằng element expression:

\[
\operatorname{SIM}
=\sum_{s\in\mathcal S}R_{\alpha_s,s}.
\tag{CPT7}
\]

### 5.6. Ưu và nhược điểm dự kiến

Ưu điểm:

- chỉ có hai biến quyết định chính trên mỗi service;
- table constraint giữ nguyên tập candidate rời rạc;
- `allDifferent` và `count` là global constraints có propagation chuyên biệt;
- không cần một biến Boolean cho mỗi candidate triple.

Nhược điểm:

- formulation full coverage rất gọn nhưng soft coverage cần một giá trị
  “unassigned” và các global constraint phải bỏ qua giá trị sentinel;
- khó mở rộng tự nhiên sang service duration khác nhau, travel time hoặc
  transition time;
- similarity phụ thuộc vào element/table expression thay vì tổng Boolean tuyến
  tính đơn giản.

### 5.7. CP-T khi coverage mềm

Mở rộng domain bằng \(\alpha_s=-1,\tau_s=-1\) cho trạng thái unassigned và
thêm \(z_s\in\{0,1\}\), reward \(r_s\), agent-slot key \(\kappa_s\), user-slot
key \(\lambda_s\) vào cùng allowed-assignment tuple. Candidate row có
\(z_s=1,r_s=R_{as},\kappa_s=T a+t,\lambda_s=t\). Unassigned row của service
\(s\) có \(z_s=r_s=0\) và **hai sentinel key riêng theo service**. Nhờ sentinel
riêng, hai service unassigned không vi phạm `allDifferent`.

Workload và \(v_{aq}\) chỉ đếm các giá trị agent \(a\geq0\). Định nghĩa
\(h_q\) là OR chính xác của \(z_s\) trong sequence và dùng
\(\sum_q(D_q-h_q)\), giống (MS3)--(MS4).

## 6. Mô hình CP-I: optional interval và alternative

CP-I dùng scheduling primitives của CP Optimizer.

### 6.1. Interval variables

- \(I_s\): interval của service \(s\), size \(=1\); mandatory trong full
  coverage và optional trong soft coverage;
- \(I_{as}\): optional interval, size \(=1\), với mỗi cặp \((a,s)\) có ít nhất
  một slot candidate.

Với mỗi service \(s\) và agent \(a\), đặt

\[
\mathcal A_s=
\left\{
a\in\mathcal A\mid
\exists t\in\mathcal T:(a,s,t)\in\mathcal E
\right\},
\tag{CPI-A}
\]

\[
\mathcal S_a=
\left\{
s\in\mathcal S\mid a\in\mathcal A_s
\right\}.
\tag{CPI-S}
\]

Optional interval \(I_{as}\) chỉ được tạo khi \(a\in\mathcal A_s\), tương
đương với \(s\in\mathcal S_a\). Nếu \(\mathcal A_s=\varnothing\) trong mô
hình full coverage thì service \(s\) không có candidate và instance infeasible;
trường hợp này nên được phát hiện ngay ở bước tiền xử lý.

Tạo step function \(F_{as}(t)\), bằng 1 iff
\((a,s,t)\in\mathcal E\), cho mọi \(s\in\mathcal S\) và
\(a\in\mathcal A_s\), rồi cấm start ở các slot có giá trị 0.

### 6.2. Chọn caregiver

\[
\operatorname{alternative}
\left(I_s,\{I_{as}:a\in\mathcal A_s\}\right)
\qquad\forall s\in\mathcal S.
\tag{CPI1}
\]

Khi \(I_s\) present, đúng một \(I_{as}\) được present và interval được chọn có
cùng start/end với \(I_s\). Trong full coverage, \(I_s\) mandatory nên luôn có
đúng một alternative. Trong soft coverage, absence của \(I_s\) kéo theo mọi
alternative absent.

### 6.3. Calendar và time window

\[
\operatorname{forbidStart}(I_{as},F_{as})
\qquad
\forall s\in\mathcal S,\ a\in\mathcal A_s.
\tag{CPI2}
\]

Với duration lớn hơn một trong tương lai, cần cân nhắc `forbidExtent` thay cho
chỉ `forbidStart` để toàn bộ interval nằm trong calendar khả dụng.

### 6.4. Không xung đột resource

\[
\operatorname{noOverlap}
\left(\{I_{as}:s\in\mathcal S_a\}\right)
\qquad\forall a\in\mathcal A,
\tag{CPI3}
\]

\[
\operatorname{noOverlap}
\left(\{I_s:s\in\mathcal S_u\}\right)
\qquad\forall u\in\mathcal U.
\tag{CPI4}
\]

### 6.5. Objective expressions

\[
y_{as}=\operatorname{presenceOf}(I_{as})
\qquad
\forall s\in\mathcal S,\ a\in\mathcal A_s,
\]

\[
W_a=
\sum_{s\in\mathcal S_a}\operatorname{presenceOf}(I_{as})
\qquad\forall a\in\mathcal A,
\]

\[
v_{aq}\Longleftrightarrow
\sum_{s\in\mathcal S_q\cap\mathcal S_a}
\operatorname{presenceOf}(I_{as})>0
\qquad
\forall a\in\mathcal A,\ q\in\mathcal Q.
\]

Nếu \(\mathcal S_q\cap\mathcal S_a=\varnothing\), có thể không tạo
\(v_{aq}\), hoặc cố định \(v_{aq}=0\). Similarity được viết trực tiếp theo
presence status:

\[
\operatorname{SIM}
=
\sum_{s\in\mathcal S}
\sum_{a\in\mathcal A_s}
R_{as}\operatorname{presenceOf}(I_{as}).
\tag{CPI5}
\]

Đặt

\[
D_q=\sum_{a\in\mathcal A}v_{aq},
\qquad
\operatorname{CONT}
=\sum_{q:|\mathcal S_q|>1}(D_q-1),
\tag{CPI6}
\]

\[
e_a=\max(0,W_a-H_a^N),
\qquad
\operatorname{OT}=\sum_{a\in\mathcal A}e_a.
\tag{CPI7}
\]

Các biểu thức (CPI5)--(CPI7) có cùng giá trị và semantics với (M15), (M10)
và (M16), nhưng mọi phép tổng chỉ tham chiếu các optional interval thực sự
tồn tại.

Trong soft coverage, đặt
\(h_q=\operatorname{OR}_{s\in\mathcal S_q}
\operatorname{presenceOf}(I_s)\) và thay (CPI6) bằng
\(\sum_q(D_q-h_q)\). Đây là cùng phép sửa đã dùng trong (MS4); nếu mọi master
interval trong sequence absent thì contribution bằng 0, không phải \(-1\).

### 6.6. Khi nào CP-I đáng kỳ vọng hơn CP-T?

CP-I đáng thử trên các instance mà xung đột calendar/time slot chi phối độ
khó. Nó còn là formulation nên ưu tiên nếu nghiên cứu mở rộng duration, travel
time, breaks, shift calendars hoặc transition times. Không nên mặc định CP-I
nhanh hơn CP-T trên HCORAP hiện tại; đây là một câu hỏi thực nghiệm.

## 7. Objective trong ba solver

Weighted B0 là một objective tuyến tính duy nhất và phải giống hệt trong cả ba
backend.

Với B1, có hai cách:

1. **Sequential portable**: solve từng tầng, ghi optimum, thêm equality/bound,
   rồi solve tầng sau. Đây nên là phương pháp chính vì semantics và cumulative
   timeout dễ kiểm toán.
2. **Native multiobjective/static lex**: dùng hierarchical objectives của
   Gurobi/CPLEX MIP hoặc static lexicographic objective của CP Optimizer. Chạy
   như ablation để xem reuse solver state/native search có lợi hay không.

Với CP Optimizer, các chiều min/max có thể đưa về cùng sense bằng đổi dấu. Ví dụ
continuity-priority dưới dạng minimization:

\[
\operatorname{lexmin}
\left[
\operatorname{CONT},
-\operatorname{SIM},
\operatorname{OT}
\right].
\]

B2 vẫn nên dùng solve tuần tự vì cần tính \(\operatorname{SIM}^*\), dựng
\(L_\delta\), rồi chạy nhiều mức \(\delta\).

## 8. Strengthening và formulation ablation

Chỉ thêm strengthening sau khi baseline đã được kiểm thử chéo. Các nhóm đáng
thử:

### 8.1. MIP strengthening

- candidate-sparse variables;
- clique constraints (M2), (M3);
- threshold overtime (M12)--(M14);
- slot-capacity upper bound theo maximum matching;
- projected service assignment và effective workload cap;
- exact symmetry breaking cho slot/service/agent equivalence classes;
- MIP starts từ cùng một heuristic schedule, nhưng chỉ trong thí nghiệm riêng.

Không bật tất cả strengthening ngay từ đầu vì sẽ không biết cải thiện đến từ
solver hay formulation.

### 8.2. CP strengthening/search

- CP-T so với CP-I;
- default search trước, custom search phase sau;
- branch theo service có candidate domain nhỏ nhất;
- ưu tiên các service thuộc sequence dài hoặc time window hẹp;
- redundant workload/count bounds;
- symmetry breaking chỉ trên equivalence class được chứng minh chính xác;
- starting point từ cùng heuristic schedule, báo cáo riêng.

Search strategy tùy chỉnh phải được chọn trên training/pilot subset và khóa
trước khi chạy test set.

## 9. Câu hỏi nghiên cứu có thể bổ sung

- **RQ6.** Gurobi MIP, CPLEX MIP và CP Optimizer khác nhau thế nào về số
  instance giải tối ưu, time-to-proof, PAR-2, peak memory và chất lượng incumbent
  dưới cùng timeout?
- **RQ7.** Load ratio, candidate density, time-window tightness, sequence length
  và symmetry dự báo backend thắng tốt đến mức nào?
- **RQ8.** CP-T hay CP-I phù hợp hơn với HCORAP duration-one-slot hiện tại?
- **RQ9.** Kết luận về weighted, lexicographic và
  \(\varepsilon\)-constraint có ổn định qua ba backend hay phụ thuộc mạnh vào
  formulation/solver?
- **RQ10.** Native multiobjective có lợi hơn sequential portable bao nhiêu, và
  có giữ đúng cùng objective semantics dưới cumulative timeout không?

## 10. Giả thuyết thực nghiệm

- **H6.** CP-T có lợi trên instance có candidate domain nhỏ và xung đột
  agent-slot/user-slot chặt nhờ table, `allDifferent` và count propagation.
- **H7.** MIP-E cạnh tranh hơn khi candidate graph dày và objective tuyến tính
  chi phối, vì LP relaxation và presolve có nhiều lựa chọn để tổng hợp.
- **H8.** CP-I có lợi tương đối khi time-window conflict chi phối, nhưng có thể
  thua CP-T trên duration-one-slot do số optional intervals lớn hơn.
- **H9.** Gurobi và CPLEX có thể khác đáng kể về runtime dù giải cùng MIP-E;
  objective vector tối ưu phải luôn trùng.
- **H10.** Không có backend thắng trên mọi phân tầng load/candidate density;
  portfolio recommendation có giá trị hơn một kết luận tổng quát.

Các giả thuyết này phải được viết là kỳ vọng cần kiểm nghiệm, không phải nhận
định sẵn rằng MIP hay CP “phù hợp hơn” HCORAP.

## 11. Thiết kế thực nghiệm công bằng

### 11.1. Hai tầng so sánh

1. **Solver comparison:** Gurobi MIP và CPLEX MIP chạy đúng cùng MIP-E.
2. **Paradigm/formulation comparison:** MIP-E, CP-T và CP-I có semantics nghiệm
   và objective giống nhau nhưng formulation khác nhau.

Không so sánh trực tiếp số biến/ràng buộc MIP với số interval/global
constraints CP như thể chúng có cùng ý nghĩa.

### 11.2. Implementation

Khuyến nghị dùng:

- một parser C++ dùng chung;
- một immutable instance object dùng chung;
- ba backend adapters: Gurobi C++, CPLEX Concert C++, CP Optimizer Concert C++;
- một verifier C++ độc lập dùng chung;
- cùng output schema JSON.

Điều này phù hợp với protocol hiện tại, trong đó runtime Python không được đưa
vào bảng chính.

### 11.3. Cấu hình cần khóa

- cùng máy, CPU core, giới hạn RAM và power mode;
- một thread cho mọi solver trong bảng chính;
- cùng global/cumulative timeout cho toàn policy;
- cùng parser, preprocessing và candidate set;
- cùng weighted coefficients và thứ tự lexicographic;
- cùng rounding \(L_\delta=\lceil(1-\delta)\operatorname{SIM}^*\rceil\);
- cùng định nghĩa `OPTIMUM`, `TIMEOUT_FEASIBLE`, `TIMEOUT`,
  `INFEASIBLE`, `ERROR`;
- cùng test instances và thứ tự chạy đã randomize bằng seed cố định;
- khóa solver version, API version và parameter file;
- default-parameter campaign và tuned campaign phải tách riêng;
- warm start/no warm start phải là hai thí nghiệm riêng.

Nên warm-up binary/license ngoài tập đo. Trong mỗi run, báo cáo đồng thời:

- end-to-end time: parse + build + solve + extract + verify;
- model-build time;
- solver time;
- verification time.

Global timeout nên trừ model-build time trước khi gọi solver nếu mục tiêu là
cùng ngân sách end-to-end.

### 11.4. KPI chung

- solved/optimal count;
- time to first feasible;
- time to best incumbent;
- time to proof;
- PAR-2;
- peak memory;
- objective vector
  \((\operatorname{COV},\operatorname{SIM},
    \operatorname{CONT},\operatorname{OT})\);
- weighted score khi chạy B0;
- best bound và optimality gap tại timeout nếu backend cung cấp;
- số nghiệm sai bị verifier từ chối, kỳ vọng bằng 0.

KPI riêng MIP như node count không nên so trực tiếp với CP branches/fails.
Chúng chỉ dùng để giải thích hành vi bên trong từng họ solver.

### 11.5. Phân tầng instance

Ít nhất phân tầng theo:

- load ratio \(\rho=S/\sum_a(H_a^N+H_a^E)\);
- số candidate trung bình trên service;
- tỷ lệ service có một candidate;
- time-window width/tightness;
- số service trên user;
- sequence length;
- tỷ lệ agent/service/slot symmetry;
- tỷ lệ capacity cần dùng để đạt full coverage.

Phân tích paired theo từng instance, không chỉ so median toàn bộ benchmark.

## 12. Validation bắt buộc

Mọi backend phải xuất danh sách \((a,s,t)\). Verifier độc lập:

1. kiểm tra candidate membership;
2. kiểm tra mỗi service đúng một assignment;
3. kiểm tra agent-slot conflict;
4. kiểm tra user-slot conflict;
5. kiểm tra workload capacity;
6. tính lại \(\operatorname{SIM},\operatorname{CONT},\operatorname{OT}\);
7. kiểm tra inherited bounds của từng lexicographic/\(\varepsilon\) stage.

Test suite tối thiểu:

- brute-force tiny instances và nhiều nghiệm đồng tối ưu;
- service không có candidate;
- \(H_a^E=0\), \(H_a^N=0\);
- singleton sequence;
- identical agents và symmetry;
- instance bắt buộc overtime;
- trade-off similarity--continuity;
- trade-off similarity--overtime;
- \(\delta=0\) và các trường hợp ceiling thay đổi lower bound;
- soft coverage: unserved và partially served sequences;
- optimum bằng 0;
- timeout có incumbent nhưng chưa có proof.

## 13. Các lỗi diễn giải cần tránh

- Không gọi CP Optimizer là “CPLEX MIP ở chế độ CP”.
- Không nói \(e_a\) chính xác chỉ từ (M11) nếu overtime chưa được tối ưu.
- Không dùng \(D_q-1\) khi sequence có thể hoàn toàn không được phục vụ.
- Không gọi số service là “giờ” nếu slot duration chưa được định nghĩa là một
  giờ.
- Không đưa \(\operatorname{SIM}_N,\operatorname{CONT}_N,
  \operatorname{OT}_N\) vào solver objective trừ khi đây là một policy mới;
  chúng chủ yếu là reporting metrics.
- Không dùng một big weight để giả lập lexicographic trong bảng chính.
- Không tuyên bố một TIMEOUT incumbent là optimum.
- Không so native multiobjective của một solver với sequential của solver khác
  mà không ghi rõ đây là khác cả thuật toán điều phối.
- Không tune trên toàn benchmark rồi báo kết quả trên chính benchmark đó.

## 14. Đoạn văn ngắn có thể đưa vào kế hoạch nghiên cứu

> Nghiên cứu bổ sung hai baseline MIP và hai formulation CP có cùng feasible
> schedules và objective semantics với mô hình MaxSAT. MIP-E là mô hình
> time-indexed thưa, chỉ tạo biến \(x_{ast}\) trên các bộ ba agent--service--slot
> thỏa đồng thời qualification, agent availability và service time window.
> MIP-E được giải độc lập bằng Gurobi Optimizer và CPLEX Optimizer để tách ảnh
> hưởng của solver khỏi ảnh hưởng của formulation. Đối với IBM ILOG CP
> Optimizer, CP-T biểu diễn mỗi service bằng hai biến nguyên caregiver/slot,
> table constraint trên candidate pairs, `allDifferent` cho các xung đột và
> count expressions cho workload/continuity. CP-I biểu diễn service bằng
> interval variable và lựa chọn caregiver bằng optional alternatives, với
> `forbidStart` và `noOverlap` cho calendar và resource conflicts. CP-T là
> baseline chính cho giả định duration bằng một slot; CP-I là formulation
> ablation và nền tảng cho các mở rộng scheduling.

> Ba tiêu chí được tính thống nhất là similarity
> \(\operatorname{SIM}\), continuity penalty \(\operatorname{CONT}\) và
> overtime \(\operatorname{OT}\). Nghiên cứu chạy riêng weighted objective,
> hai chính sách lexicographic và similarity-budget
> \(\varepsilon\)-constraint. Mọi nghiệm được giải mã thành các bộ ba
> \((a,s,t)\) và kiểm chứng bằng cùng một verifier độc lập. So sánh sử dụng
> cumulative timeout, một CPU thread, cùng instance order và cùng output
> schema; báo cáo cả end-to-end time, time-to-proof, PAR-2, peak memory và
> objective vector. Các chỉ số nội bộ như MIP node count và CP failures chỉ
> được dùng để giải thích trong từng paradigm, không được xem là đại lượng
> trực tiếp tương đương.

## 15. Tài liệu sản phẩm liên quan

- Gurobi multiple objectives:
  <https://docs.gurobi.com/projects/optimizer/en/current/reference/misc/misc/multiobjective.html>
- IBM CPLEX multiobjective optimization:
  <https://www.ibm.com/docs/en/icos/22.1.1?topic=optimization-specifying-multiple-objective-problems>
- IBM CP Optimizer interval constraints:
  <https://www.ibm.com/docs/en/icos/22.1.2?topic=models-modeling-constraints-interval-variables>
- IBM CP Optimizer `alternative`:
  <https://www.ibm.com/docs/en/icos/22.1.1?topic=functions-alternative>
- IBM CP Optimizer `noOverlap`:
  <https://www.ibm.com/docs/en/icos/22.1.1?topic=functions-nooverlap>
- IBM CP Optimizer `presenceOf`:
  <https://www.ibm.com/docs/en/icos/22.1.0?topic=f-presenceof>
- IBM CP Optimizer `allowedAssignments`:
  <https://www.ibm.com/docs/en/icos/22.1.0?topic=functions-allowedassignments>
- IBM CP Optimizer static lexicographic objective:
  <https://www.ibm.com/docs/en/icos/22.1.2?topic=functions-minimizestaticlex>

## 16. Ánh xạ sang implementation hiện tại

Implementation C++ nằm trong `src/proposed/cpp/commercial` và executable
`bin/release/hcorap_commercial`:

| Formulation trong tài liệu | Backend CLI | Source |
|---|---|---|
| MIP-E | `gurobi-mip --formulation mip-e` | `HCORAPMIPModel.cpp`, `GurobiMIPBackend.cpp` |
| MIP-E | `cplex-mip --formulation mip-e` | `HCORAPMIPModel.cpp`, `CplexMIPBackend.cpp` |
| CP-T | `cplex-cp --formulation cp-t` | `CplexCPBackend.cpp::solveCPT` |
| CP-I | `cplex-cp --formulation cp-i` | `CplexCPBackend.cpp::solveCPI` |

Hai MIP backend dịch cùng một solver-neutral linear model; đây là invariant
quan trọng để so Gurobi với CPLEX như một solver comparison. CP-T và CP-I dựng
trực tiếp bằng Concert CP API vì global constraints/interval semantics không
thể hiện đúng qua linear intermediate representation.

Mọi policy B0/B1/B2 dùng sequential driver chung. Coverage mềm luôn được tối
ưu và cố định trước. Mọi overtime indicator và continuity OR đều là liên kết
hai chiều nên metric không phụ thuộc objective đang chạy. Mỗi incumbent được
giải mã thành \((a,s,t)\), kiểm chứng độc lập, rồi mới được ghi vào JSON.

Hướng dẫn build, tham số khóa, preset campaign và trạng thái validation nằm
trong [`COMMERCIAL_SOLVERS.md`](COMMERCIAL_SOLVERS.md).

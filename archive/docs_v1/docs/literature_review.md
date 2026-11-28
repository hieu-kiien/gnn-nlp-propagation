# Tổng quan Tài liệu Học thuật & Xác minh Công trình Nền tảng

Tài liệu này tổng hợp và xác minh thông tin xuất bản chính thức (tác giả, năm công bố, hội nghị/tạp chí, mã DOI/arXiv), công thức toán học cốt lõi và hạn chế của các công trình nền tảng liên quan trực tiếp đến đề tài **"Mạng nơ-ron đồ thị và xử lý ngôn ngữ tự nhiên trong mô hình hóa sự phân cực quan điểm và lan truyền thông tin trên mạng xã hội"** (kiến trúc đề xuất **SBi-FAGAT**). Mọi trích dẫn tương ứng được lưu chuẩn BibTeX tại [`paper/references.bib`](../paper/references.bib).

---

## 1. GCN — Graph Convolutional Networks (`kipf2017semisupervised`)

- **Tiêu đề**: *Semi-Supervised Classification with Graph Convolutional Networks*
- **Tác giả**: Thomas N. Kipf, Max Welling
- **Xuất bản**: *Proceedings of the 5th International Conference on Learning Representations (ICLR 2017)*, 2017
- **Liên kết xác minh**: OpenReview `https://openreview.net/forum?id=SJU4ayYgl` | `arXiv:1609.02907`
- **Công thức cốt lõi**: Xấp xỉ bậc nhất phép tích chập phổ đồ thị bằng đa thức Chebyshev đối xứng hóa có vòng tự khép:

$$\mathbf{H}^{(l+1)} = \sigma\left(\tilde{\mathbf{D}}^{-\frac{1}{2}} \tilde{\mathbf{A}} \tilde{\mathbf{D}}^{-\frac{1}{2}} \mathbf{H}^{(l)} \mathbf{W}^{(l)}\right)$$

trong đó $\tilde{\mathbf{A}} = \mathbf{A} + \mathbf{I}_N$, $\tilde{D}_{ii} = \sum_j \tilde{A}_{ij}$, $\mathbf{W}^{(l)}$ là ma trận trọng số khả huấn và $\sigma(\cdot)$ là hàm kích hoạt phi tuyến (ví dụ ReLU).

- **Hạn chế đối với bài toán dự đoán sớm phân cực & quy mô lan truyền**:
  1. **Bộ lọc thông thấp cố định (*Low-Pass Filtering*)**: Phép chuẩn hóa $\tilde{\mathbf{D}}^{-\frac{1}{2}}\tilde{\mathbf{A}}\tilde{\mathbf{D}}^{-\frac{1}{2}}$ luôn làm trơn đặc trưng giữa các nút kề nhau (giả định *homophily*). Khi cây bình luận có nhiều tranh cãi (*heterophily* — bình luận con phản bác bài cha), phép cộng trung bình không âm làm nhòe ranh giới đối lập và gây suy thoái biểu diễn (*over-smoothing*).
  2. **Vô hướng & Bất biến theo ngữ nghĩa cạnh**: Trọng số kết nối chỉ phụ thuộc bậc nút $(d_i, d_j)$, không phân biệt chiều nhân quả từ bài gốc xuống bình luận (Top-Down) với chiều phản hồi từ cộng đồng lên gốc (Bottom-Up), và không nhận biết sắc thái lập trường (ủng hộ / trung lập / phản bác).

---

## 2. GATv2 — Dynamic Graph Attention Networks (`brody2022how`)

- **Tiêu đề**: *How Attentive are Graph Attention Networks?*
- **Tác giả**: Shaked Brody, Uri Alon, Eran Yahav
- **Xuất bản**: *Proceedings of the 10th International Conference on Learning Representations (ICLR 2022)*, 2022
- **Liên kết xác minh**: OpenReview `https://openreview.net/forum?id=F72ximsx7C1` | `arXiv:2105.14491`
- **Công thức cốt lõi**: Chỉ ra rằng GAT gốc chỉ tạo được *static attention* (thứ hạng láng giềng không phụ thuộc nút truy vấn) và đề xuất đảo thứ tự phép biến đổi tuyến tính vào bên trong hàm phi tuyến `LeakyReLU` để đạt *dynamic attention*:

$$e(\mathbf{h}_i, \mathbf{h}_j) = \mathbf{a}^\top \operatorname{LeakyReLU}\left(\mathbf{W} \left[\mathbf{h}_i \parallel \mathbf{h}_j\right]\right)$$

$$\alpha_{ij} = \operatorname{softmax}_j\left(e(\mathbf{h}_i, \mathbf{h}_j)\right) = \frac{\exp(e(\mathbf{h}_i, \mathbf{h}_j))}{\sum_{k \in \mathcal{N}_i \cup \{i\}} \exp(e(\mathbf{h}_i, \mathbf{h}_k))}, \quad \mathbf{h}_i^{\prime} = \sigma\left(\sum_{j \in \mathcal{N}_i \cup \{i\}} \alpha_{ij} \mathbf{W}\mathbf{h}_j\right)$$

- **Hạn chế đối với bài toán**:
  1. **Ràng buộc lồi không âm của `softmax`**: Mọi hệ số chú ý $\alpha_{ij} \in (0, 1)$ và $\sum_j \alpha_{ij} = 1$. Do không thể sinh trọng số âm, GATv2 chỉ điều chỉnh được mức độ *làm trơn*, không thể tạo cơ chế đẩy/khuếch đại sự khác biệt vector giữa hai quan điểm đối kháng (không hoạt động như bộ lọc thông cao).
  2. **Thiếu giám sát lập trường tường minh**: Nếu không có tín hiệu cạnh từ Cross-Encoder, tầng chú ý phải tự dò tìm tương đồng/đối lập hoàn toàn từ phép chiếu tuyến tính $\mathbf{W}\mathbf{h}$, kém hiệu quả khi số mẫu huấn luyện cây lan truyền bị giới hạn ở $K=20$.

---

## 3. SGCN — Signed Graph Convolutional Networks (`derr2018signed`)

- **Tiêu đề**: *Signed Graph Convolutional Networks*
- **Tác giả**: Tyler Derr, Yao Ma, Jiliang Tang
- **Xuất bản xác minh**: *2018 IEEE International Conference on Data Mining (ICDM)*, pp. 929–934, 2018
- **Liên kết xác minh**: DOI `10.1109/ICDM.2018.00113` | `arXiv:1808.06354`
  > *Ghi chú xác minh tài liệu*: Một số tài liệu thứ cấp đôi khi ghi nhầm SGCN xuất bản tại WSDM 2018. Qua đối chiếu trực tiếp, bài báo hội nghị chính thức của Derr et al. (2018) được công bố tại **IEEE ICDM 2018** (pp. 929–934).
- **Công thức cốt lõi**: Dựa trên **Lý thuyết Cân bằng Cấu trúc (*Structural Balance Theory*)** — phân tách đường đi thành tập cân bằng $B$ (số cạnh âm chẵn: *"bạn của bạn là bạn"*, *"kẻ thù của kẻ thù là bạn"*) và tập mất cân bằng $U$ (số cạnh âm lẻ). Mỗi nút duy trì hai vector biểu diễn song song $\mathbf{H}_b^{(l)}$ ("bạn") và $\mathbf{H}_u^{(l)}$ ("thù"):

$$\mathbf{H}_b^{(l+1)} = \sigma\left(\tilde{\mathbf{D}}_b^{-1}\tilde{\mathbf{A}}_b \mathbf{H}_b^{(l)} \mathbf{W}_b^{(1)} + \tilde{\mathbf{D}}_u^{-1}\tilde{\mathbf{A}}_u \mathbf{H}_u^{(l)} \mathbf{W}_b^{(2)}\right)$$

$$\mathbf{H}_u^{(l+1)} = \sigma\left(\tilde{\mathbf{D}}_u^{-1}\tilde{\mathbf{A}}_u \mathbf{H}_b^{(l)} \mathbf{W}_u^{(1)} + \tilde{\mathbf{D}}_b^{-1}\tilde{\mathbf{A}}_b \mathbf{H}_u^{(l)} \mathbf{W}_u^{(2)}\right)$$

và ghép nối đầu ra $\mathbf{z}_i = [\mathbf{h}_{b,i}^{(L)} \parallel \mathbf{h}_{u,i}^{(L)}]$.

- **Hạn chế đối với bài toán**:
  1. **Rời rạc hóa nhị phân $\{-1, +1\}$**: SGCN yêu cầu dấu cạnh cứng $+1$ hoặc $-1$, đánh mất thông tin xác suất liên tục và bỏ qua lớp bình luận trung lập/đặt câu hỏi (*neutral*), vốn chiếm tỷ trọng lớn trong mạng xã hội.
  2. **Giả định cân bằng bậc hai dễ gãy trên tranh luận đa chiều**: Trong cây hội thoại, hai người cùng phản bác bài đăng gốc không nhất thiết đồng ý với nhau (họ có thể đại diện cho hai hướng phản biện trái ngược).
  3. **Không xử lý tính có hướng của cây theo thời gian**: Thiết kế cho đồ thị mạng xã hội vô hướng tĩnh, không phân biệt dòng lan truyền gốc $\to$ lá và dòng phản hồi lá $\to$ gốc.

---

## 4. Bi-GCN — Bi-Directional Graph Convolutional Networks (`bian2020rumor`)

- **Tiêu đề**: *Rumor Detection on Social Media with Bi-Directional Graph Convolutional Networks*
- **Tác giả**: Tian Bian, Xi Xiao, Tingyang Xu, Pei-Lin Zhao, Wen-Bing Huang, Yu Rong, Jun-Zhou Huang
- **Xuất bản**: *Proceedings of the AAAI Conference on Artificial Intelligence (AAAI 2020)*, Vol. 34, No. 01, pp. 549–556, 2020
- **Liên kết xác minh**: DOI `10.1609/aaai.v34i01.5393` | `arXiv:2001.06362`
- **Công thức cốt lõi**: Tách cây hội thoại thành hai đồ thị có hướng: **Top-Down ($\mathbf{A}_{\text{TD}}$)** mô hình hóa *lan truyền* từ bài gốc xuống lá và **Bottom-Up ($\mathbf{A}_{\text{BU}}$)** mô hình hóa *phân tán/phản hồi* từ lá dội lên gốc, kết hợp *DropEdge* ($\dot{\mathbf{A}}_\delta = \operatorname{DropEdge}(\mathbf{A}_\delta, p)$) và *Root Feature Enhancement* (ghép biểu diễn nút gốc sau mỗi tầng tích chập: $\tilde{\mathbf{H}}_\delta^{(l)} = [\mathbf{H}_\delta^{(l)} \parallel \mathbf{H}_{\text{root}}^{(l-1)}]$):

$$\mathbf{H}_{\text{TD}}^{(1)} = \operatorname{ReLU}\left(\hat{\mathbf{D}}_{\text{TD}}^{-1}\hat{\mathbf{A}}_{\text{TD}} \mathbf{X} \mathbf{W}_{\text{TD}}^{(0)}\right), \qquad \mathbf{H}_{\text{TD}}^{(2)} = \operatorname{ReLU}\left(\hat{\mathbf{D}}_{\text{TD}}^{-1}\hat{\mathbf{A}}_{\text{TD}} \left[\mathbf{H}_{\text{TD}}^{(1)} \,\middle\|\, \mathbf{X}_{\text{root}}\right] \mathbf{W}_{\text{TD}}^{(1)}\right)$$

$$\mathbf{H}_{\text{BU}}^{(1)} = \operatorname{ReLU}\left(\hat{\mathbf{D}}_{\text{BU}}^{-1}\hat{\mathbf{A}}_{\text{BU}} \mathbf{X} \mathbf{W}_{\text{BU}}^{(0)}\right), \qquad \mathbf{H}_{\text{BU}}^{(2)} = \operatorname{ReLU}\left(\hat{\mathbf{D}}_{\text{BU}}^{-1}\hat{\mathbf{A}}_{\text{BU}} \left[\mathbf{H}_{\text{BU}}^{(1)} \,\middle\|\, \mathbf{X}_{\text{root}}\right] \mathbf{W}_{\text{BU}}^{(1)}\right)$$

sau đó ghép nối đầu ra tầng 2 với gốc $\left[\mathbf{H}_\delta^{(2)} \parallel \mathbf{H}_{\text{root}}^{(1,\delta)}\right]$ trên cả hai nhánh, gộp trung bình ($\operatorname{MEAN}$) rồi hợp nhất hai hướng để phân loại tin đồn (trong cài đặt PyG đối chứng có thể dùng `GCNConv` chuẩn hóa đối xứng tương ứng $\tilde{\mathbf{D}}_\delta^{-1/2}\tilde{\mathbf{A}}_\delta\tilde{\mathbf{D}}_\delta^{-1/2}$).

- **Hạn chế đối với bài toán**:
  1. **Mù lập trường (*Stance-Blind*)**: Bi-GCN vẫn dùng tích chập không dấu ($\hat{\mathbf{A}}_{\text{TD}}, \hat{\mathbf{A}}_{\text{BU}} \ge 0$), đối xử tương đương giữa một bình luận đồng tình lan tỏa tin giả và một bình luận phản bác/đính chính mạnh mẽ.
  2. **Chỉ dùng phép gộp trung bình & đơn nhiệm**: `MEAN` readout làm mờ các cụm bình luận đối kháng cục bộ và kiến trúc chỉ tối ưu cho phân loại tin đồn, không dự đoán đồng thời quy mô lan truyền ($y_{\text{size}}$) hay mức độ phân cực ($y_{\text{polar}}$).

---

## 5. FAGCN — Frequency Adaptation Graph Convolutional Network (`bo2021beyond`)

- **Tiêu đề**: *Beyond Low-frequency Information in Graph Convolutional Networks*
- **Tác giả**: Deyu Bo, Xiao Wang, Chuan Shi, Huawei Shen
- **Xuất bản**: *Proceedings of the AAAI Conference on Artificial Intelligence (AAAI 2021)*, Vol. 35, No. 5, pp. 3950–3957, 2021
- **Liên kết xác minh**: DOI `10.1609/aaai.v35i5.16514` | `arXiv:2101.00797`
- **Công thức cốt lõi**: Đưa ra cơ chế tự đóng mở cổng tần số (*self-gating*) sinh trọng số cạnh có dấu $\alpha_{ij}^G \in (-1, 1)$ bằng hàm $\tanh$, cho phép mạng tự thích nghi giữa lọc thông thấp ($\alpha_{ij}^G > 0$) và lọc thông cao ($\alpha_{ij}^G < 0$):

$$\alpha_{ij}^G = \tanh\left(\mathbf{g}^\top \left[\mathbf{h}_i \parallel \mathbf{h}_j\right]\right)$$

$$\mathbf{h}_i^{(l+1)} = \varepsilon \mathbf{h}_i^{(0)} + \sum_{j \in \mathcal{N}_i} \frac{\alpha_{ij}^G}{\sqrt{d_i d_j}} \mathbf{h}_j^{(l)}$$

trong đó $\varepsilon$ là hệ số neo biểu diễn ban đầu $\mathbf{h}_i^{(0)}$ để chống *over-smoothing*.

- **Hạn chế đối với bài toán**:
  1. **Thiếu dẫn hướng lập trường ngữ nghĩa cặp câu**: Cổng $\alpha_{ij}^G$ chỉ suy diễn gián tiếp từ phép ghép hai vector nút độc lập $[\mathbf{h}_i \parallel \mathbf{h}_j]$, không tận dụng được khả năng chú ý chéo (*cross-attention*) trực tiếp giữa văn bản cha và văn bản con của Cross-Encoder. Trong điều kiện cây cực nhỏ ($K \le 20$), việc tự học dấu $\alpha_{ij}^G$ từ đầu dễ nhiễu.
  2. **Chuẩn hóa đối xứng vô hướng**: Công thức $\frac{1}{\sqrt{d_i d_j}}$ mặc định đồ thị vô hướng, chưa phân tách được hai cơ chế lan truyền bất đối xứng Top-Down và Bottom-Up của cây hội thoại.

---

## 6. Quantifying Controversy on Social Media (`garimella2018quantifying`, `garimella2016quantifying`)

- **Tiêu đề**: *Quantifying Controversy on Social Media* (bản tạp chí mở rộng); tiền thân *Quantifying Controversy in Social Media* (hội nghị WSDM)
- **Tác giả**: Kiran Garimella, Gianmarco De Francisci Morales, Aristides Gionis, Michail Mathioudakis
- **Xuất bản xác minh**:
  - Bản tạp chí chính thức: *ACM Transactions on Social Computing (TSC)*, Vol. 1, No. 1, Article 3, pp. 1–26, 2018 — DOI `10.1145/3140565` | `arXiv:1507.05224`.
  - Bản hội nghị tiền thân: *Proceedings of the Ninth ACM International Conference on Web Search and Data Mining (WSDM 2016)*, pp. 33–42, 2016 — DOI `10.1145/2835776.2835792`.
- **Công thức cốt lõi**: Dựng đồ thị tương tác người dùng, phân hoạch thành hai phe đối lập $(X, Y)$ rồi đo mức độ cô lập buồng vang (*echo chamber*) bằng chỉ số **Random Walk Controversy (RWC)**:

$$\operatorname{RWC} = P_{XX} P_{YY} - P_{YX} P_{XY}$$

trong đó $P_{AB} = P(\text{bắt đầu ở } A \mid \text{kết thúc ở } B)$ là xác suất có điều kiện của bước đi ngẫu nhiên trên đồ thị phân hoạch.

- **Hạn chế đối với bài toán & phạm vi áp dụng Hai tầng**:
  1. **Phụ thuộc đồ thị trưởng thành cỡ lớn**: RWC đòi hỏi đồ thị tương tác/retweet quy mô lớn, dày đặc đủ để thuật toán phân hoạch hai cụm cộng đồng rõ nét. Trên **Tầng 1** (một cây hội thoại đơn lẻ chỉ có tối đa $K=20$ bình luận ban đầu), cấu trúc liên thông hình cây bị chi phối bởi nút gốc khiến RWC suy biến và mất độ nhạy. Tuy nhiên ở **Tầng 2** (`src/data/user_network.py`), khi chiếu toàn bộ tương tác xuyên chủ đề lên mạng người dùng vĩ mô (`13,718` người dùng VN; `50,062` người dùng PHEME), RWC và Signed Modularity ($Q_{\text{signed}}$, `kunegis2010spectral`) lại đo lường rất sắc nét mức độ phân cực hai phe.
  2. **Bỏ qua nội dung phản bác trong văn bản**: Phương pháp thuần cấu trúc (*structure-only*) coi mọi cạnh kết nối (dù là trả lời để tranh luận gay gắt hay retweet để tán thành) là như nhau nếu không phân tích ngữ nghĩa lập trường.

---

## 7. PhoBERT — Pre-trained Language Models for Vietnamese (`nguyen2020phobert`)

- **Tiêu đề**: *PhoBERT: Pre-trained language models for Vietnamese*
- **Tác giả**: Dat Quoc Nguyen, Anh Tuan Nguyen
- **Xuất bản**: *Findings of the Association for Computational Linguistics: EMNLP 2020*, pp. 1037–1042, 2020
- **Liên kết xác minh**: DOI `10.18653/v1/2020.findings-emnlp.92` | `arXiv:2003.00744`
- **Công thức cốt lõi**: Áp dụng quy trình huấn luyện tiền kỳ RoBERTa (Masked Language Modeling động, bỏ Next Sentence Prediction) trên kho ngữ liệu 20GB văn bản tiếng Việt đã **tách từ âm tiết ghép (*word-level segmentation*)** kết hợp từ điển BPE 64K:

$$\mathcal{L}_{\text{MLM}}(\theta) = -\sum_{i \in \mathcal{M}} \ln P_\theta\left(w_i \mid \tilde{\mathbf{w}}_{\setminus \mathcal{M}}\right)$$

- **Hạn chế đối với bài toán**:
  1. **Giới hạn độ dài ngữ cảnh & nhạy cảm với tách từ mạng xã hội**: Ngữ cảnh tối đa 256 token không thể nạp trọn một cây bình luận dài; văn bản bình luận mạng xã hội tiếng Việt chứa nhiều tiếng lóng, teencode, emoji dễ gãy qua tách từ âm tiết ghép (do đó trong nhánh tiếng Việt của SBi-FAGAT ta ưu tiên dùng `ViSoBERT` trực tiếp trên văn bản chuẩn hóa Unicode NFC thay cho `PhoBERT`).
  2. **Mù cấu trúc cây**: Transformer đơn lẻ chỉ mã hóa tốt ngữ nghĩa từng nút hoặc từng cặp câu `(cha, con)`, cần một kiến trúc GNN phía trên để tổng hợp cấu trúc phân nhánh và động lực thời gian.

---

## 8. RoBERTa — A Robustly Optimized BERT Pretraining Approach (`liu2019roberta`)

- **Tiêu đề**: *RoBERTa: A Robustly Optimized BERT Pretraining Approach*
- **Tác giả**: Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov
- **Xuất bản**: *arXiv preprint arXiv:1907.11692*, 2019
- **Liên kết xác minh**: DOI `10.48550/arXiv.1907.11692` | `arXiv:1907.11692`
- **Công thức cốt lõi**: Cải tiến BERT bằng *Dynamic Masking* theo từng epoch, loại bỏ tác vụ NSP (*Next Sentence Prediction*), mở rộng batch size và kho dữ liệu huấn luyện (160GB văn bản tiếng Anh) với Byte-Level BPE (50K subwords), tối ưu hóa hàm mất mát MLM động và hỗ trợ xuất cặp câu chuẩn qua thẻ phân tách `<s> A </s></s> B </s>`.
- **Hạn chế đối với bài toán**:
  1. **Chi phí bộ nhớ bậc hai $\mathcal{O}(L_{\text{seq}}^2)$**: Không khả thi khi nối toàn bộ các nhánh bình luận trên GPU 16GB VRAM nếu huấn luyện đầu-cuối cùng lúc với đồ thị.
  2. **Thiếu biểu diễn cấu trúc phân cấp**: Khi làm phẳng các bình luận thành chuỗi văn bản tuyến tính, mô hình đánh mất quan hệ cha-con, độ sâu hội thoại và khoảng trễ thời gian $\Delta t_v$ giữa các lượt tương tác.

---

## 9. ViSoBERT & UIT-ViHSD — Mô hình Ngôn ngữ và Kho Ngữ liệu Mạng xã hội Tiếng Việt (`nguyen2023visobert`, `luu2021vihsd`)

- **ViSoBERT (`nguyen2023visobert`)**:
  - **Tiêu đề**: *ViSoBERT: A Pre-Trained Language Model for Vietnamese Social Media Text Processing*
  - **Tác giả**: Nam Quoc Nguyen, Thang Chau Phan, Duc-Vu Nguyen, Kiet Van Nguyen
  - **Xuất bản**: *EMNLP 2023* (Main Conference), pp. 5191–5207, ACL Anthology `2023.emnlp-main.315` — DOI `10.18653/v1/2023.emnlp-main.315` | `arXiv:2310.11166`.
  - **Vai trò trong SBi-FAGAT**: Là kiến trúc XLM-R tiền huấn luyện chuyên biệt trên ngữ liệu mạng xã hội tiếng Việt (hỗ trợ emoji, teencode, văn phong diễn đàn/Reddit với tokenizer SentencePiece), đảm nhiệm vai trò **bộ mã hóa nút Ancestor-Conditioned SpanPool đóng băng** ($\mathbf{e}_v^{\text{Span}} \in \mathbb{R}^{768}$), **bộ chấm độ gay gắt** ($h_{uv}$) khi tinh chỉnh trên `UIT-ViHSD`, và **bộ mã hóa chéo** `DualHeadCrossEncoder` nhánh tiếng Việt.
- **UIT-ViHSD (`luu2021vihsd`)**:
  - **Tiêu đề**: *A Large-scale Dataset for Hate Speech Detection on Vietnamese Social Media Texts*
  - **Tác giả**: Son T. Luu, Kiet Van Nguyen, Ngan Luu-Thuy Nguyen
  - **Xuất bản**: *IEA/AIE 2021* (*Advances and Trends in Artificial Intelligence. Artificial Intelligence Practices*) — DOI `10.1007/978-3-030-79457-6_35` | `arXiv:2103.11528`.
  - **Vai trò**: Bộ dữ liệu chuẩn 3 lớp (`Clean`, `Offensive`, `Hate`) dùng để tinh chỉnh mô hình thầy chấm độ gay gắt tiếng Việt không cần gán nhãn thủ công lại từ đầu.

---

## 10. DeBERTaV3 & BERT-NLI / XNLI cho Suy luận Lập trường (`he2023debertav3`, `laurer2024less`)

- **DeBERTaV3 (`he2023debertav3`)**: *DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing* — Pengcheng He, Jianfeng Gao, Weizhu Chen (*ICLR 2023*). Đưa ra cơ chế *disentangled attention* tách nội dung và vị trí tương đối cùng *ELECTRA-style Replaced Token Detection*, giúp mô hình hóa tốt quan hệ ngữ nghĩa cặp tiền đề–giả thuyết.
- **BERT-NLI / Multilingual NLI (`laurer2024less`)**: *Less Annotating, More Classifying: Addressing the Data Scarcity Issue of Supervised Machine Learning with Deep Transfer Learning and BERT-NLI* — Moritz Laurer, Wouter van Atteveldt, Andreu Casas, Kasper Welbers (*Political Analysis*, Vol. 32, No. 1, pp. 84–100, 2024, DOI `10.1017/pan.2023.20`). Chứng minh khả năng suy luận lập trường/chủ đề ít nhãn bằng mô hình NLI đa ngữ (`mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`), được dùng làm mô hình thầy theo **Symmetric 2-Hypothesis Stance Verbalizer** (đánh giá hai giả thuyết đối xứng *ủng hộ* vs. *phản đối*, gán mâu thuẫn kép $\min(c_{\text{sup}}, c_{\text{opp}})$ về *neutral*, tính $s_{uv} = p_{\text{sup}} - p_{\text{opp}}$).

---

## 11. Kho Dữ liệu Chuẩn: PHEME & UPFD (`zubiaga2016analysing`, `kochkina2018pheme`, `dou2021upfd`)

- **PHEME (`zubiaga2016analysing`, `kochkina2018pheme`)**:
  - Arkaitz Zubiaga, Maria Liakata, Rob Procter, Geraldine Wong Sak Hoi, Peter Tolmie (*PLOS ONE*, Vol. 11, No. 3, `e0150989`, March 2016, DOI `10.1371/journal.pone.0150989`): *Analysing How People Orient to and Spread Rumours in Social Media by Looking at Conversational Threads*.
  - Elena Kochkina, Maria Liakata, Arkaitz Zubiaga (*Figshare 2018*, DOI `10.6084/m9.figshare.6392078`): *PHEME dataset for Rumour Detection and Veracity Classification* — 9 sự kiện tin tức nóng kèm cây phản hồi và nhãn tin đồn/xác thực.
- **UPFD (`dou2021upfd`)**:
  - Yingtong Dou, Kai Shu, Congying Xia, Philip S. Yu, Lichao Sun (*SIGIR '21*, DOI `10.1145/3404835.3462990` | `arXiv:2104.12259`): *User Preference-aware Fake News Detection* — cung cấp chuẩn kiểm chứng mở rộng liên miền (`GossipCop` / `PolitiFact`, đặc trưng nút BERT 768 chiều/SpaCy/Profile/Content).
- **SemEval RumourEval & TweetEval (`gorrell2019semeval`, `barbieri2020tweeteval`)**:
  - `SemEval-2019 Task 7: RumourEval` (DOI `10.18653/v1/S19-2147`) định nghĩa chuẩn phân loại lập trường hội thoại SDQC phục vụ kiểm định teacher NLI, trong khi `TweetEval` (*Findings of EMNLP 2020*, DOI `10.18653/v1/2020.findings-emnlp.148`) cung cấp bộ mã hóa xã hội Tiếng Anh `cardiffnlp/twitter-roberta-base` và bộ phát hiện ngôn từ công kích `cardiffnlp/twitter-roberta-base-offensive`.

---

## 12. Các Công trình Nâng cấp & Nền tảng Kiểm chứng cho SBi-FAGAT (`platonov2023critical`, `rossi2023edge`, `pfister2013good`, `li2017deepcas`)

1. **Đánh giá GNN dưới Heterophily (`platonov2023critical`)**: Oleg Platonov et al. (*ICLR 2023*, `arXiv:2302.11640`) chứng minh tầm quan trọng của tách phân hoạch nghiêm ngặt không trùng thực thể (`topic_splits.json`), tiền chuẩn hóa (Pre-Norm) và cơ chế kết nối dư khi cạnh mang tính dị biệt lập trường (*heterophily*).
2. **Dir-GNN cho Đồ thị Có hướng Dị biệt (`rossi2023edge`)**: Emanuele Rossi, Bertrand Charpentier, Francesco Di Giovanni, Fabrizio Frasca, Stephan Günnemann, Michael M. Bronstein (*LoG 2023*, PMLR 231:25:1–25:27, 2024, `arXiv:2305.10498`, `https://proceedings.mlr.press/v231/rossi24a.html`) chứng minh việc tách biệt phép tổng hợp theo chiều đi (Top-Down) và chiều ngược (Bottom-Up) kết hợp trộn có học cải thiện mạnh hiệu năng trên đồ thị dị biệt — là cơ sở lý thuyết trực tiếp cho tầng trộn hai hướng Dir-GNN trong `SBi-FAGAT`.
3. **DeepCas & Bimodality Coefficient (`li2017deepcas`, `pfister2013good`)**: Thành phần dự báo quy mô lan truyền theo cửa sổ sớm kế thừa bài toán từ *DeepCas* (*WWW '17*, DOI `10.1145/3038912.3052643`, `arXiv:1611.05373`), trong khi kiểm chứng phân đôi lập trường độc lập tham chiếu công thức *Sarle Bimodality Coefficient* của Pfister et al. (*Frontiers in Psychology 2013*, DOI `10.3389/fpsyg.2013.00700`).

---

## 13. Ma trận Đối sánh Tổng hợp: Khoảng trống Nghiên cứu & Đóng góp của SBi-FAGAT

| Công trình / Mô hình | Lan truyền 2 chiều (Dir-GNN TD + BU) | Tín hiệu cạnh 2D $[s_{uv}, h_{uv}]$ | Bộ lọc đối xứng dấu + Kênh kích thích | Teacher NLI + Hostility $\to$ Reply-Span Student | Công bằng `hidden_dim=64`, 10D `struct_x`, 8D Stats | Đa nhiệm Quy mô + Phân cực (+ Tin đồn) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **GCN** (`kipf2017semisupervised`) | Không (Vô hướng) | Không | Không (Thông thấp) | Không | Có (trên benchmark) | Có (qua `MultiTaskHead`) |
| **GATv2 / EdgeGATv2** (`brody2022how`) | Không (hoặc BiEdgeGATv2 đối chứng) | Có (bản `EdgeGATv2` / `BiEdgeGATv2`) | Không ($\alpha_{ij} > 0$) | Nhận `edge_attr` nếu cấp ngoài | Có | Có |
| **SGCN** (`derr2018signed`) | Không (Vô hướng) | Rời rạc $\{-1, +1\}$ | Cân bằng $B/U$ cứng | Không | Không | Không |
| **Bi-GCN** (`bian2020rumor`) | **Có (TD + BU)** | Không | Không (Thông thấp) | Không | Có | Có |
| **FAGCN** (`bo2021beyond`) | Không (Vô hướng) | Ngầm định $\alpha_{ij}^G$ | **Có ($\tanh$)** | Không | Có | Có |
| **RWC** (`garimella2018quantifying`) | Không | Không | Không (Bước đi ngẫu nhiên) | Không | Suy biến ở Tầng 1 (cây nhỏ); **Dùng ở Tầng 2** | Chỉ đo phân cực |
| **ViSoBERT / Twitter-RoBERTa** (`nguyen2023visobert`, `barbieri2020tweeteval`) | Không | Tinh chỉnh cặp câu | Không | **Nền tảng SpanPool & Cross-Encoder 2D** | Mù cấu trúc cây | Cần kết hợp GNN |
| **SBi-FAGAT (Đề xuất)** | **Có (TD + BU + Dir-GNN Mix)** | **Có ($[s_{uv}, h_{uv}]^\top$)** | **2 kênh Phân cực/Kích thích + Pre-Norm + BFS 64-hop Dipole** | **Symmetric NLI + ViHSD/TweetEval $\to$ Segment-Isolated Student** | **8 mô hình cố định công bằng (`hidden_dim=64`)** | **KL-calibrated BCE + Smooth-L1 + Warmup/EMA** |

**Kết luận tổng hợp**: Hệ thống kết nối **vector ngữ nghĩa nút Ancestor-Conditioned SpanPool** và **tín hiệu cạnh 2D $[s_{uv}, h_{uv}]^\top$** (từ bộ thầy kép `mDeBERTa-v3-xnli` Symmetric Stance Verbalizer + `ViSoBERT`/`twitter-roberta-base-offensive` chưng cất cô lập phân đoạn trên phân hoạch chủ đề nhất quán `topic_splits.json`) trực tiếp vào **hai kênh truyền thông điệp Phân cực có dấu / Kích thích không dấu** của `SBi-FAGAT` ở Tầng 1 (kết hợp phân tích trung thực giới hạn cây nhỏ $\le 21$ nút và đóng góp của 3 lớp lập trường so với kênh công kích), đồng thời mở rộng sang **Tầng 2 Mạng người dùng có dấu vĩ mô** dùng Signed Laplacian (`kunegis2010spectral`), Structural Balance, $Q_{\text{signed}}$ và RWC (`garimella2018quantifying`) trên tập `13,718` người dùng VN & `50,062` người dùng PHEME.

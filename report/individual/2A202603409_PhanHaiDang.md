# Báo cáo Day 7

## Dữ liệu và benchmark

- Nguồn, quyền sử dụng, số trang và số chunk: OpenStax / Rice University, Introduction to Python Programming (https://openstax.org/details/books/introduction-python-programming), CC BY-NC-SA 4.0; 60 trang PDF vật lý 17–76; 258 chunks strategy A kế thừa Day06, chunk size 500 ký tự, overlap 50.
- Vector store, embedding model, phiên bản và số chiều: PostgreSQL 16 + pgvector 0.8.7, dùng lại bảng day6_chunks, strategy A; gemini-embedding-001, 768 chiều; document dùng RETRIEVAL_DOCUMENT, query dùng RETRIEVAL_QUERY; cosine similarity.
- Agent model, cách chạy và phân chia dev/test: gemini-3.5-flash-lite; retrieval dev gồm 15 câu có đáp án; agent dev gồm 17 câu (15 có đáp án và 2 thiếu evidence). Đã chạy test, chưa có nhãn audit; đã chốt cấu hình agent dense 1200 ký tự, đối chứng agent dense 800 ký tự theo quyết định người thực hiện.
- Cách xác định câu có/thiếu evidence và tiêu chí chấm: retrieval dùng data/day6/questions-baseline.json; evidence khớp nguyên văn trong chunk đúng nguồn/trang. Evidence q06 được rút ngắn thành “using the assignment operator (=).” để nằm trọn trong một chunk; giữ nguyên câu hỏi và đáp án. Câu thiếu evidence không nằm trong mẫu số retrieval; đánh giá grounded answer, citation và abstention cần chấm thủ công.

## Retrieval và thử nghiệm

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| Baseline | TF-IDF; top-3; không filter/rerank; 800 ký tự | Mốc lexical retrieval | dev | 10/15 (66,67%) | 13/15 (86,67%) | 10/15 (66,67%) | Trung vị 0,7291 ms; index 0,02967 s; dịch vụ $0 |
| Phương pháp 2 | Dense cosine; strategy A; top-3; không filter/rerank; 800 ký tự | Tìm evidence theo vector ngữ nghĩa | dev | 10/15 (66,67%) | 15/15 (100%) | 10/15 (66,67%) | Trung vị 414,3054 ms; tái sử dụng index; chi phí embedding chưa đo |
| Phương pháp 3 | Hybrid TF-IDF + dense; RRF k=60; 10 candidates/mỗi nhánh; top-3; không filter/rerank; 800 ký tự | Kết hợp thứ hạng lexical và dense để cải thiện ranking | dev | 13/15 (86,67%) | 15/15 (100%) | 13/15 (86,67%) | Trung vị 411,0907 ms; lexical index 0,02238 s; chi phí embedding chưa đo |
| Cải thiện 1 | Agent dense; strategy A; top-3; không filter/rerank; 1200 ký tự; tối đa 4 search; citation-repair-v2 | Tăng context từ 800 lên 1200; context evidence cuối tăng 10/15 lên 12/15 | dev | 10/15 (66,67%), lượt đầu | 15/15 (100%), lượt đầu | 12/15 (80%), context cuối agent | Trung vị toàn agent 3872,22 ms/câu; text model $0,0136877/17 câu |
| Cải thiện 2 | Agent hybrid; RRF k=30; 10 candidates/nhánh; top-3; 800 ký tự; tối đa 4 search; citation-repair-v2 | Giảm RRF k từ 60 xuống 30; ranking/context metrics chưa tăng, lần chạy mới không có lỗi | dev | 13/15 (86,67%), lượt đầu | 15/15 (100%), lượt đầu | 13/15 (86,67%), context cuối agent | Trung vị toàn agent 3463,19 ms/câu; text model $0,0123388/17 câu |
| Cấu hình đã chọn | Agent dense; strategy A; top-3; không filter/rerank; 1200 ký tự; tối đa 4 search; citation-repair-v2 | Chốt bằng dev, đánh giá trên test | test | 3/5 (60%), lượt đầu | 4/5 (80%), lượt đầu | 3/5 (60%), context cuối | Trung vị toàn agent 3394,81 ms; text model $0,0044691/7 câu |
| Baseline agent test | Agent dense; strategy A; top-3; không filter/rerank; 800 ký tự; tối đa 4 search; citation-repair-v2 | Đối chứng context budget | test | 3/5 (60%), lượt đầu | 4/5 (80%), lượt đầu | 4/5 (80%), context cuối | Trung vị toàn agent 4412,22 ms; text model $0,0049762/7 câu |

Một cấu hình có thể đồng thời là phương pháp mới và cải thiện; giải thích rõ. Báo `n/N` và tỷ lệ, ghi đơn vị thời gian/chi phí, so sánh test sau khi chốt. Nêu hai lỗi dev, nguyên nhân, cách thử và lý do chọn cấu hình:

### Nhận xét ba phương pháp ban đầu

Hybrid tăng Hit@1 và Context Evidence Rate thêm 3/15 câu (20 điểm phần trăm) so với TF-IDF và dense. Dense và hybrid đều tìm được evidence trong top-3 của 15/15 câu, nhưng context cuối chỉ giữ evidence ở lần lượt 10/15 và 13/15 câu. Cần đọc trace để phân tích phần mất evidence khi đóng gói context. Chưa chọn cấu hình cuối chỉ từ các chỉ số retrieval.

Context trung bình: TF-IDF 622,2 ký tự; dense 600,07 ký tự; hybrid 622,2 ký tự. Cả ba dùng cùng chính sách top-3, deduplication và first-fit trong budget 800 ký tự, tính cả thông tin nguồn. Thời gian dense/hybrid gồm query embedding API, kết nối/truy vấn database và đóng gói context; TF-IDF chạy cục bộ. Chênh lệch thời gian nhỏ giữa dense và hybrid trong một lần chạy chưa chứng minh hybrid nhanh hơn.

Bằng chứng retrieval: [TF-IDF](../../artifacts/dev-tfidf/summary.json), [dense](../../artifacts/dev-dense/summary.json), [hybrid](../../artifacts/dev-hybrid/summary.json); traces.json nằm cùng thư mục với mỗi summary. Kết quả gốc cũng có tại starter/results/dev-tfidf, dev-dense và dev-hybrid.

## Agent và đánh giá

- Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi: controller Gemini chọn search/answer/abstain; tối đa 4 search, top-3, budget 800 ký tự. Bản citation-repair-v2 cho một lượt sửa action/citation, giữ query gốc ở lần đầu và ưu tiên kết quả search mới trong context tích lũy có deduplication. API/database errors được ghi riêng. Kiểm tra ID citation không thay thế chấm nguồn hỗ trợ câu trả lời.

| Cấu hình / split | Tool use | Số lượt gọi | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|

### Kết quả vận hành agent dev hiện có (chưa chấm chất lượng)

| Retrieval | Tổng câu | Answered | Abstain | Error | service_cost_usd |
|---|---:|---:|---:|---:|---:|
| TF-IDF | 17 | 14 | 3 | 0 | 0,0131420 |
| Dense | 17 | 14 | 3 | 0 | 0,0117855 |
| Hybrid | 17 | 14 | 2 | 1 | 0,0118315 |

Các số answered/abstain là trạng thái chương trình, không phải số câu đúng/từ chối đúng. Chưa có nhãn audit nên chưa điền grounded answer, citation validity, abstention đúng hoặc từ chối nhầm vào bảng chất lượng phía trên.

Chi phí là ước tính token sinh văn bản của gemini-3.5-flash-lite theo giá người dùng cung cấp: $0,30/triệu input tokens và $2,50/triệu output tokens; tính cả lượt sửa citation. Một lượt hybrid thiếu usage được tính $0 theo quy ước người dùng. Chưa gồm query embeddings hoặc hạ tầng, không phải hóa đơn thực tế.

- Đường dẫn trace và summary agent: starter/results/agent-tfidf-dev, starter/results/agent-dense-dev, starter/results/agent-hybrid-dev (thư mục local bị Git ignore; cần sao chép artifacts trước khi nộp). Nhãn thủ công chưa có.
- Kết quả trước/sau, ví dụ lỗi, đánh đổi và giới hạn:

## Thử thách nâng cao (nếu thực hiện)

- Tài liệu, nguồn, quyền sử dụng, số trang và phiên bản:
- Cách xử lý text, bảng, ảnh, biểu đồ; phạm vi index và phần chưa xử lý:
- Cách chia benchmark, schema evidence, model và cấu hình đã chốt trên dev:

| Nhóm test | Đạt / tổng | Tỷ lệ | Lỗi / nhận xét |
|---|---|---|---|
| Bảng | | | |
| Ảnh, sơ đồ | | | |
| Biểu đồ | | | |
| Multi-hop | | | |
| Text | | | |
| Thiếu bằng chứng | | | |
| **Toàn bộ** | | | |

- Trace, nhãn và evidence từng câu; thời gian, chi phí, lỗi hệ thống:
- Giới hạn, cách tái lập và đường dẫn benchmark:


## Thí nghiệm cải thiện 1: dense, context 1200 ký tự

So sánh với agent dense dev context 800 ký tự; giữ nguyên bộ câu hỏi, model gemini-3.5-flash-lite, embedding, strategy A, top-3, tối đa 4 lượt search và chính sách chọn context. Chỉ tăng budget lên 1200 ký tự (tính cả ID và nguồn). Giả thuyết: giữ thêm chunk có evidence, giảm từ chối nhầm; đánh đổi là tăng input tokens và chi phí. Chất lượng thực tế cần chấm grounded answer/citation/abstention, không chỉ đếm answered. Với agent nhiều lượt, model có thể chọn các query khác khi context thay đổi; không mặc định Hit@1/Hit@3 giữ nguyên.

Lệnh chạy từ thư mục gốc Day07 trong CMD:

```bat
python starter\run_agent.py --cases data\agent_cases.json --split dev --retriever dense --budget 1200 --output starter\results\agent-dense-1200-dev
```

### Kết quả đã chạy và đối chiếu

Số liệu summary và traces lấy từ starter/results/agent-dense-1200-dev, đối chứng starter/results/agent-dense-dev. Hit@1/Hit@3 dưới đây tính trên kết quả search đầu tiên bằng câu hỏi gốc; Context Evidence Rate kiểm tra evidence khớp nguyên văn trong chunk đúng nguồn/trang được giữ ở context cuối. Cả ba chỉ số có mẫu số 15 câu có đáp án; thời gian và chi phí tính trên 17 câu agent dev. Đây là phép kiểm tra evidence bằng code, chưa thay thế nhãn chất lượng thủ công.

| Chỉ số | Dense 800 ký tự | Dense 1200 ký tự |
|---|---:|---:|
| Hit@1 lượt đầu | 10/15 (66,67%) | 10/15 (66,67%) |
| Evidence Hit@3 lượt đầu | 15/15 (100%) | 15/15 (100%) |
| Context Evidence Rate cuối agent | 10/15 (66,67%) | 12/15 (80%) |
| Answered / Abstain / Error | 14 / 3 / 0 | 14 / 3 / 0 |
| Câu có đáp án bị từ chối | q07 (1/15) | q07 (1/15) |
| Trạng thái câu thiếu evidence | q21, q22: abstain | q21, q22: abstain |
| Tổng search thực thi | 22 | 21 |
| Context cuối trung bình, trên 17 câu | 593 ký tự | 864,88 ký tự |
| Thời gian toàn agent trung vị/câu | 4846,15 ms | 3872,22 ms |
| Chi phí model sinh văn bản | $0,0117855 | $0,0136877 |

Tăng budget giúp context cuối giữ evidence thêm 2/15 câu (+13,33 điểm phần trăm), nhưng số answered và từ chối nhầm chưa đổi. Chi phí sinh văn bản tăng $0,0019022 (khoảng 16,14%); chưa tính query embeddings. Thời gian trung vị giảm trong lần chạy này, nhưng không kết luận budget lớn làm agent nhanh hơn vì số lượt tìm khác nhau, API/network và model có biến động. Chưa có nhãn grounded answer/citation/abstention đúng nên chưa kết luận cải thiện chất lượng câu trả lời hoặc chọn cấu hình cuối.

Kết quả agent nhiều lượt phải được phân biệt với benchmark retrieval một lượt của ba phương pháp ban đầu. Lưu summary.json và traces.json vào artifacts riêng trước khi nộp; hiện các kết quả này còn ở starter/results bị Git ignore.


## Thí nghiệm cải thiện 2: hybrid RRF k=30

Đối chứng là agent hybrid ban đầu (RRF k=60, 10 candidates/nhánh, top-3, context 800 ký tự, tối đa 4 search). Giữ nguyên corpus strategy A, bộ câu hỏi dev, model, controller và chính sách context. Chỉ giảm hằng số RRF xuống 30: score = sum(1/(30+rank)). Thay đổi này tăng trọng số tương đối của thứ hạng đầu, nhưng không bảo đảm đổi ranking hoặc tăng chất lượng. Đã đo kết quả vận hành và evidence từ trace dưới đây; chất lượng câu trả lời/citation và abstention đúng vẫn cần nhãn thủ công trước khi chọn cấu hình.

```bat
python starter\run_agent.py --cases data\agent_cases.json --split dev --retriever hybrid --rrf-k 30 --candidate-k 10 --budget 800 --output starter\results\agent-hybrid-rrf30-dev
```

### Kết quả cải thiện 2 và đối chiếu

Nguồn: starter/results/agent-hybrid-rrf30-dev/summary.json và traces.json; đối chứng starter/results/agent-hybrid-dev. Hit@1/Hit@3 tính trên lượt tìm đầu tiên bằng query gốc, context evidence kiểm tra khớp nguyên văn đúng nguồn/trang trong context cuối; mẫu số cố định 15 câu có đáp án. Đối chứng có lỗi q03 sau retrieval nên các chỉ số evidence vẫn tính trên trace tìm kiếm đã ghi nhận; đây không phải 15 lần trả lời agent thành công. Thời gian và chi phí tính trên toàn bộ 17 câu, gồm lần lỗi.

| Chỉ số | Hybrid RRF k=60 | Hybrid RRF k=30 |
|---|---:|---:|
| Hit@1 lượt đầu | 13/15 (86,67%) | 13/15 (86,67%) |
| Evidence Hit@3 lượt đầu | 15/15 (100%) | 15/15 (100%) |
| Context Evidence Rate cuối agent | 13/15 (86,67%) | 13/15 (86,67%) |
| Answered / Abstain / Error | 14 / 2 / 1 | 15 / 2 / 0 |
| Câu có đáp án bị từ chối | 0/15; có 1 lỗi q03 | 0/15; không lỗi |
| Trạng thái câu thiếu evidence | q21, q22: abstain | q21, q22: abstain |
| Tổng search thực thi | 20 | 21 |
| Context cuối trung bình, trên 17 câu | 625 ký tự | 625 ký tự |
| Thời gian toàn agent trung vị/câu | 3783,49 ms | 3463,19 ms |
| Chi phí model sinh văn bản | $0,0118315 | $0,0123388 |

Giảm RRF k chưa cải thiện Hit@1, Hit@3 hoặc Context Evidence Rate trong lần thử này. Lần k=30 trả lời 15 câu có đáp án và từ chối 2 câu gắn nhãn thiếu evidence; trạng thái answered không chứng minh câu trả lời đúng hoặc citation hỗ trợ đầy đủ. Cần chấm thủ công trước khi khẳng định chất lượng.

Lần k=60 có lỗi hệ thống ở q03 còn k=30 không có lỗi; không quy việc hết lỗi cho thay đổi RRF k. Chi phí sinh văn bản tăng $0,0005073 (khoảng 4,29%), chưa gồm embeddings. Đối chứng có một lượt thiếu usage được tính $0 theo quy ước, nên chi phí không phản ánh phép so sánh hai lần chạy hoàn toàn tương đương. Chênh lệch thời gian một lần chạy cũng chưa đủ để kết luận k=30 nhanh hơn.

Kết quả này là một thí nghiệm cải thiện có giả thuyết, dù các chỉ số evidence không tăng. Chưa chốt cấu hình test. Lưu summary.json và traces.json vào artifacts riêng trước khi nộp; hiện kết quả còn ở starter/results bị Git ignore.

## Chốt cấu hình và đánh giá test

Người thực hiện chọn **dense 1200 ký tự** làm cấu hình cuối và **dense 800 ký tự** làm baseline agent test. TF-IDF trong bảng retrieval dev vẫn là mốc lexical ban đầu; đây là hai vai trò đối chứng khác nhau. Lý do chọn dense 1200: dev cho thấy context evidence tăng từ 10/15 lên 12/15 so với dense 800, trong khi số câu trả lời/từ chối chưa đổi và chi phí tăng khoảng 16,14%. Lựa chọn này nhằm kiểm tra riêng tác động của budget; chưa chứng minh dense 1200 có chất lượng tổng thể cao nhất trong ba phương pháp.

Cố định cùng model gemini-3.5-flash-lite, embedding gemini-embedding-001 768 chiều, corpus strategy A, top-3, tối đa 4 search, controller citation-repair-v2, chính sách newest-first có deduplication. Chỉ khác budget 800/1200. Từ thời điểm chốt, không dùng kết quả test để chỉnh cấu hình; nếu đã dùng các câu test cũ để chỉnh hệ thống thì phải tạo test mới và ghi rõ trước khi báo đánh giá độc lập.

Kiểm tra bộ test có tối thiểu 5 câu có đáp án và 2 câu thiếu evidence đã xác minh; câu hỏi và nhãn phải giống nhau giữa hai lần chạy. Chạy trong CMD từ gốc repo:

```bat
python starter\run_agent.py --cases data\agent_cases.json --split test --retriever dense --budget 800 --output starter\results\agent-dense-800-test
python starter\run_agent.py --cases data\agent_cases.json --split test --retriever dense --budget 1200 --output starter\results\agent-dense-1200-test
```

Sau khi chạy, chấm ít nhất 5 câu test có đáp án và 2 câu thiếu evidence cho cả hai cấu hình: context 0/1/2; grounded answer 0/1/2; citation có trong context và thực sự hỗ trợ câu trả lời; abstention đúng và từ chối nhầm. Lưu reviews.json riêng cho mỗi cấu hình rồi dùng starter/agent_metrics.py tổng hợp theo starter/README.md. Báo số lỗi riêng, thời gian và chi phí; lưu summary, traces, reviews và metrics vào artifacts. Không dùng số answered làm số câu đúng.

**Trạng thái:** đã chạy hai cấu hình trên 7 câu test; số liệu tự động được ghi dưới đây. Nhãn audit test và phần đánh giá dev thủ công còn cần hoàn thiện.

### Kết quả test đã ghi nhận

Nguồn summary.json và traces.json tại starter/results/agent-dense-800-test và starter/results/agent-dense-1200-test. Cùng 7 câu: q16–q20 có đáp án, q23–q24 gắn nhãn thiếu evidence. Hit@1/Hit@3 đo search đầu tiên; context evidence đo evidence khớp nguyên văn đúng nguồn/trang trong context cuối. Mẫu số retrieval/context là 5, số liệu vận hành tính trên 7 câu.

| Chỉ số | Baseline dense 800 | Cấu hình chọn dense 1200 |
|---|---:|---:|
| Hit@1 lượt đầu | 3/5 (60%) | 3/5 (60%) |
| Evidence Hit@3 lượt đầu | 4/5 (80%) | 4/5 (80%) |
| Context Evidence Rate cuối agent | 4/5 (80%) | 3/5 (60%) |
| Answered / Abstain / Error | 5 / 2 / 0 | 4 / 3 / 0 |
| Câu có đáp án bị từ chối | 0/5 | 1/5: q18 |
| Trạng thái câu thiếu evidence | q23, q24: abstain | q23, q24: abstain |
| Tổng search thực thi | 9 | 8 |
| Context cuối trung bình, trên 7 câu | 646 ký tự | 686,71 ký tự |
| Thời gian toàn agent trung vị/câu | 4412,22 ms | 3394,81 ms |
| Chi phí sinh văn bản | $0,0049762 | $0,0044691 |

Cấu hình 1200 không giữ được mức cải thiện evidence đã thấy trên dev: context evidence test thấp hơn baseline 1/5 câu, đồng thời từ chối q18 có đáp án. Chưa thể kết luận mọi câu answered là đúng; cần đọc câu trả lời, context và citation. Hai câu thiếu evidence đều có trạng thái abstain nhưng vẫn cần đọc để xác nhận không thêm khẳng định thiếu căn cứ. Cấu hình 1200 có chi phí/thời gian thấp hơn trong lần chạy này, cùng số search ít hơn; không kết luận context lớn luôn rẻ hoặc nhanh hơn.

Giữ quyết định chọn cấu hình trước test và báo trung thực kết quả kém hơn; không chỉnh lại bằng test này. Các câu q16–q20 kế thừa bộ test Day06: tính độc lập còn phụ thuộc việc chúng có từng được dùng để chỉnh hệ thống hay chưa, cần xác nhận trong báo cáo cuối. Kết quả hiện chưa đủ nhãn để tính grounded answer, citation validity và context grade 0/1/2. Cần hoàn thiện audit rồi lưu kết quả/nhãn vào artifacts.

## Hồ sơ chấm thủ công đã đóng gói

Xem [hướng dẫn artifacts](../../artifacts/README.md). Đã sao chép summary/traces của ba agent dev, hai cải thiện dev và hai cấu hình test vào artifacts cùng cases.json và reviews.json. Reviews hiện là mẫu chưa chấm (null), không phải nhãn đã xác nhận. Ưu tiên hoàn thiện artifacts/agent-dense-800-test và artifacts/agent-dense-1200-test, sau đó tổng hợp metrics và đưa nhãn dev cần thiết vào báo cáo. Các đường dẫn starter/results phía trên là nguồn chạy ban đầu; bản artifacts tương ứng là bằng chứng được đóng gói để nộp.

## Kết luận thực nghiệm và giới hạn

Đã triển khai ba phương pháp TF-IDF, dense và hybrid RRF trên cùng corpus. Hybrid có ranking/context evidence tốt nhất trong benchmark retrieval dev đã chạy; đây không tự động chứng minh chất lượng câu trả lời agent. Hai thí nghiệm cải thiện đều được đo và báo cả kết quả không tăng: dense1200 tăng context evidence dev nhưng không giữ cải thiện trên test; hybrid RRF30 không tăng chỉ số evidence dev. Dense1200 được chọn trước test để đối chiếu riêng budget với dense800; trên test cấu hình chọn có context evidence thấp hơn và từ chối thêm q18. Giữ lựa chọn và báo kết quả này, không đổi lại cấu hình theo test.

### Hai vấn đề dev đã quan sát từ trace

1. Dense q07: cả budget800 và1200 đều abstain dù benchmark có đáp án. Tăng budget không giải quyết trường hợp này. Cần audit context và nguồn để tách lỗi retrieval, selection và quyết định model; chưa gán nguyên nhân nội dung khi chưa chấm.
2. Hybrid k60 q03: trace ghi ServerError, HTTP503; đây là lỗi dịch vụ, không phải evidence thiếu hoặc grounded answer sai. Lần k30 không gặp lỗi nhưng không kết luận đổi RRF khắc phục503. Lần gọi thiếu usage tính $0 theo quy ước. Pipeline ghi lỗi và không thay bằng kết quả giả.

Giới hạn: corpus chỉ gồm phạm vi trang đã chọn; evidence matching nguyên văn bỏ sót các cách diễn đạt hợp lệ; context first-fit/newest-first có thể loại evidence cũ; biến động model/API và số lượt tìm ảnh hưởng chi phí/thời gian. Chưa có kiểm định nhiều lần chạy; giá sinh văn bản do người dùng cung cấp, không phải hóa đơn. Nhãn audit chưa hoàn thiện nên chưa kết luận grounded answer/citation validity/abstention correctness. Test kế thừa Day06 cần xác nhận không dùng để chỉnh hệ thống.

## Tái lập và trạng thái kiểm tra

- Hướng dẫn cài đặt, tái tạo database từ chunks chia sẻ, chạy ba phương pháp, hai cải thiện và test: SETUP_CMD.md ở gốc repo.
- Code Day07 không import repo Day06; dùng database hiện hữu hoặc starter/rebuild_store.py trên database PostgreSQL/pgvector mới. Tái tạo embeddings gọi API và phát sinh chi phí; không ghi đè strategy A đã có.
- Đã kiểm tra JSON artifacts đọc được, bản sao summary/traces khớp nguồn và số nhãn citation trong mẫu review khớp trace.
- Benchmark có đúng 15 câu có đáp án +2 câu thiếu evidence ở dev, 5+2 ở test. Lý do absence hiện là mô tả; bước đọc nguồn và ghi cách xác minh vẫn thuộc phần audit còn lại.
- Đã thử chạy unittest trong môi trường công cụ nhưng Python bị chặn khởi chạy; chưa tuyên bố tests pass. Người nộp chạy `python -m unittest discover -s tests -v` trong CMD tại gốc repo và giữ kết quả xác nhận.
- Không thực hiện phần bonus. Chưa có nhãn thủ công, chưa tạo metrics chất lượng thay thế bằng số đếm answered. Quyền truy cập repo và tính độc lập test cần người nộp xác nhận trước khi nộp.

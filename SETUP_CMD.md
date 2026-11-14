# Day07 — hướng dẫn chạy bằng CMD

Repo chứa code độc lập; dùng lại database Day06 qua DATABASE_URL, không import code từ repo Day06.

## Cài đặt và kiểm tra

```bat
py -3 -m venv .venv
.venv\Scripts\activate
python -m pip install -r starter\requirements.txt
copy .env.example .env
notepad .env
python starter\prepare_store.py --strategy A
python -m unittest discover -s tests -v
```

Nếu đã có .env, không copy đè. Điền GEMINI_API_KEY và DATABASE_URL của PostgreSQL/pgvector đang chạy. Query embedding: gemini-embedding-001, RETRIEVAL_QUERY, 768 chiều; document index phải dùng cùng model, RETRIEVAL_DOCUMENT. Text model: gemini-3.5-flash-lite. Không commit secret.

## Tái tạo database khi không có database Day06

Có thể dùng PostgreSQL 16 + pgvector 0.8.7 mới. Ví dụ Docker local (chỉ chạy nếu không dùng database cũ):

```bat
docker run --name day07-reproduction -e POSTGRES_USER=day6 -e POSTGRES_PASSWORD=day6_local_only -e POSTGRES_DB=day6_lab -p 127.0.0.1:55435:5432 -v day07_reproduction:/var/lib/postgresql/data -d pgvector/pgvector:0.8.7-pg16
```

Đợi PostgreSQL sẵn sàng, đặt DATABASE_URL trong .env thành postgresql://day6:day6_local_only@127.0.0.1:55435/day6_lab, rồi:

```bat
python starter\rebuild_store.py
python starter\prepare_store.py --strategy A
```

Script tạo bảng day6_chunks từ data/day6/chunks.json và gọi API để tạo lại embeddings; có chi phí. Từ chối ghi đè strategy A đã có. Không cần repo Day06. Kết quả tái tạo có thể khác nhẹ do API/model; artifacts là kết quả các lần chạy đã ghi nhận.

## Chạy retrieval dev

```bat
python starter\baseline.py --chunks data\day6\chunks.json --questions data\day6\questions-baseline.json --split dev --output starter\results\dev-tfidf
python starter\dense.py --split dev --output starter\results\dev-dense
python starter\hybrid.py --split dev --output starter\results\dev-hybrid
```

## Chạy agent và hai cải thiện

```bat
python starter\run_agent.py --cases data\agent_cases.json --split dev --retriever tfidf --output starter\results\agent-tfidf-dev
python starter\run_agent.py --cases data\agent_cases.json --split dev --retriever dense --output starter\results\agent-dense-dev
python starter\run_agent.py --cases data\agent_cases.json --split dev --retriever hybrid --output starter\results\agent-hybrid-dev
python starter\run_agent.py --cases data\agent_cases.json --split dev --retriever dense --budget 1200 --output starter\results\agent-dense-1200-dev
python starter\run_agent.py --cases data\agent_cases.json --split dev --retriever hybrid --rrf-k 30 --candidate-k 10 --output starter\results\agent-hybrid-rrf30-dev
```

Mặc định top-k=3, budget=800 ký tự, max-searches=4, RRF k=60 và 10 candidates mỗi nhánh. Controller JSON action chọn search/answer/abstain; giữ khung agentic.py, sửa citation tối đa một lượt, không dùng gold/answerable/absence_reason làm input model. Context tích lũy dedup, ưu tiên search mới và first-fit trong budget. Không dùng filter từ gold. Citation ID đúng chưa chứng minh nội dung có nguồn hỗ trợ. Lỗi không có fallback giả.

## Test của cấu hình đã chốt

```bat
python starter\run_agent.py --cases data\agent_cases.json --split test --retriever dense --budget 800 --output starter\results\agent-dense-800-test
python starter\run_agent.py --cases data\agent_cases.json --split test --retriever dense --budget 1200 --output starter\results\agent-dense-1200-test
```

Đã có artifacts, không cần chạy lại để đọc/chấm. Lệnh ghi đè output cũ; dùng thư mục khác khi thử lại. Chọn dense1200 trước test, dense800 làm đối chứng agent; TF-IDF là baseline retrieval dev. Không chỉnh cấu hình bằng test đã xem.

## Kết quả, chi phí và chấm

Xem artifacts/README.md và report/individual/MSSV_HoTen.md. Các review null chưa được chấm; metrics chất lượng chỉ tạo sau khi chấm đủ. Tool/model calls, context, citations, thời gian và lỗi nằm trong traces.json. Test mock chỉ kiểm tra hành vi code.

service_cost_usd là ước tính sinh văn bản: $0.30/M input và $2.50/M output theo giá người dùng cung cấp; gồm repair calls, thiếu usage tính $0 theo quy ước. Chưa gồm query embeddings/hạ tầng. Tính offline bằng starter/costs.py, không cần chạy lại API. Model khác/cache/thinking chưa có giá hỗ trợ giữ null.

Dữ liệu: OpenStax / Rice University, Introduction to Python Programming, https://openstax.org/details/books/introduction-python-programming ; CC BY-NC-SA 4.0, truy cập 2026-11-13. Text trích/chunk là bản chuyển thể, giữ attribution, noncommercial và share-alike. 60 trang PDF vật lý 17–76, strategy A 258 chunks, 500 ký tự/overlap50. Evidence q06 trong questions-baseline.json được rút ngắn để nằm trong một chunk; questions.json giữ bản chuyển gốc. Benchmark agent có 15+2 dev và 5+2 test. Quyền sử dụng/nguồn và tính độc lập test cần người nộp xác nhận.

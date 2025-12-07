# Vietnamese Sentiment Assistant
Ứng dụng phân tích cảm xúc tiếng Việt sử dụng mô hình PhoBERT đã fine-tune, xây dựng giao diện với Streamlit và lưu lịch sử bằng SQLite.

## Mô tả
Hệ thống phân loại câu tiếng Việt thành ba nhãn: POSITIVE, NEGATIVE và NEUTRAL.
Ứng dụng hỗ trợ tiền xử lý văn bản, mở rộng từ viết tắt, hiển thị độ tự tin và lưu lịch sử phân tích.

## Tính năng chính
* Fine-tuned PhoBERT cho tiếng Việt
* Xử lý và chuẩn hóa văn bản đầu vào
* Mở rộng từ viết tắt từ abbreviation.csv
* Giao diện Streamlit đơn giản, dễ sử dụng
* Lưu lịch sử vào SQLite và phân trang
* Confidence score cho mỗi dự đoán

## Kiến trúc xử lý
1. Nhận câu tiếng Việt
2. Tiền xử lý: viết thường, mở rộng từ viết tắt, loại ký tự đặc biệt
3. Mô hình PhoBERT phân loại cảm xúc
4. Áp dụng ngưỡng confidence (<0.5 được gán NEUTRAL)
5. Lưu kết quả vào SQLite
6. Hiển thị trên giao diện Streamlit

## Cấu trúc thư mục
```
Vietnamese-Sentiment-Analysis/
├── app.py
├── sentiment_model.py
├── database.py
├── finetune_phobert.py
├── sentiment_data.csv
├── abbreviation.csv
├── requirements.txt
└── README.md
```

## Công nghệ sử dụng
* Streamlit
* PyTorch, Transformers
* PhoBERT (vinai/phobert-base)
* SQLite
* Python 3.8+

## Hướng dẫn cài đặt
### 1. Clone dự án

```
git clone https://github.com/quangbui5504/Vietnamese-Sentiment-Analysis.git
cd Vietnamese-Sentiment-Analysis
```

### 2. Tạo môi trường ảo

```
python -m venv .venv
.venv\Scripts\activate   # Windows
```

### 3. Cài đặt thư viện

```
pip install -r requirements.txt
```

### 4. Fine-tune mô hình (bắt buộc)

```
python finetune_phobert.py
```

### 5. Chạy ứng dụng

```
streamlit run app.py
```

Truy cập tại: [http://localhost:8501](http://localhost:8501)

## Fine-tuning PhoBERT
* Dữ liệu: sentiment_data.csv (hơn 8k mẫu)
* Chia train/validation theo tỉ lệ 80/20
* Tham số: 3 epochs, learning rate 2e-5, batch size 8
* Kết quả: thư mục phobert-sentiment-final/

Hiệu năng dự kiến: accuracy ~85%.

## Cấu trúc database
* Tệp SQLite: sentiment.db
* Bảng: sentiments(id, text, sentiment, timestamp)
* Hỗ trợ phân trang 5 dòng mỗi trang


import streamlit as st
import math
from sentiment_model import SentimentService
from database import init_db, insert_record, get_by_page, get_total_count



CUSTOM_CSS = """
<style>

    .stApp {
        background-color: #e8f5e9 !important;
    }

    .title {
        padding: 15px;
        color: white;
        text-align: center;
        font-size: 34px;
        border-radius: 12px;
        background: linear-gradient(90deg, #2e7d32, #66bb6a);
        margin-bottom: 25px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.18);
        letter-spacing: 1px;
    }

    .block {
        background: #f1f8e9; /* Kem xanh lá */
        padding: 22px;
        border-radius: 14px;
        border: 1px solid #dcedc8;
        box-shadow: 0 3px 12px rgba(0, 0, 0, 0.10);
        margin-bottom: 25px;
        color: #1b5e20 !important; /* Xanh đậm, dễ đọc */
    }

    body, p, span, label, .stTextInput, .stTextArea {
        color: #1b5e20 !important; /* Xanh rừng đậm */
        font-weight: 500;
    }

    label {
        font-weight: 600 !important;
        font-size: 16px;
    }

    thead tr th {
        background-color: #c8e6c9 !important;
        color: #1b5e20 !important;
        font-weight: 700 !important;
    }

    tbody tr td {
        background-color: #f9fff6 !important;
        color: #1b5e20 !important;
    }

    .history-title, h2, h3 {
        color: #1b5e20 !important;
    }

    button[kind="primary"] {
        background-color: #43a047 !important;
        color: white !important;
        border-radius: 6px;
    }

</style>
"""


st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.set_page_config(
    page_title="Vietnamese Sentiment Assistant", 
    layout="centered"
)

@st.cache_resource
def _service():
    init_db()
    svc = SentimentService(use_tokenize=True, abbr_path="abbreviation.csv")
    try:
        _ = svc.analyze("ok")
    except:
        pass
    return svc

def get_paginated_history(page=1, per_page=5):
    offset = (page - 1) * per_page
    total_count = get_total_count()
    total_pages = math.ceil(total_count / per_page) if total_count > 0 else 1
    rows = get_by_page(per_page, offset)
    return rows, total_pages, total_count


def main():
    st.markdown('<div class="title">Vietnamese Sentiment Analysis</div>', unsafe_allow_html=True)
    
    svc = _service()

    # PHÂN LOẠI CẢM XÚC
    st.markdown('<div class="block">', unsafe_allow_html=True)
    st.subheader("Phân loại cảm xúc")

    with st.form("sentiment_form"):
        text = st.text_area(
            "Nhập văn bản(không được bỏ trống hoặc viết thiếu):",
            height=120,
            placeholder="VD: Hôm nay tôi rất vui"
        )
        submitted = st.form_submit_button("Phân tích")

        if submitted:
            if not text or not text.strip():
                st.warning("Câu không hợp lệ,xin hãy thử lại.")
            elif len(text.strip()) < 5:
                st.error("Câu quá ngắn! (≥ 5 ký tự)")
            else:
                try:
                    res = svc.analyze(text)
                    sent = res["sentiment"]

                    if sent == "INVALID":
                        st.warning("Câu không hợp lệ, xin hãy thử lại.")
                    else:
                        # Hiển thị cảm xúc dạng thẻ màu
                        if sent == "POSITIVE":
                            st.success("🟢 **Tích cực**")
                        elif sent == "NEGATIVE":
                            st.error("🔴 **Tiêu cực**")
                        else:
                            st.info("🟡 **Trung tính**")

                        # Văn bản đã xử lý
                        if res["text"] != text.strip():
                            st.write(f"Văn bản sau xử lý: *{res['text']}*")

                        insert_record(res["text"], sent)

                except Exception as e:
                    st.error("Lỗi xử lý, vui lòng thử lại.")
                    print("[Pipeline error]", e)

    st.markdown('</div>', unsafe_allow_html=True)

    #LỊCH SỬ 
    st.markdown('<div class="block">', unsafe_allow_html=True)

    st.subheader("Lịch sử phân tích gần đây")

    if "current_page" not in st.session_state:
        st.session_state.current_page = 1

    rows, total_pages, total_count = get_paginated_history(st.session_state.current_page, 5)

    if rows:
        st.write(f"Tổng cộng: **{total_count}** bản ghi")

        df_data = [
            {"ID": r[0], "Văn bản": r[1], "Cảm xúc": r[2], "Thời gian": r[3]}
            for r in rows
        ]

        st.dataframe(df_data, hide_index=True, width="stretch")

        # Pagination buttons
        cols = st.columns([1, 1, 2, 1, 1])

        with cols[0]:
            if st.button("⏮", disabled=st.session_state.current_page == 1):
                st.session_state.current_page = 1
                st.rerun()

        with cols[1]:
            if st.button("◀", disabled=st.session_state.current_page == 1):
                st.session_state.current_page -= 1
                st.rerun()

        with cols[2]:
            st.write(f"Trang **{st.session_state.current_page}** / **{total_pages}**")

        with cols[3]:
            if st.button("▶", disabled=st.session_state.current_page == total_pages):
                st.session_state.current_page += 1
                st.rerun()

        with cols[4]:
            if st.button("⏭", disabled=st.session_state.current_page == total_pages):
                st.session_state.current_page = total_pages
                st.rerun()

    else:
        st.info("Chưa có lịch sử phân tích.")

    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
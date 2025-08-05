import json
import streamlit as st
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from PIL import Image
import os
import torch
import re
from difflib import SequenceMatcher

# Xác định thiết bị (CPU hoặc GPU nếu có)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tải dữ liệu tuyển sinh
try:
    with open('admissions_data.json', 'r', encoding='utf-8') as f:
        admissions_data = json.load(f)
except FileNotFoundError:
    st.error("Không tìm thấy file admissions_data.json. Vui lòng kiểm tra lại.")
    admissions_data = {"questions": []}

# Khởi tạo mô hình NLP từ Hugging Face
try:
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased')
    # Chuyển mô hình sang thiết bị ngay sau khi tải
    model.to(device)
    # Khởi tạo pipeline với thiết bị đã xác định
    nlp_model = pipeline('text-classification', model=model, tokenizer=tokenizer, device=device if torch.cuda.is_available() else -1)
except Exception as e:
    st.error(f"Lỗi khởi tạo mô hình: {e}")
    nlp_model = None

# Hàm tính độ tương đồng giữa hai chuỗi
def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

# Hàm tìm câu trả lời và hình ảnh tương ứng
def find_answer_and_images(question):
    if not isinstance(question, str):
        question = str(question)

    # Chuẩn hóa câu hỏi: loại bỏ khoảng trắng thừa và chuyển về chữ thường
    question = re.sub(r'\s+', ' ', question.strip().lower())

    best_match = None
    best_score = 0.0

    for item in admissions_data['questions']:
        questions = item['question'] if isinstance(item['question'], list) else [item['question']]

        for q in questions:
            if isinstance(q, str):
                normalized_q = re.sub(r'\s+', ' ', q.strip().lower())
                # Tính độ tương đồng
                score = similarity(question, normalized_q)
                # Kiểm tra khớp chính xác hoặc độ tương đồng cao
                if normalized_q == question or score > best_score:
                    best_score = score
                    best_match = item

    if best_match and best_score >= 0.7:  # Ngưỡng độ tương đồng tối thiểu
        images = best_match.get('images', [])
        captions = best_match.get('captions', [])

        # Xử lý trường hợp images là chuỗi đơn
        if isinstance(images, str):
            images = [images] if os.path.exists(images) else []
            captions = [captions] if isinstance(captions, str) else captions or []
        elif not isinstance(images, list):
            images = []
            captions = []

        # Đảm bảo captions là danh sách và khớp với số lượng ảnh
        if isinstance(captions, str):
            captions = [captions]
        elif captions is None or len(captions) == 0:
            captions = [f"Ảnh {i + 1}" for i in range(len(images))] if images else []

        # Xác thực đường dẫn hình ảnh
        valid_images = [img for img in images if isinstance(img, str) and os.path.exists(img)]
        valid_captions = captions[:len(valid_images)] if len(captions) >= len(valid_images) else captions + [f"Ảnh {i + 1}" for i in range(len(valid_images) - len(captions))]

        return best_match.get('answer', "Không có câu trả lời."), valid_images, valid_captions

    return "Tôi không có thông tin chính xác về câu hỏi này. Vui lòng thử lại hoặc liên hệ văn phòng tuyển sinh.", [], []

# Hàm xử lý câu hỏi và hiển thị hình ảnh
def chatbot_response(user_input):
    if not nlp_model:
        return "Mô hình không khả dụng. Vui lòng kiểm tra kết nối internet hoặc cài đặt lại thư viện.", [], []

    answer, image_paths, captions = find_answer_and_images(user_input)

    # Tải hình ảnh từ đường dẫn
    images = []
    for path in image_paths:
        try:
            if os.path.exists(path):
                images.append(Image.open(path))
            else:
                images.append(None)
        except Exception as e:
            st.warning(f"Không thể tải ảnh {path}: {e}")
            images.append(None)

    return answer, images, captions

# Giao diện Streamlit
def main():
    st.title("Chatbot Tư vấn Tuyển sinh")
    st.markdown("Hỏi về thông tin tuyển sinh và xem hình ảnh liên quan!")

    # Khởi tạo lịch sử trò chuyện
    if 'history' not in st.session_state:
        st.session_state.history = []

    # Ô nhập câu hỏi
    user_input = st.text_input("Câu hỏi của bạn:", key="question_input")

    # Nút gửi câu hỏi
    if st.button("Gửi", key="submit_button"):
        if user_input:
            response, images, captions = chatbot_response(user_input)
            st.text_area("Câu trả lời:", value=response, height=200, key="response_area")

            # Hiển thị hình ảnh
            valid_images = [img for img in images if img is not None]
            if valid_images:
                st.subheader("Ảnh liên quan:")
                num_cols = min(len(valid_images), 3)
                if num_cols > 0:
                    cols = st.columns(num_cols)
                    col_idx = 0
                    for i, img in enumerate(images):
                        if img is not None:
                            with cols[col_idx]:
                                st.image(img, caption=captions[i] if i < len(captions) else f"Ảnh {i + 1}",
                                         use_container_width=True)
                            col_idx = (col_idx + 1) % num_cols
            else:
                st.warning("Không có ảnh liên quan hoặc ảnh không thể tải.")

    # Nút lưu lịch sử
    if user_input and st.button("Lưu lịch sử", key="save_history_button"):
        response, images, captions = chatbot_response(user_input)
        # Kiểm tra để tránh lưu trùng
        if not st.session_state.history or st.session_state.history[-1][0] != user_input:
            st.session_state.history.append((user_input, response, images, captions))

    # Nút xóa lịch sử
    if st.button("Xóa lịch sử", key="clear_history_button"):
        st.session_state.history = []
        st.success("Đã xóa lịch sử trò chuyện.")

    # Hiển thị lịch sử trò chuyện
    st.subheader("Lịch sử trò chuyện (5 lần gần nhất):")
    if st.session_state.history:
        for q, r, imgs, caps in st.session_state.history[-5:]:
            st.write(f"**Hỏi**: {q}\n**Trả lời**: {r}")
            valid_images = [img for img in imgs if img is not None]
            if valid_images:
                num_cols = min(len(valid_images), 3)
                if num_cols > 0:
                    cols = st.columns(num_cols)
                    col_idx = 0
                    for i, img in enumerate(imgs):
                        if img is not None:
                            with cols[col_idx]:
                                st.image(img, caption=caps[i] if i < len(caps) else f"Ảnh {i + 1}",
                                         use_container_width=True)
                            col_idx = (col_idx + 1) % num_cols
    else:
        st.info("Chưa có lịch sử trò chuyện.")

if __name__ == "__main__":
    main()
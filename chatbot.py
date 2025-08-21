# Copyright (c) [2025] [Nguyễn Minh Tấn Phúc]. Bảo lưu mọi quyền.
# Nguồn: https://tlmchattest.streamlit.app/
import json
import streamlit as st
import os
import re
from PIL import Image
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
from pyvi import ViTokenizer
import unicodedata
from underthesea import word_tokenize


# --- CÁC HÀM TIỆN ÍCH ---

# Hàm loại bỏ từ dừng tiếng Việt
def remove_vietnamese_stopwords(tokenized_text):
    # Mở rộng danh sách từ dừng để xử lý các từ thường đi kèm tên trường
    stopwords = [
        'là', 'và', 'có', 'của', 'trong', 'được', 'cho', 'với', 'tại', 'từ',
        'bởi', 'để', 'như', 'thì', 'mà', 'này', 'kia', 'đó', 'nào', 'cái',
        'những', 'một', 'các', 'đã', 'lại', 'còn', 'nếu', 'vì', 'do', 'bị',
        'về', 'trường', 'thpt', 'ten', 'lơ', 'man', 'ernst', 'thalmann'
    ]
    tokens = tokenized_text.split() if isinstance(tokenized_text, str) else tokenized_text
    return [token for token in tokens if token not in stopwords]


# Hàm chuẩn hóa văn bản
def normalize_text(text):
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text


# Hàm tách từ dính
def split_sticky_words(text):
    return word_tokenize(text, format="text")


# --- CẤU HÌNH VÀ TẢI DỮ LIỆU ---
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    with open(os.path.join(os.path.dirname(__file__), 'admissions_data.json'), 'r', encoding='utf-8') as f:
        admissions_data = json.load(f)
except Exception as e:
    st.error(f"Lỗi khi tải admissions_data.json: {e}")
    admissions_data = {"questions": []}

try:
    @st.cache_resource
    def load_sentence_transformer_model():
        return SentenceTransformer('bkai-foundation-models/vietnamese-bi-encoder', device=device)


    model = load_sentence_transformer_model()
except Exception as e:
    st.error(f"Lỗi khi tải mô hình SentenceTransformer: {e}")
    model = None

# Mã hóa các câu hỏi từ admissions_data.json để tìm kiếm
if model and 'question_embeddings' not in st.session_state:
    st.session_state.question_texts = []
    st.session_state.question_data_map = {}
    for item in admissions_data.get('questions', []):
        questions = item.get('question', [])
        if not isinstance(questions, list):
            questions = [questions]

        for q in questions:
            if not isinstance(q, str) or not q.strip():
                continue

            # Quy trình xử lý tương tự như câu hỏi của người dùng
            norm_q = normalize_text(q)
            split_q = split_sticky_words(norm_q)
            tokenized_q = ViTokenizer.tokenize(split_q)
            # Không xóa từ dừng ở đây để giữ ngữ nghĩa gốc
            st.session_state.question_texts.append(tokenized_q)
            st.session_state.question_data_map[tokenized_q] = item

    if st.session_state.question_texts:
        st.session_state.question_embeddings = model.encode(st.session_state.question_texts, convert_to_tensor=True)
    else:
        st.session_state.question_embeddings = None


# --- HÀM TÌM KIẾM CÂU TRẢ LỜI ---
def find_answer_and_media(question):
    if not model or st.session_state.question_embeddings is None:
        return "Chatbot đang gặp sự cố, vui lòng thử lại sau.", "text", None

    # 1. Chuẩn hóa và làm sạch câu hỏi của người dùng
    norm_question = normalize_text(question)

    # Mẫu regex để nhận diện và loại bỏ tên trường
    school_patterns = re.compile(
        r'\b(của\s+)?(trường\s+)?(thpt\s+)?(ten\s+lơ\s+man|ernst\s+thälmann|ernst\s+thalmann)\b',
        re.IGNORECASE
    )
    # Loại bỏ tên trường khỏi câu hỏi
    core_question = school_patterns.sub('', norm_question).strip()

    # Nếu câu hỏi chỉ chứa tên trường, trả về lời chào chung
    if not core_question or core_question in ['về', 'của']:
        return admissions_data['questions'][22]['answer'], "image", (admissions_data['questions'][22]['images'],
                                                                     admissions_data['questions'][22]['captions'])

    # Xử lý phần lõi của câu hỏi
    split_question = split_sticky_words(core_question)
    tokenized_question = ViTokenizer.tokenize(split_question)

    # Loại bỏ từ dừng sau khi đã xử lý tên trường
    clean_question = ' '.join(remove_vietnamese_stopwords(tokenized_question.split()))

    if not clean_question.strip():
        return "Câu hỏi của bạn chưa rõ ràng, vui lòng cung cấp thêm chi tiết.", "text", None

    # 2. Tìm kiếm ngữ nghĩa (Semantic Search)
    query_embedding = model.encode(clean_question, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(query_embedding, st.session_state.question_embeddings)[0]

    # 3. Lấy top 5 kết quả có điểm cao nhất
    top_k = min(5, len(st.session_state.question_texts))
    top_results = torch.topk(cosine_scores, k=top_k)

    best_score = top_results[0][0].item()
    best_index = top_results[1][0].item()

    # 4. Kiểm tra ngưỡng điểm
    if best_score < 0.45:  # Tăng ngưỡng để kết quả chính xác hơn
        return "Xin lỗi, tôi không tìm thấy thông tin phù hợp. Bạn có thể thử hỏi bằng cách khác được không?", "text", None

    # 5. Lấy câu trả lời và media tương ứng
    best_match_text = st.session_state.question_texts[best_index]
    best_match_data = st.session_state.question_data_map[best_match_text]

    answer_text = best_match_data.get('answer', "Không có câu trả lời.")
    images = best_match_data.get('images')
    captions = best_match_data.get('captions')
    video_url = best_match_data.get('video_url')

    # Xử lý định dạng media
    if images and isinstance(images, str):
        images = [images]

    if video_url:
        return answer_text, "video", video_url
    if images:
        return answer_text, "image", (images, captions)

    return answer_text, "text", None


# --- GIAO DIỆN STREAMLIT (Không thay đổi) ---
def main():
    st.title("Chatbot Tư vấn Tuyển sinh")
    st.markdown("Hỏi về thông tin tuyển sinh và xem hình ảnh hoặc video liên quan!")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if "text" in message:
                st.markdown(message["text"])
            if "video" in message:
                st.video(message["video"])
            if "images" in message:
                valid_images_paths = [img_path for img_path in message["images"] if
                                      isinstance(img_path, str) and os.path.exists(img_path) and img_path.strip() != ""]
                if valid_images_paths:
                    num_cols = min(len(valid_images_paths), 3)
                    cols = st.columns(num_cols)
                    captions = message.get("captions", [])
                    if not isinstance(captions, list):
                        captions = [captions] if captions else [f"Ảnh {i + 1}" for i in range(len(valid_images_paths))]
                    captions = captions[:len(valid_images_paths)]
                    for i, img_path in enumerate(valid_images_paths):
                        with cols[i % num_cols]:
                            st.image(img_path,
                                     caption=captions[i] if i < len(captions) else f"Ảnh {i + 1}",
                                     use_container_width=True)

    if prompt := st.chat_input("Câu hỏi của bạn:"):
        st.session_state.messages.append({"role": "user", "text": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        response, media_type, media_content = find_answer_and_media(prompt)

        with st.chat_message("assistant"):
            if media_type == "video":
                st.markdown(response)
                st.video(media_content)
                st.session_state.messages.append({"role": "assistant", "text": response, "video": media_content})
            elif media_type == "image":
                st.markdown(response)
                images, captions = media_content
                if images:
                    valid_images_paths = [img_path for img_path in images if
                                          isinstance(img_path, str) and os.path.exists(
                                              img_path) and img_path.strip() != ""]
                    if valid_images_paths:
                        num_cols = min(len(valid_images_paths), 3)
                        cols = st.columns(num_cols)
                        if not isinstance(captions, list):
                            captions = [captions] if captions else [f"Ảnh {i + 1}" for i in
                                                                    range(len(valid_images_paths))]
                        captions = captions[:len(valid_images_paths)]
                        for i, img_path in enumerate(valid_images_paths):
                            with cols[i % num_cols]:
                                st.image(img_path, caption=captions[i] if i < len(captions) else f"Ảnh {i + 1}",
                                         use_container_width=True)
                st.session_state.messages.append(
                    {"role": "assistant", "text": response, "images": images, "captions": captions})
            else:
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "text": response})

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Giới thiệu về trường", key="suggested_question_button"):
            hardcoded_response = "Tôi xin giới thiệu bạn video về trường."
            hardcoded_video_url = "https://www.youtube.com/watch?v=HzvZVAvBkto"
            st.session_state.messages.append({"role": "user", "text": "Giới thiệu về trường"})
            with st.chat_message("assistant"):
                st.markdown(hardcoded_response)
                st.video(hardcoded_video_url)
            st.session_state.messages.append(
                {"role": "assistant", "text": hardcoded_response, "video": hardcoded_video_url})
            st.rerun()

    with col2:
        if st.button("Xóa lịch sử trò chuyện", key="clear_history_button"):
            st.session_state.messages = []
            st.rerun()


# Thêm câu lệnh import cần thiết cho PyTorch
from sentence_transformers import util

if __name__ == "__main__":
    main()
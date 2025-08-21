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


# Hàm loại bỏ từ dừng tiếng Việt
def remove_vietnamese_stopwords(tokenized_text):
    stopwords = [
        'là', 'và', 'có', 'của', 'trong', 'được', 'cho', 'với', 'tại', 'từ',
        'bởi', 'để', 'như', 'thì', 'mà', 'này', 'kia', 'đó', 'nào', 'cái',
        'những', 'một', 'các', 'đã', 'lại', 'còn', 'nếu', 'vì', 'do', 'bị'
    ]
    tokens = tokenized_text.split() if isinstance(tokenized_text, str) else tokenized_text
    return [token for token in tokens if token not in stopwords]


# Hàm normalize: chuyển không dấu, lowercase, loại bỏ dấu thừa
def normalize_text(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text


# Hàm tách từ dính
def split_sticky_words(text):
    text = text.lower().replace(' ', '')
    tokenized = word_tokenize(text, format="text")
    return tokenized


# --- Cấu hình và tải dữ liệu ---
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    with open(os.path.join(os.path.dirname(__file__), 'admissions_data.json'), 'r', encoding='utf-8') as f:
        admissions_data = json.load(f)
        if not isinstance(admissions_data, dict) or 'questions' not in admissions_data:
            raise ValueError("admissions_data.json must be a dictionary with a 'questions' key")
        if not isinstance(admissions_data['questions'], list):
            raise ValueError("'questions' in admissions_data.json must be a list")
except FileNotFoundError:
    st.error("Không tìm thấy file admissions_data.json. Vui lòng kiểm tra lại.")
    admissions_data = {"questions": []}
except ValueError as e:
    st.error(f"Lỗi trong admissions_data.json: {e}")
    admissions_data = {"questions": []}
except Exception as e:
    st.error(f"Lỗi khi tải admissions_data.json: {e}")
    admissions_data = {"questions": []}

try:
    @st.cache_resource
    def load_sentence_transformer_model():
        model = SentenceTransformer('bkai-foundation-models/vietnamese-bi-encoder', device=device)
        return model


    model = load_sentence_transformer_model()
except Exception as e:
    st.error(f"Lỗi khi tải mô hình SentenceTransformer: {e}")
    model = None

if model and 'question_embeddings' not in st.session_state:
    st.session_state.question_texts = []
    st.session_state.question_data_map = {}
    for item in admissions_data['questions']:
        if not isinstance(item, dict) or 'question' not in item:
            print(f"Warning: Skipping invalid item in admissions_data['questions']: {item}")
            continue
        questions = item['question'] if isinstance(item['question'], list) else [item['question']]
        for q in questions:
            if not isinstance(q, str) or not q.strip():
                print(f"Warning: Skipping invalid question: {q}")
                continue

            # --- THAY ĐỔI 1: XỬ LÝ CÂU HỎI TỪ CƠ SỞ DỮ LIỆU ---
            norm_q = normalize_text(q)
            # Thêm dòng này để loại bỏ các đuôi câu không cần thiết, đồng bộ với cách xử lý input
            norm_q = re.sub(r'\s+(cua|ve)\s+truong$', '', norm_q).strip()

            split_q = split_sticky_words(norm_q)
            tokenized_q = ViTokenizer.tokenize(split_q)
            clean_q = ' '.join(remove_vietnamese_stopwords(tokenized_q)) if remove_vietnamese_stopwords(
                tokenized_q) else tokenized_q
            st.session_state.question_texts.append(clean_q)
            st.session_state.question_data_map[clean_q] = item

    if st.session_state.question_texts:
        st.session_state.question_embeddings = model.encode(st.session_state.question_texts)
    else:
        st.session_state.question_embeddings = None
        print("Warning: No valid questions found in admissions_data['questions'].")


# --- Hàm tìm câu trả lời, hình ảnh và video ---
def find_answer_and_media(question):
    if not model or st.session_state.question_embeddings is None:
        return "Chatbot không thể xử lý vì không có dữ liệu câu hỏi hoặc mô hình ngôn ngữ gặp sự cố.", "text", None

    if not isinstance(question, str):
        question = str(question)
    question = re.sub(r'\s+', ' ', question.strip().lower())
    if any(keyword in question for keyword in ['thời tiết', 'tin tức', 'giá cả', 'bóng đá']):
        return "Xin lỗi, tôi chỉ hỗ trợ thông tin về tuyển sinh. Vui lòng hỏi về học phí, học bổng, hoặc thông tin trường!", "text", None

    question_to_process = re.sub(r'(tôi muốn biết|tìm hiểu|giới thiệu|thông tin|hỏi|biết)\s*(về)?\s*', '',
                                 question).strip()
    norm_question = normalize_text(question_to_process)

    school_patterns = re.compile(
        r'\b(truong\s+)?thpt\s+ten\s+lo\s+man\b|\btrung\s+hoc\s+pho\s+thong\s+ten\s+lo\s+man\b|\bthpt\s+ernst\s+thalmann\b|\bten\s+lo\s+man\b|\bernst\s+thalmann\b|\bersnt\s+thalmann\b',
        re.IGNORECASE)
    school_name_match = school_patterns.search(question)
    school_name = school_name_match.group(0).strip() if school_name_match else None

    # --- THAY ĐỔI 2: XỬ LÝ CÂU HỎI TỪ NGƯỜI DÙNG ---
    # Logic mới chỉ loại bỏ phần tên trường cụ thể, giữ lại các từ khóa chung
    clean_norm_question = school_patterns.sub('', norm_question).strip()
    clean_norm_question = re.sub(r'\s+', ' ', clean_norm_question)

    if not clean_norm_question:
        return "Xin lỗi, câu hỏi của bạn quá chung chung. Vui lòng cung cấp thêm chi tiết!", "text", None

    split_question = split_sticky_words(clean_norm_question)
    tokenized_question = ViTokenizer.tokenize(split_question)
    clean_question = ' '.join(remove_vietnamese_stopwords(tokenized_question)) if remove_vietnamese_stopwords(
        tokenized_question) else tokenized_question
    print(f"Cleaned query: {clean_question}")

    query_embedding = model.encode([clean_question])
    cosine_scores = cosine_similarity(query_embedding, st.session_state.question_embeddings)[0]
    print(f"Cosine scores: {cosine_scores}")
    best_score = np.max(cosine_scores)
    best_index = np.argmax(cosine_scores)

    if best_score < 0.6:
        return "Xin lỗi, không tìm thấy thông tin phù hợp. Vui lòng kiểm tra lại từ khóa!", "text", None

    best_match_text = st.session_state.question_texts[best_index]
    best_match = st.session_state.question_data_map[best_match_text]

    if "images" in best_match and isinstance(best_match["images"], str):
        best_match["images"] = [best_match["images"]]

    has_images = "images" in best_match and best_match["images"]
    has_video = "video_url" in best_match and best_match["video_url"]

    answer_text = best_match.get('answer', "Không có câu trả lời.")
    if school_name:
        answer_text = f"Thông tin về {school_name.title()}:\n\n" + answer_text

    if has_images and has_video:
        media_content = (best_match.get('images', []), best_match.get('captions', []), best_match.get('video_url'))
        return answer_text, "multimedia", media_content
    elif has_images:
        images = best_match.get('images', [])
        captions = best_match.get('captions', [])
        valid_images = [img for img in images if isinstance(img, str) and os.path.exists(img) and img.strip() != ""]
        valid_captions = captions[:len(valid_images)] if captions and len(captions) >= len(valid_images) else (
                                                                                                                          captions or []) + [
                                                                                                                  f"Ảnh {i + 1}"
                                                                                                                  for i
                                                                                                                  in
                                                                                                                  range(
                                                                                                                      len(valid_images) - len(
                                                                                                                          captions or []))]
        return answer_text, "image", (valid_images, valid_captions)
    else:
        return answer_text, "text", None


# --- Giao diện Streamlit ---
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
            if media_type == "multimedia":
                images, captions, video_url = media_content
                st.markdown(response)
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
                st.video(video_url)
                st.session_state.messages.append(
                    {"role": "assistant", "text": response, "images": images, "captions": captions, "video": video_url})
            elif media_type == "video":
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


if __name__ == "__main__":
    main()
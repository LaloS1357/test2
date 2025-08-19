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
import google.generativeai as genai

# Debug: In các khóa trong st.secrets
try:
    print("Available secrets:", dict(st.secrets))
except Exception as e:
    print("Error accessing secrets:", str(e))

# Khởi tạo mô hình Gemini
try:
    genai.configure(api_key=st.secrets["genai_api_key"])
    gemini_model = genai.GenerativeModel('gemini-pro')
except KeyError:
    st.error("Không tìm thấy khóa 'genai_api_key' trong secrets.toml. Vui lòng kiểm tra file C:\\Users\\dever\\Downloads\\F\\test2\\.streamlit\\secrets.toml.")
    api_key = os.getenv("GENAI_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel('gemini-pro')
    else:
        st.error("Không tìm thấy biến môi trường GENAI_API_KEY. Vui lòng đặt biến hoặc sửa file secrets.toml.")
        gemini_model = None

def get_answer_from_gemini(user_query):
    if gemini_model is None:
        return "Lỗi: Không thể khởi tạo mô hình Gemini do thiếu API key."
    try:
        response = gemini_model.generate_content(user_query)
        return response.text
    except Exception as e:
        return f"Xin lỗi, tôi không thể trả lời câu hỏi này lúc này. Lỗi: {e}"

def get_best_answer(user_query, data_questions, data_answers):
    query_embedding = model.encode(user_query, convert_to_tensor=True)
    similarities = cosine_similarity(query_embedding.cpu().numpy().reshape(1, -1), embeddings.cpu().numpy())
    best_match_index = np.argmax(similarities)
    similarity_score = similarities[0][best_match_index]
    threshold = 0.8
    if similarity_score >= threshold:
        return data_answers[best_match_index], None
    else:
        gemini_response = get_answer_from_gemini(user_query)
        return gemini_response, None

# Phần xử lý khi người dùng nhập câu hỏi
if prompt := st.chat_input("Nhập câu hỏi của bạn"):
    st.session_state.messages.append({"role": "user", "text": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        answer, extra_data = get_best_answer(prompt, data_questions, data_answers)
        st.markdown(answer)
        if extra_data:
            pass
        st.session_state.messages.append({"role": "assistant", "text": answer})

# Hàm loại bỏ từ dừng tiếng Việt
def remove_vietnamese_stopwords(tokenized_text):
    stopwords = [
        'là', 'của', 'và', 'có', 'trong', 'được', 'cho', 'với', 'tại', 'từ',
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

# Cấu hình và tải dữ liệu
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
        model = SentenceTransformer('dangvantuan/vietnamese-embedding', device=device)
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
            norm_q = normalize_text(q)
            split_q = split_sticky_words(norm_q)
            tokenized_q = ViTokenizer.tokenize(split_q)
            clean_q = ' '.join(remove_vietnamese_stopwords(tokenized_q)) if remove_vietnamese_stopwords(tokenized_q) else tokenized_q
            st.session_state.question_texts.append(clean_q)
            st.session_state.question_data_map[clean_q] = item
    if st.session_state.question_texts:
        st.session_state.question_embeddings = model.encode(st.session_state.question_texts)
    else:
        st.session_state.question_embeddings = None
        print("Warning: No valid questions found in admissions_data['questions'].")

def find_answer_and_media(question):
    if not model or st.session_state.question_embeddings is None:
        return "Chatbot không thể xử lý vì không có dữ liệu câu hỏi hoặc mô hình ngôn ngữ gặp sự cố.", "text", None
    if not isinstance(question, str):
        question = str(question)
    question = re.sub(r'\s+', ' ', question.strip().lower())
    question = re.sub(r'(tôi muốn biết|tìm hiểu|giới thiệu|thông tin|hỏi|biết)\s*(về)?\s*', '', question).strip()
    norm_question = normalize_text(question)
    split_question = split_sticky_words(norm_question)
    tokenized_question = ViTokenizer.tokenize(split_question)
    clean_question = ' '.join(remove_vietnamese_stopwords(tokenized_question)) if remove_vietnamese_stopwords(tokenized_question) else tokenized_question
    best_match = None
    for item in admissions_data['questions']:
        questions = item['question'] if isinstance(item['question'], list) else [item['question']]
        for q in questions:
            if isinstance(q, str):
                norm_q = normalize_text(q)
                split_q = split_sticky_words(norm_q)
                tokenized_q = ViTokenizer.tokenize(split_q)
                clean_q = ' '.join(remove_vietnamese_stopwords(tokenized_q)) if remove_vietnamese_stopwords(tokenized_q) else tokenized_q
                if clean_question in clean_q or tokenized_question in tokenized_q:
                    best_match = item
                    break
        if best_match:
            break
    if best_match is None:
        query_embedding = model.encode([clean_question])
        cosine_scores = cosine_similarity(query_embedding, st.session_state.question_embeddings)[0]
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
    if has_images and has_video:
        images = best_match.get('images', [])
        captions = best_match.get('captions', [])
        video_url = best_match["video_url"]
        return answer_text, "multimedia", (images, captions, video_url)
    elif has_video:
        return answer_text, "video", best_match["video_url"]
    elif has_images:
        images = best_match.get('images', [])
        captions = best_match.get('captions', [])
        valid_images = [img for img in images if isinstance(img, str) and os.path.exists(img) and img.strip() != ""]
        if valid_images and captions is not None:
            valid_captions = captions[:len(valid_images)] if len(captions) >= len(valid_images) else captions + [
                f"Ảnh {i + 1}" for i in range(len(valid_images) - len(captions))]
        else:
            valid_captions = []
        return answer_text, "image", (valid_images, valid_captions)
    else:
        return answer_text, "text", None

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
            st.session_state.messages.append({
                "role": "assistant",
                "text": hardcoded_response,
                "video": hardcoded_video_url
            })
            st.rerun()
    with col2:
        if st.button("Xóa lịch sử trò chuyện", key="clear_history_button"):
            st.session_state.messages = []
            st.rerun()

if __name__ == "__main__":
    main()
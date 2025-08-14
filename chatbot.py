# Copyright (c) [2025] [Nguyễn Minh Tấn Phúc]. Bảo lưu mọi quyền.
# Nguồn: https://tlmchattest.streamlit.app/
import json
import streamlit as st
import os
import torch
import re
import numpy as np
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from pyvi.ViTokenizer import tokenize
from sklearn.metrics.pairwise import cosine_similarity

# Xác định thiết bị
device = "cuda" if torch.cuda.is_available() else "cpu"

# Tải dữ liệu tuyển sinh
try:
    with open(os.path.join(os.path.dirname(__file__), 'admissions_data.json'), 'r', encoding='utf-8') as f:
        admissions_data = json.load(f)
        # Validate JSON structure
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

# Tải mô hình và tokenizer
try:
    @st.cache_resource
    def load_model_and_tokenizer():
        tokenizer = AutoTokenizer.from_pretrained("VoVanPhuc/sup-SimCSE-VietNamese-phobert-base")
        model = AutoModel.from_pretrained("VoVanPhuc/sup-SimCSE-VietNamese-phobert-base").to(device)
        return tokenizer, model

    tokenizer, model = load_model_and_tokenizer()
except Exception as e:
    st.error(f"Lỗi khi tải mô hình hoặc tokenizer: {e}")
    tokenizer, model = None, None

# Chức năng tạo embedding
def get_embedding(text, tokenizer, model):
    if not model or not tokenizer:
        return None
    # Phân đoạn từ tiếng Việt
    text = tokenize(text)
    # Tokenize và chuyển thành tensor
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        # Lấy pooler_output (CLS token, phù hợp với SimCSE)
        embedding = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
    return embedding.cpu().numpy().squeeze()

# Tạo embedding cho tất cả các câu hỏi trong dữ liệu để tăng tốc độ
if model and tokenizer and 'question_embeddings' not in st.session_state:
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
            st.session_state.question_texts.append(q)
            st.session_state.question_data_map[q] = item

    if st.session_state.question_texts:
        st.session_state.question_embeddings = np.array(
            [get_embedding(q, tokenizer, model) for q in st.session_state.question_texts])
    else:
        st.session_state.question_embeddings = None
        print("Warning: No valid questions found in admissions_data['questions'].")

# Hàm tìm câu trả lời, hình ảnh và video
def find_answer_and_media(question):
    if not model or not tokenizer or st.session_state.question_embeddings is None:
        return "Chatbot không thể xử lý vì không có dữ liệu câu hỏi hoặc mô hình ngôn ngữ gặp sự cố.", "text", None

    if not isinstance(question, str):
        question = str(question)
    question = re.sub(r'\s+', ' ', question.strip().lower())

    # Tạo embedding cho câu hỏi của người dùng
    query_embedding = get_embedding(question, tokenizer, model)

    # Tính độ tương đồng cosine giữa câu hỏi người dùng và tất cả các câu hỏi đã có
    cosine_scores = cosine_similarity([query_embedding], st.session_state.question_embeddings)[0]

    best_score = np.max(cosine_scores)
    best_index = np.argmax(cosine_scores)

    best_match_text = st.session_state.question_texts[best_index]
    best_match = st.session_state.question_data_map[best_match_text]

    # Ngưỡng độ tương đồng được điều chỉnh cho cosine similarity
    if best_score >= 0.8:  # Tăng ngưỡng để giảm nhầm lẫn
        if "images" in best_match and isinstance(best_match["images"], str):
            best_match["images"] = [best_match["images"]]

        has_images = "images" in best_match and best_match["images"]
        has_video = "video_url" in best_match and best_match["video_url"]

        if has_images and has_video:
            images = best_match.get('images', [])
            captions = best_match.get('captions', [])
            video_url = best_match["video_url"]
            return best_match.get('answer', "Đây là nội dung bạn yêu cầu."), "multimedia", (images, captions, video_url)
        elif has_video:
            return best_match.get('answer', "Đây là video bạn yêu cầu."), "video", best_match["video_url"]
        elif has_images:
            images = best_match.get('images', [])
            captions = best_match.get('captions', [])
            valid_images = [img_path for img_path in images if isinstance(img_path, str) and os.path.exists(img_path) and img_path.strip() != ""]
            if valid_images and captions is not None:
                valid_captions = captions[:len(valid_images)] if len(captions) >= len(valid_images) else captions + [
                    f"Ảnh {i + 1}" for i in range(len(valid_images) - len(captions))]
            else:
                valid_captions = []  # Gán giá trị mặc định nếu có lỗi
            return best_match.get('answer', "Không có câu trả lời."), "image", (valid_images, valid_captions)
        else:
            return best_match.get('answer', "Không có câu trả lời."), "text", None

    return "Xin lỗi, không tìm thấy thông tin phù hợp. Vui lòng kiểm tra lại từ khóa!", "text", None

# Giao diện Streamlit
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
                                          isinstance(img_path, str) and os.path.exists(img_path) and img_path.strip() != ""]
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
                                          isinstance(img_path, str) and os.path.exists(img_path) and img_path.strip() != ""]
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
        if st.button("Gợi ý: Giới thiệu về trường", key="suggested_question_button"):
            suggested_prompt = "Giới thiệu về trường"
            st.session_state.messages.append({"role": "user", "text": suggested_prompt})

            response, media_type, media_content = find_answer_and_media(suggested_prompt)
            if media_type == "multimedia":
                images, captions, video_url = media_content
                st.session_state.messages.append(
                    {"role": "assistant", "text": response, "images": images, "captions": captions, "video": video_url})
            elif media_type == "video":
                st.session_state.messages.append({"role": "assistant", "text": response, "video": media_content})
            elif media_type == "image":
                images, captions = media_content
                st.session_state.messages.append(
                    {"role": "assistant", "text": response, "images": images, "captions": captions})
            else:
                st.session_state.messages.append({"role": "assistant", "text": response})

            st.rerun()

    with col2:
        if st.button("Xóa lịch sử trò chuyện", key="clear_history_button"):
            st.session_state.messages = []

if __name__ == "__main__":
    main()
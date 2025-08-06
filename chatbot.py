import json
import streamlit as st
import os
import torch
import re
from difflib import SequenceMatcher
from PIL import Image

# Xác định thiết bị
device = 0 if torch.cuda.is_available() else -1

# Tải dữ liệu tuyển sinh
try:
    with open(os.path.join(os.path.dirname(__file__), 'admissions_data.json'), 'r', encoding='utf-8') as f:
        admissions_data = json.load(f)
except FileNotFoundError:
    st.error("Không tìm thấy file admissions_data.json. Vui lòng kiểm tra lại.")
    admissions_data = {"questions": []}


# Hàm tính độ tương đồng
def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


# Hàm tìm câu trả lời, hình ảnh và video
def find_answer_and_media(question):
    if not isinstance(question, str):
        question = str(question)
    question = re.sub(r'\s+', ' ', question.strip().lower())

    best_match = None
    best_score = 0.0
    for item in admissions_data['questions']:
        questions = item['question'] if isinstance(item['question'], list) else [item['question']]
        for q in questions:
            if isinstance(q, str):
                normalized_q = re.sub(r'\s+', ' ', q.strip().lower())
                score = similarity(question, normalized_q)
                if normalized_q == question or score > best_score:
                    best_score = score
                    best_match = item

    if best_match and best_score >= 0.5:
        # Chuyển đổi 'images' thành danh sách nếu nó là một chuỗi
        if "images" in best_match and isinstance(best_match["images"], str):
            best_match["images"] = [best_match["images"]]

        # Kiểm tra cả hình ảnh và video
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
            valid_images = [img for img in images if isinstance(img, str) and os.path.exists(img) and img.strip() != ""]
            valid_captions = captions[:len(valid_images)] if len(captions) >= len(valid_images) else captions + [
                f"Ảnh {i + 1}" for i in range(len(valid_images) - len(captions))]
            return best_match.get('answer', "Không có câu trả lời."), "image", (valid_images, valid_captions)
        else:  # Thêm khối else này để xử lý câu trả lời chỉ có văn bản
            return best_match.get('answer', "Không có câu trả lời."), "text", None

    return "Tôi không có thông tin chính xác về câu hỏi này. Vui lòng thử lại hoặc liên hệ văn phòng tuyển sinh.", "text", None


# Giao diện Streamlit
def main():
    st.title("Chatbot Tư vấn Tuyển sinh")
    st.markdown("Hỏi về thông tin tuyển sinh và xem hình ảnh hoặc video liên quan!")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Hiển thị lịch sử tin nhắn
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
                    for i, img_path in enumerate(valid_images_paths):
                        with cols[i % num_cols]:
                            st.image(img_path,
                                     caption=message["captions"][i] if i < len(message["captions"]) else f"Ảnh {i + 1}",
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

                # Hiển thị hình ảnh
                if images:
                    valid_images_paths = [img_path for img_path in images if
                                          isinstance(img_path, str) and os.path.exists(
                                              img_path) and img_path.strip() != ""]
                    if valid_images_paths:
                        num_cols = min(len(valid_images_paths), 3)
                        cols = st.columns(num_cols)
                        for i, img_path in enumerate(valid_images_paths):
                            with cols[i % num_cols]:
                                st.image(img_path, caption=captions[i] if i < len(captions) else f"Ảnh {i + 1}",
                                         use_container_width=True)

                # Hiển thị video
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
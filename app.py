# Copyright (c) [2025] [Nguyễn Minh Tấn Phúc]. Bảo lưu mọi quyền.
# Nguồn: https://tlmchattest.streamlit.app/
import json
import streamlit as st
import os
import re
from sentence_transformers import SentenceTransformer, util
import torch
from pyvi import ViTokenizer
import unicodedata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import py_vncorenlp
import difflib


# --- CÁC HÀM TIỆN ÍCH ---

def remove_vietnamese_accents(text):
    """
    Hàm này nhận một chuỗi văn bản tiếng Việt và trả về chuỗi đó không có dấu.
    """
    return "".join(c for c in unicodedata.normalize('NFD', text)
                   if unicodedata.category(c) != 'Mn')


def remove_vietnamese_stopwords(tokenized_text):
    stopwords = [
        'là', 'và', 'có', 'của', 'trong', 'được', 'cho', 'với', 'tại', 'từ',
        'bởi', 'để', 'như', 'thì', 'mà', 'này', 'kia', 'đó', 'nào', 'cái',
        'những', 'một', 'các', 'đã', 'lại', 'còn', 'nếu', 'vì', 'do', 'bị',
        'về'
    ]
    tokens = tokenized_text.split() if isinstance(tokenized_text, str) else tokenized_text
    return [token for token in tokens if token not in stopwords]


def normalize_text(text):
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text


def get_answer(question):
    norm_question = normalize_text(question)
    SCHOOL_NAME_VARIANTS = [
        "trường thpt", "thpt", "trường trung học phổ thông", "trung học phổ thông",
        "ten lơ men", "ten lơ man", "ten-lơ-man", "ten-lơ-men", "trường cấp 3", "cấp 3", "cấp ba", "trường cấp ba",
        "ernst thälmann", "ernst thalmann", "trường công lập", "công lập", "tlm", "t.l.m", "t l m",
        "trường ten lơ man", "trường ten lơ men", "truong thpt", "truong trung hoc pho thong", "truong cap 3",
        "truong cap ba", "truong cong lap", "truong ten lo man", "truong ten lo men", "trường ernst",
        "trường ernst thälmann", "trường ernst thalmann", "ernst", "trường tlm", "trường t.l.m", "trường t l m",
        "school", "high school", "secondary school", "tenlo man", "tenlo men", "tenloman", "tenlomen",
        "trường tenlo man", "trường tenlo men", "trường tenloman", "trường tenlomen"
    ]
    school_pattern = r"(" + r"|".join(
        [re.escape(variant).replace(" ", r"\\s*") for variant in SCHOOL_NAME_VARIANTS]) + r")"
    if re.fullmatch(rf"(\s*{school_pattern}\s*)+", norm_question, flags=re.IGNORECASE):
        return get_school_info_answer()
    core_question = re.sub(school_pattern, "", norm_question, flags=re.IGNORECASE).strip()
    if not core_question or core_question in ["", "về", "của"]:
        return get_school_info_answer()
    tokens = core_question.split()
    tokens = [t for t in tokens if t not in ["về", "của"]]
    if not tokens:
        return get_school_info_answer()
    return find_answer(core_question)


def get_school_info_answer():
    for item in admissions_data.get('questions', []):
        questions = item.get('question', [])
        if isinstance(questions, str):
            questions = [questions]
        for q in questions:
            norm_q = normalize_text(q)
            if norm_q in ["khái quát", "sơ lược", "khái quát về trường", "sơ lược về trường",
                          "cho tôi thông tin sơ lược và khái quát về trường"]:
                return item.get('answer', "Thông tin về trường THPT Ten Lơ Man...")
    return "Thông tin về trường THPT Ten Lơ Man..."


def remove_school_name(question):
    pattern = r"(trường\s+thpt\s+ten\s+lơ\s+man|thpt\s+ten\s+lơ\s+man|ernst\s+thälmann|ernst\s+thalmann)"
    return re.sub(pattern, "", question, flags=re.IGNORECASE).strip()


def find_answer(core_question):
    return find_answer_and_media(core_question)[0]


def split_sticky_words(text):
    # Use VnCoreNLP for word segmentation
    segments = vncorenlp_model.word_segment(text)
    return ' '.join(segments)


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

# Tải và cache TF-IDF vectorizer
try:
    @st.cache_resource
    def load_tfidf_vectorizer():
        return TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}')


    tfidf_vectorizer = load_tfidf_vectorizer()
except Exception as e:
    st.error(f"Lỗi khi tải TfidfVectorizer: {e}")
    tfidf_vectorizer = None

# Initialize VnCoreNLP for word segmentation
@st.cache_resource
def load_vncorenlp_model():
    return py_vncorenlp.VnCoreNLP(save_dir=os.path.join(os.path.dirname(__file__), 'vncorenlp'))


vncorenlp_model = load_vncorenlp_model()

# Cập nhật quy trình mã hóa dữ liệu
if model and tfidf_vectorizer and 'question_embeddings' not in st.session_state:
    st.session_state.question_texts = []
    unaccented_questions_for_encoding = []
    st.session_state.question_data_map = {}

    for item in admissions_data.get('questions', []):
        questions = item.get('question', [])
        if not isinstance(questions, list):
            questions = [questions]
        for q in questions:
            if not isinstance(q, str) or not q.strip():
                continue
            norm_q = normalize_text(q)
            split_q = split_sticky_words(norm_q)
            tokenized_q = ViTokenizer.tokenize(split_q)
            unaccented_q = remove_vietnamese_accents(tokenized_q)
            unaccented_questions_for_encoding.append(unaccented_q)
            st.session_state.question_texts.append(tokenized_q)
            st.session_state.question_data_map[tokenized_q] = item

    if unaccented_questions_for_encoding:
        st.session_state.question_embeddings = model.encode(unaccented_questions_for_encoding, convert_to_tensor=True)
        st.session_state.tfidf_matrix = tfidf_vectorizer.fit_transform(unaccented_questions_for_encoding)
    else:
        st.session_state.question_embeddings = None
        st.session_state.tfidf_matrix = None


# --- HÀM TÌM KIẾM CÂU TRẢ LỜI (HYBRID SEARCH) ---
def fuzzy_match_question(user_question, admissions_data, min_ratio=0.7):
    """
    Fuzzy match user question against all question variants in admissions_data.
    Returns (answer, images, captions) if found, else None.
    """
    user_question_norm = normalize_text(user_question)
    best_ratio = 0
    best_item = None
    for item in admissions_data.get('questions', []):
        questions = item.get('question', [])
        if isinstance(questions, str):
            questions = [questions]
        for q in questions:
            q_norm = normalize_text(q)
            ratio = difflib.SequenceMatcher(None, user_question_norm, q_norm).ratio()
            if user_question_norm in q_norm or q_norm in user_question_norm:
                ratio += 0.2  # boost for substring
            if ratio > best_ratio:
                best_ratio = ratio
                best_item = item
    if best_item and best_ratio >= min_ratio:
        return best_item.get('answer'), best_item.get('images'), best_item.get('captions')
    return None


def find_answer_and_media(question):
    if not model or st.session_state.question_embeddings is None or st.session_state.tfidf_matrix is None:
        return "Chatbot đang gặp sự cố, vui lòng thử lại sau.", "text", None

    norm_question = normalize_text(question)
    school_pattern = r"(trường\s*thpt|thpt|trường\s*trung\s*học\s*phổ\s*thông|trung\s*học\s*phổ\s*thông|ten\s*lơ\s*men|ten\s*lơ\s*man|ten-lơ-man|ten-lơ-men|trường\s*cấp\s*3|cấp\s*3|cấp\s*ba|trường\s*cấp\s*ba|ernst\s*thälmann|ernst\s*thalmann|trường\s*công\s*lập|công\s*lập|tlm|t.l.m|t l m|trường\s*ten\s*lơ\s*man|trường\s*ten\s*lơ\s*men|truong\s*thpt|truong\s*trung\s*hoc\s*pho\s*thong|truong\s*cap\s*3|truong\s*cap\s*ba|truong\s*cong\s*lap|truong\s*ten\s*lo\s*man|truong\s*ten\s*lo\s*men|trường\s*ernst|trường\s*ernst\s*thälmann|trường\s*ernst\s*thalmann|ernst|trường\s*tlm|trường\s*t.l.m|trường\s*t l m|school|high\s*school|secondary\s*school|tenlo man|tenlo men|tenloman|tenlomen|trường\s*tenlo man|trường\s*tenlo men|trường\s*tenloman|trường\s*tenlomen)"
    # Only remove school name if the question is just the school name
    if re.fullmatch(rf"(\s*{school_pattern}\s*)+", norm_question, flags=re.IGNORECASE):
        return get_school_info_answer(), "text", None
    core_question = re.sub(school_pattern, "", norm_question, flags=re.IGNORECASE).strip()
    if not core_question or core_question in ['về', 'của']:
        return get_school_info_answer(), "text", None
    tokens = core_question.split()
    tokens = [t for t in tokens if t not in ["về", "của"]]
    if not tokens:
        return get_school_info_answer(), "text", None

    split_question = split_sticky_words(core_question)
    tokenized_question = ViTokenizer.tokenize(split_question)
    clean_question = ' '.join(remove_vietnamese_stopwords(tokenized_question.split()))
    unaccented_clean_question = remove_vietnamese_accents(clean_question)
    # Accept questions with at least 1 word
    if not unaccented_clean_question.strip() or len(unaccented_clean_question.split()) < 1:
        # Fuzzy match fallback for very short queries
        fuzzy_result = fuzzy_match_question(question, admissions_data)
        if fuzzy_result:
            answer, images, captions = fuzzy_result
            if images and isinstance(images, str):
                images = [images]
            if images:
                return answer, "image", (images, captions)
            return answer, "text", None
        return "Câu hỏi của bạn quá ngắn hoặc không rõ ràng, vui lòng cung cấp thêm chi tiết.", "text", None

    # Hybrid Search
    query_embedding = model.encode(unaccented_clean_question, convert_to_tensor=True)
    semantic_scores = util.pytorch_cos_sim(query_embedding, st.session_state.question_embeddings)[0]
    query_tfidf = tfidf_vectorizer.transform([unaccented_clean_question])
    lexical_scores = cosine_similarity(query_tfidf, st.session_state.tfidf_matrix)[0]
    alpha = 0.6  # Tăng trọng số semantic để ưu tiên ý nghĩa
    beta = 0.4  # Giảm trọng số lexical để bổ trợ
    hybrid_scores = alpha * semantic_scores.cpu().numpy() + beta * lexical_scores

    top_k = min(5, len(st.session_state.question_texts))
    top_indices = np.argsort(hybrid_scores)[-top_k:][::-1]
    best_index = top_indices[0]
    best_score = hybrid_scores[best_index]
    threshold = 0.45  # Ngưỡng tin cậy điều chỉnh để cân bằng độ chính xác

    def prettify(text):
        return text.replace("_", " ")

    # Nếu không vượt ngưỡng, gợi ý các câu hỏi liên quan
    if best_score < threshold:
        # Fuzzy match fallback for low confidence
        fuzzy_result = fuzzy_match_question(question, admissions_data)
        if fuzzy_result:
            answer, images, captions = fuzzy_result
            if images and isinstance(images, str):
                images = [images]
            if images:
                return answer, "image", (images, captions)
            return answer, "text", None
        related_questions = []
        for i in top_indices:
            score = hybrid_scores[i]
            if score > 0.2:
                related_questions.append(prettify(st.session_state.question_texts[i]))
        if related_questions:
            suggestions = '\n'.join([f"{i + 1}. {q}" for i, q in enumerate(related_questions[:3])])
            return f"Xin lỗi, tôi chưa tìm thấy thông tin phù hợp. Bạn có thể thử hỏi:\n{suggestions}", "text", None
        else:
            popular_questions = []
            for item in admissions_data.get('questions', [])[:3]:
                qs = item.get('question', [])
                if isinstance(qs, str):
                    popular_questions.append(prettify(qs))
                elif isinstance(qs, list) and qs:
                    popular_questions.append(prettify(qs[0]))
            if popular_questions:
                suggestions = '\n'.join([f"{i + 1}. {q}" for i, q in enumerate(popular_questions)])
                return f"Xin lỗi, tôi chưa tìm thấy thông tin phù hợp. Bạn có thể hỏi về:\n{suggestions}", "text", None
            return "Xin lỗi, tôi chưa tìm thấy thông tin phù hợp. Bạn có thể thử hỏi về tuyển sinh, học phí, hoạt động ngoại khóa, giáo viên, ...", "text", None

    best_match_text = st.session_state.question_texts[best_index]
    best_match_data = st.session_state.question_data_map[best_match_text]
    answer_text = best_match_data.get('answer', "Không có câu trả lời.")
    images = best_match_data.get('images')
    captions = best_match_data.get('captions')
    video_url = best_match_data.get('video_url')

    if images and isinstance(images, str):
        images = [images]
    if video_url:
        return answer_text, "video", video_url
    if images:
        return answer_text, "image", (images, captions)
    return answer_text, "text", None


# --- GIAO DIỆN STREAMLIT ---
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
                                     caption=captions[i] if i < len(captions) else f"Ảnh {i + 1}")

    if prompt := st.chat_input("Câu hỏi của bạn:"):
        st.session_state.messages.append({"role": "user", "text": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Đang xử lý câu hỏi..."):
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
                    valid_images_paths = []
                    for img_path in images:
                        if isinstance(img_path, str) and img_path.strip() != "":
                            abs_img_path = os.path.join(os.path.dirname(__file__), img_path)
                            if os.path.exists(abs_img_path):
                                valid_images_paths.append(abs_img_path)
                            else:
                                st.warning(f"Không tìm thấy hình ảnh: {img_path}")
                    if valid_images_paths:
                        num_cols = min(len(valid_images_paths), 3)
                        cols = st.columns(num_cols)
                        if not isinstance(captions, list):
                            captions = [captions] if captions else [f"Ảnh {i + 1}" for i in range(len(valid_images_paths))]
                        captions = captions[:len(valid_images_paths)]
                        for i, abs_img_path in enumerate(valid_images_paths):
                            with cols[i % num_cols]:
                                st.image(abs_img_path, caption=captions[i] if i < len(captions) else f"Ảnh {i + 1}")
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
# Copyright (c) [2025] [Nguyễn Minh Tấn Phúc]. Bảo lưu mọi quyền.
# Nguồn: https://tlmchattest.streamlit.app/
import json
import streamlit as st
import os
import re
from sentence_transformers import SentenceTransformer, util
import torch
from pyvi import ViTokenizer
import unicodedata  # Thêm thư viện unicodedata
from underthesea import word_tokenize


# --- CÁC HÀM TIỆN ÍCH ---

# *** BẮT ĐẦU THAY ĐỔI 1: Thêm hàm loại bỏ dấu ***
# Hàm loại bỏ dấu tiếng Việt
def remove_vietnamese_accents(text):
    """
    Hàm này nhận một chuỗi văn bản tiếng Việt và trả về chuỗi đó không có dấu.
    """
    return "".join(c for c in unicodedata.normalize('NFD', text)
                   if unicodedata.category(c) != 'Mn')


# *** KẾT THÚC THAY ĐỔI 1 ***


# Hàm loại bỏ từ dừng tiếng Việt
def remove_vietnamese_stopwords(tokenized_text):
    stopwords = [
        'là', 'và', 'có', 'của', 'trong', 'được', 'cho', 'với', 'tại', 'từ',
        'bởi', 'để', 'như', 'thì', 'mà', 'này', 'kia', 'đó', 'nào', 'cái',
        'những', 'một', 'các', 'đã', 'lại', 'còn', 'nếu', 'vì', 'do', 'bị',
        'về', 'trường', 'thpt', 'ten', 'lơ', 'man', 'men', 'ernst', 'thalmann'
    ]
    tokens = tokenized_text.split() if isinstance(tokenized_text, str) else tokenized_text
    return [token for token in tokens if token not in stopwords]


# Hàm chuẩn hóa văn bản
def normalize_text(text):
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text


# Python
# Use the existing normalize_text function
def get_answer(question):
    norm_question = normalize_text(question)
    # Danh sách các biến thể tên trường
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
    # Tạo regex từ danh sách biến thể
    school_pattern = r"(" + r"|".join([re.escape(variant).replace(" ", r"\\s*") for variant in SCHOOL_NAME_VARIANTS]) + r")"
    # Comprehensive school name/variant matching using regex
    # If question is just the school name/variant (possibly with extra spaces/casing)
    if re.fullmatch(rf"(\s*{school_pattern}\s*)+", norm_question, flags=re.IGNORECASE):
        return get_school_info_answer()
    core_question = re.sub(school_pattern, "", norm_question, flags=re.IGNORECASE).strip()
    # If nothing left after removing school name/variant, or only stopwords, return school info
    if not core_question or core_question in ["", "về", "của"]:
        return get_school_info_answer()
    tokens = core_question.split()
    tokens = [t for t in tokens if t not in ["về", "của"]]
    if not tokens:
        return get_school_info_answer()
    return find_answer(core_question)

# Implement stubs for missing functions
def get_school_info_answer():
    # Search for the khái quát/sơ lược group in admissions_data
    for item in admissions_data.get('questions', []):
        questions = item.get('question', [])
        if isinstance(questions, str):
            questions = [questions]
        for q in questions:
            norm_q = normalize_text(q)
            if norm_q in ["khái quát", "sơ lược", "khái quát về trường", "sơ lược về trường", "cho tôi thông tin sơ lược và khái quát về trường"]:
                return item.get('answer', "Thông tin về trường THPT Ten Lơ Man...")
    return "Thông tin về trường THPT Ten Lơ Man..."

def remove_school_name(question):
    # Remove school name from question
    pattern = r"(trường\s+thpt\s+ten\s+lơ\s+man|thpt\s+ten\s+lơ\s+man|ernst\s+thälmann|ernst\s+thalmann)"
    return re.sub(pattern, "", question, flags=re.IGNORECASE).strip()

def find_answer(core_question):
    # Use your existing find_answer_and_media or similar logic
    return find_answer_and_media(core_question)[0]
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

# *** BẮT ĐẦU THAY ĐỔI 2: Cập nhật quy trình mã hóa dữ liệu ***
# Mã hóa phiên bản KHÔNG DẤU của các câu hỏi từ admissions_data.json để tìm kiếm
if model and 'question_embeddings' not in st.session_state:
    # Lưu các câu hỏi gốc (có dấu, đã tokenize) để làm key tra cứu
    st.session_state.question_texts = []
    # Dùng để mã hóa thành vector
    unaccented_questions_for_encoding = []
    st.session_state.question_data_map = {}

    for item in admissions_data.get('questions', []):
        questions = item.get('question', [])
        if not isinstance(questions, list):
            questions = [questions]

        for q in questions:
            if not isinstance(q, str) or not q.strip():
                continue

            # Xử lý câu hỏi gốc
            norm_q = normalize_text(q)
            split_q = split_sticky_words(norm_q)
            tokenized_q = ViTokenizer.tokenize(split_q)

            # Tạo phiên bản không dấu để tìm kiếm ngữ nghĩa
            unaccented_q = remove_vietnamese_accents(tokenized_q)
            unaccented_questions_for_encoding.append(unaccented_q)

            # Lưu câu hỏi gốc có dấu làm key
            st.session_state.question_texts.append(tokenized_q)
            st.session_state.question_data_map[tokenized_q] = item

    if unaccented_questions_for_encoding:
        # Mã hóa các câu hỏi KHÔNG DẤU
        st.session_state.question_embeddings = model.encode(unaccented_questions_for_encoding, convert_to_tensor=True)
    else:
        st.session_state.question_embeddings = None


# *** KẾT THÚC THAY ĐỔI 2 ***


# --- HÀM TÌM KIẾM CÂU TRẢ LỜI ---
def find_answer_and_media(question):
    if not model or st.session_state.question_embeddings is None:
        return "Chatbot đang gặp sự cố, vui lòng thử lại sau.", "text", None

    norm_question = normalize_text(question)
    school_pattern = r"(trường\s*thpt|thpt|trường\s*trung\s*học\s*phổ\s*thông|trung\s*học\s*phổ\s*thông|ten\s*lơ\s*men|ten\s*lơ\s*man|ten-lơ-man|ten-lơ-men|trường\s*cấp\s*3|cấp\s*3|cấp\s*ba|trường\s*cấp\s*ba|ernst\s*thälmann|ernst\s*thalmann|trường\s*công\s*lập|công\s*lập|tlm|t.l.m|t l m|trường\s*ten\s*lơ\s*man|trường\s*ten\s*lơ\s*men|truong\s*thpt|truong\s*trung\s*hoc\s*pho\s*thong|truong\s*cap\s*3|truong\s*cap\s*ba|truong\s*cong\s*lap|truong\s*ten\s*lo\s*man|truong\s*ten\s*lo\s*men|trường\s*ernst|trường\s*ernst\s*thälmann|trường\s*ernst\s*thalmann|ernst|trường\s*tlm|trường\s*t.l.m|trường\s*t l m|school|high\s*school|secondary\s*school|tenlo man|tenlo men|tenloman|tenlomen|trường\s*tenlo man|trường\s*tenlo men|trường\s*tenloman|trường\s*tenlomen)"
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
    if not unaccented_clean_question.strip():
        return "Câu hỏi của bạn chưa rõ ràng, vui lòng cung cấp thêm chi tiết.", "text", None
    query_embedding = model.encode(unaccented_clean_question, convert_to_tensor=True)

    cosine_scores = util.pytorch_cos_sim(query_embedding, st.session_state.question_embeddings)[0]
    top_k = min(5, len(st.session_state.question_texts))
    top_results = torch.topk(cosine_scores, k=top_k)
    best_score = top_results[0][0].item()
    best_index = top_results[1][0].item()
    threshold = 0.5

    # Từ khóa liên quan lấy từ câu hỏi gốc
    keywords = ["giáo viên", "kinh nghiệm"]
    related_keywords = [kw for kw in keywords if kw in core_question]
    # Từ khóa chuyên biệt về tổ/môn học
    specific_keywords = ["toán", "lý", "hóa", "sinh", "văn", "sử", "địa", "anh", "tin", "ngoại ngữ", "thể dục", "giáo dục công dân", "mỹ thuật", "âm nhạc"]

    def prettify(text):
        return text.replace("_", " ")

    def filter_related(questions):
        if related_keywords:
            # Chia thành nhóm khái quát và nhóm chuyên biệt
            general = []
            specific = []
            for q in questions:
                if any(kw in q for kw in related_keywords):
                    if any(sk in q for sk in specific_keywords):
                        specific.append(q)
                    else:
                        general.append(q)
            # Ưu tiên nhóm khái quát, sau đó đến nhóm chuyên biệt
            return general + specific
        return questions

    # Nếu không vượt ngưỡng, gợi ý các câu hỏi liên quan
    if best_score < threshold:
        related_questions = []
        for i in range(top_k):
            score = top_results[0][i].item()
            idx = top_results[1][i].item()
            if score > 0.2:
                related_questions.append(st.session_state.question_texts[idx])
        # Chuyển về dạng tự nhiên và lọc liên quan
        related_questions = [prettify(q) for q in related_questions]
        related_questions = filter_related(related_questions)
        if related_questions:
            suggestions = '\n'.join([f"- {q}" for q in related_questions])
            return f"Xin lỗi, tôi chưa tìm thấy thông tin phù hợp. Bạn có thể thử hỏi theo các gợi ý sau:\n{suggestions}", "text", None
        else:
            # Chủ đề phổ biến cũng chuyển về dạng tự nhiên
            popular_questions = []
            for item in admissions_data.get('questions', [])[:3]:
                qs = item.get('question', [])
                if isinstance(qs, str):
                    popular_questions.append(prettify(qs))
                elif isinstance(qs, list) and qs:
                    popular_questions.append(prettify(qs[0]))
            if popular_questions:
                suggestions = '\n'.join([f"- {q}" for q in popular_questions])
                return f"Xin lỗi, tôi chưa tìm thấy thông tin phù hợp. Bạn có thể hỏi về các chủ đề sau:\n{suggestions}", "text", None
            else:
                return "Xin lỗi, tôi chưa tìm thấy thông tin phù hợp. Bạn có thể thử hỏi về tuyển sinh, học phí, hoạt động ngoại khóa, giáo viên, ...", "text", None

    best_match_text = st.session_state.question_texts[best_index]
    best_match_data = st.session_state.question_data_map[best_match_text]
    answer_text = best_match_data.get('answer', "Không có câu trả lời.")
    images = best_match_data.get('images')
    captions = best_match_data.get('captions')
    video_url = best_match_data.get('video_url')

    # Nếu câu hỏi chứa từ khóa liên quan mà câu trả lời không chứa, trả về gợi ý
    if related_keywords:
        if not any(kw in answer_text for kw in related_keywords):
            related_questions = []
            for i in range(top_k):
                score = top_results[0][i].item()
                idx = top_results[1][i].item()
                if score > 0.2:
                    related_questions.append(st.session_state.question_texts[idx])
            related_questions = [prettify(q) for q in related_questions]
            related_questions = filter_related(related_questions)
            if related_questions:
                suggestions = '\n'.join([f"- {q}" for q in related_questions])
                return f"Xin lỗi, tôi chưa tìm thấy thông tin phù hợp. Bạn có thể thử hỏi theo các gợi ý sau:\n{suggestions}", "text", None
            else:
                popular_questions = []
                for item in admissions_data.get('questions', [])[:3]:
                    qs = item.get('question', [])
                    if isinstance(qs, str):
                        popular_questions.append(prettify(qs))
                    elif isinstance(qs, list) and qs:
                        popular_questions.append(prettify(qs[0]))
                if popular_questions:
                    suggestions = '\n'.join([f"- {q}" for q in popular_questions])
                    return f"Xin lỗi, tôi chưa tìm thấy thông tin phù hợp. Bạn có thể hỏi về các chủ đề sau:\n{suggestions}", "text", None
                else:
                    return "Xin lỗi, tôi chưa tìm thấy thông tin phù hợp. Bạn có thể thử hỏi về tuyển sinh, học phí, hoạt động ngoại khóa, giáo viên, ...", "text", None

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


if __name__ == "__main__":
    main()
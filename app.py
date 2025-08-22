# Copyright (c) [2025] [Nguyễn Minh Tấn Phúc]. Bảo lưu mọi quyền.
# Nguồn: https://tlmchattest.streamlit.app/
import json
import os
import re
import time
import torch
from flask import Flask, request, jsonify, render_template, send_from_directory
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from pyvi import ViTokenizer
import unicodedata
from underthesea import word_tokenize

# --- KHỞI TẠO ỨNG DỤNG FLASK ---
app = Flask(__name__)

print('Đang tải chatbot... BẮT ĐẦU')
t0 = time.time()

# --- CÁC HÀM TIỆN ÍCH ---

# Hàm loại bỏ dấu tiếng Việt
def remove_vietnamese_accents(text):
    """
    Hàm này nhận một chuỗi văn bản tiếng Việt và trả về chuỗi đó không có dấu.
    """
    return "".join(c for c in unicodedata.normalize('NFD', text)
                   if unicodedata.category(c) != 'Mn')

# Hàm loại bỏ từ dừng tiếng Việt
def remove_vietnamese_stopwords(tokenized_text):
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
print(f"Sử dụng thiết bị: {device}")

try:
    with open(os.path.join(os.path.dirname(__file__), 'admissions_data.json'), 'r', encoding='utf-8') as f:
        admissions_data = json.load(f)
except Exception as e:
    print(f"Lỗi khi tải admissions_data.json: {e}")
    admissions_data = {"questions": []}

try:
    def load_sentence_transformer_model():
        return SentenceTransformer('bkai-foundation-models/vietnamese-bi-encoder', device=device)

    model = load_sentence_transformer_model()
except Exception as e:
    print(f"Lỗi khi tải mô hình SentenceTransformer: {e}")
    model = None

# Mã hóa phiên bản KHÔNG DẤU của các câu hỏi từ admissions_data.json để tìm kiếm
question_embeddings = None
question_texts = []
question_data_map = {}

if model:
    # Lưu các câu hỏi gốc (có dấu, đã tokenize) để làm key tra cứu
    question_texts = []
    # Dùng để mã hóa thành vector
    unaccented_questions_for_encoding = []
    question_data_map = {}

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
            question_texts.append(tokenized_q)
            question_data_map[tokenized_q] = item

    if unaccented_questions_for_encoding:
        # Mã hóa các câu hỏi KHÔNG DẤU
        question_embeddings = model.encode(unaccented_questions_for_encoding, convert_to_tensor=True)
    else:
        question_embeddings = None

# --- HÀM TÌM KIẾM CÂU TRẢ LỜI ---
def find_answer_and_media(question):
    if not model or question_embeddings is None:
        return "Chatbot đang gặp sự cố, vui lòng thử lại sau.", "text", None

    # 1. Chuẩn hóa và làm sạch câu hỏi của người dùng
    norm_question = normalize_text(question)
    school_patterns = re.compile(
        r'\b(của\s+)?(trường\s+)?(thpt\s+)?(ten\s+lơ\s+man|ernst\s+thälmann|ernst\s+thalmann)\b',
        re.IGNORECASE
    )
    core_question = school_patterns.sub('', norm_question).strip()

    if not core_question or core_question in ['về', 'của']:
        return admissions_data['questions'][22]['answer'], "image", (admissions_data['questions'][22]['images'],
                                                                     admissions_data['questions'][22]['captions'])

    # Xử lý phần lõi của câu hỏi
    split_question = split_sticky_words(core_question)
    tokenized_question = ViTokenizer.tokenize(split_question)
    clean_question = ' '.join(remove_vietnamese_stopwords(tokenized_question.split()))

    # Chuyển câu hỏi của người dùng về dạng không dấu để so sánh
    unaccented_clean_question = remove_vietnamese_accents(clean_question)

    if not unaccented_clean_question.strip():
        return "Câu hỏi của bạn chưa rõ ràng, vui lòng cung cấp thêm chi tiết.", "text", None

    # 2. Tìm kiếm ngữ nghĩa (Semantic Search)
    # Mã hóa câu hỏi KHÔNG DẤU của người dùng
    query_embedding = model.encode(unaccented_clean_question, convert_to_tensor=True)

    cosine_scores = util.pytorch_cos_sim(query_embedding, question_embeddings)[0]

    # 3. Lấy top 5 kết quả có điểm cao nhất
    top_k = min(5, len(question_texts))
    top_results = torch.topk(cosine_scores, k=top_k)

    best_score = top_results[0][0].item()
    best_index = top_results[1][0].item()

    # 4. Kiểm tra ngưỡng điểm
    if best_score < 0.45:
        return "Xin lỗi, tôi không tìm thấy thông tin phù hợp. Bạn có thể thử hỏi bằng cách khác được không?", "text", None

    # 5. Lấy câu trả lời và media tương ứng
    best_match_text = question_texts[best_index]
    best_match_data = question_data_map[best_match_text]

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

print(f'Đã tải chatbot... HOÀN TẤT (thời gian: {time.time() - t0:.2f} giây)')

# --- CÁC ĐIỂM CUỐI (API ENDPOINTS) CỦA FLASK ---

@app.route('/')
def index():
    # Phục vụ file index.html cho giao diện người dùng
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({"error": "Vui lòng cung cấp câu hỏi trong JSON (key: 'question')"}), 400

    question = data['question']
    response, media_type, media_content = find_answer_and_media(question)

    # Xây dựng phản hồi JSON dựa trên kết quả
    json_response = {
        "response": response,
        "media_type": media_type,
        "images": [],
        "captions": [],
        "video_url": None
    }

    if media_type == "video":
        json_response["video_url"] = media_content
    elif media_type == "image":
        images, captions = media_content
        # Tạo đường dẫn URL đầy đủ cho hình ảnh
        json_response["images"] = [f"/images/{os.path.basename(img)}" for img in images if isinstance(img, str) and img.strip()]
        json_response["captions"] = captions if captions else []

    return jsonify(json_response), 200

@app.route('/images/<path:filename>')
def serve_image(filename):
    # Phục vụ các file ảnh tĩnh từ thư mục 'images'
    return send_from_directory('images', filename)

# --- CHẠY ỨNG DỤNG ---
if __name__ == "__main__":
    # Chạy máy chủ Flask, có thể truy cập từ các thiết bị khác trong mạng
    app.run(host='0.0.0.0', port=8080)
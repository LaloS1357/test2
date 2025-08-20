# app.py
# Copyright (c) [2025] [Nguyễn Minh Tấn Phúc]. Bảo lưu mọi quyền.
import json
import os
import re
import time
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
from pyvi import ViTokenizer
import unicodedata
from underthesea import word_tokenize

app = Flask(__name__)

print('loading chatbot... BEGIN')  #region
t0 = time.time()

# Hàm loại bỏ từ dừng tiếng Việt
def remove_vietnamese_stopwords(tokenized_text):
    stopwords = [
        'là', 'của', 'và', 'có', 'trong', 'được', 'cho', 'với', 'tại', 'từ',
        'bởi', 'để', 'như', 'thì', 'mà', 'này', 'kia', 'đó', 'nào', 'cái',
        'những', 'một', 'các', 'đã', 'lại', 'còn', 'nếu', 'vì', 'do', 'bị'
    ]
    tokens = tokenized_text.split() if isinstance(tokenized_text, str) else tokenized_text
    return [token for token in tokens if token not in stopwords]

# Hàm normalize
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

# Load data and model (global để tránh load lại mỗi request)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = None
question_embeddings = None
admissions_data = {"questions": []}

try:
    with open(os.path.join(os.path.dirname(__file__), 'admissions_data.json'), 'r', encoding='utf-8') as f:
        admissions_data = json.load(f)
        if not isinstance(admissions_data, dict) or 'questions' not in admissions_data:
            raise ValueError("admissions_data.json must be a dictionary with a 'questions' key")
        if not isinstance(admissions_data['questions'], list):
            raise ValueError("'questions' in admissions_data.json must be a list")
except FileNotFoundError:
    print("Không tìm thấy file admissions_data.json. Vui lòng kiểm tra lại.")
except ValueError as e:
    print(f"Lỗi trong admissions_data.json: {e}")
except Exception as e:
    print(f"Lỗi khi tải admissions_data.json: {e}")

try:
    model = SentenceTransformer('dangvantuan/vietnamese-embedding', device=device)
    question_texts = []
    question_data_map = {}
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
            question_texts.append(clean_q)
            question_data_map[clean_q] = item
    if question_texts:
        question_embeddings = model.encode(question_texts)
    else:
        question_embeddings = None
        print("Warning: No valid questions found in admissions_data['questions'].")
except Exception as e:
    print(f"Lỗi khi tải mô hình SentenceTransformer: {e}")
    model = None

# Hàm tìm câu trả lời, hình ảnh và video
def find_answer_and_media(question):
    global model, question_embeddings
    if not model or question_embeddings is None:
        return {"error": "Chatbot không thể xử lý vì không có dữ liệu hoặc mô hình."}, 500, None

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
        cosine_scores = cosine_similarity(query_embedding, question_embeddings)[0]
        best_score = np.max(cosine_scores)
        best_index = np.argmax(cosine_scores)
        if best_score < 0.6:
            return {"response": "Xin lỗi, không tìm thấy thông tin phù hợp. Vui lòng kiểm tra lại từ khóa!"}, 404, None
        best_match = question_data_map[question_texts[best_index]]

    answer_text = best_match.get('answer', "Không có câu trả lời.")
    if "images" in best_match and isinstance(best_match["images"], str):
        best_match["images"] = [best_match["images"]]

    has_images = "images" in best_match and best_match["images"]
    has_video = "video_url" in best_match and best_match["video_url"]

    if has_images and has_video:
        images = best_match.get('images', [])
        captions = best_match.get('captions', [])
        video_url = best_match["video_url"]
        return {"response": answer_text, "images": images, "captions": captions, "video_url": video_url}, 200, "multimedia"
    elif has_video:
        return {"response": answer_text, "video_url": best_match["video_url"]}, 200, "video"
    elif has_images:
        images = best_match.get('images', [])
        captions = best_match.get('captions', [])
        valid_images = [img for img in images if isinstance(img, str) and os.path.exists(img) and img.strip() != ""]
        if valid_images and captions is not None:
            valid_captions = captions[:len(valid_images)] if len(captions) >= len(valid_images) else captions + [
                f"Ảnh {i + 1}" for i in range(len(valid_images) - len(captions))]
        else:
            valid_captions = []
        return {"response": answer_text, "images": valid_images, "captions": valid_captions}, 200, "image"
    else:
        return {"response": answer_text}, 200, "text"

print(f'loading chatbot... END (elapsed time: {time.time()-t0:.2f} seconds)')  #endregion


# Endpoint API
@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({"error": "Vui lòng cung cấp câu hỏi trong JSON (key: 'question')"}), 400
    question = data['question']
    response, status_code, media_type = find_answer_and_media(question)
    if status_code != 200:
        return jsonify(response), status_code
    return jsonify({
        "response": response["response"],
        "media_type": media_type,
        "images": response.get("images", []),
        "captions": response.get("captions", []),
        "video_url": response.get("video_url", None)
    }), status_code


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

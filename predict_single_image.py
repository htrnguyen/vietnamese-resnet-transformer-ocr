import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from dataset_polygon import char2idx, idx2char
from model_cnn_transformer import OCRModel
from predict_eval import calculate_metrics, decode_sequence, preprocess_image

# --- Cấu hình ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VOCAB_SIZE = len(char2idx)
MODEL_PATH = "best_ocr_model.pth"
LABEL_DIR = os.path.join("vietnamese", "labels")

# --- Xác định token SOS ---
SOS_TOKEN = None
for token in char2idx.keys():
    if "SOS" in token:
        SOS_TOKEN = token
        print(f"Found SOS token: '{SOS_TOKEN}'")
        break

if SOS_TOKEN is None:
    raise ValueError("SOS token not found in vocabulary!")

# --- Tải mô hình ---
model = OCRModel(vocab_size=VOCAB_SIZE).to(DEVICE)

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print(f"Loaded model from {MODEL_PATH}")
except FileNotFoundError:
    alt_model_path = "ocr_model.pth"
    model.load_state_dict(torch.load(alt_model_path, map_location=DEVICE))
    print(f"Could not find {MODEL_PATH}, loaded model from {alt_model_path} instead")

model.eval()


# --- Hàm dự đoán ---
def greedy_decode(model, image_tensor):
    """Perform greedy decoding using the model"""
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(DEVICE)
        memory = model.encoder(image_tensor)
        ys = torch.tensor([[char2idx[SOS_TOKEN]]], device=DEVICE)

        for i in range(36):  # MAX_LEN
            out = model.decoder(
                ys,
                memory,
                tgt_mask=model.generate_square_subsequent_mask(ys.size(1)).to(DEVICE),
            )

            prob = out[:, -1, :]
            _, next_word = torch.max(prob, dim=1)
            ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1)

            if next_word.item() == char2idx["<EOS>"]:
                break

        return decode_sequence(ys.squeeze(0).tolist())


# --- Hàm đọc label ---
def get_label_from_image_id(img_id):
    """Get the ground truth label for an image ID from the label file"""
    label_path = os.path.join(LABEL_DIR, f"gt_{img_id}.txt")
    if not os.path.exists(label_path):
        return None

    with open(label_path, encoding="utf-8") as f:
        labels = []
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 9 or parts[8] == "###":
                continue
            polygon = list(map(int, parts[:8]))
            text_gt = parts[8]

            # Tính toán bounding box
            x_coords, y_coords = polygon[::2], polygon[1::2]
            x1, y1, x2, y2 = min(x_coords), min(y_coords), max(x_coords), max(y_coords)

            labels.append({"text": text_gt, "box": (x1, y1, x2, y2)})

    return labels


# --- Hàm dự đoán ảnh đơn ---
def predict_image(img_path):
    """Predict text from an image and compare with ground truth if available"""
    # Get image ID from filename
    img_filename = os.path.basename(img_path)
    img_id = img_filename.replace("im", "").replace(".jpg", "")

    try:
        img_id = int(img_id)
    except ValueError:
        print(f"Cannot extract image ID from filename: {img_filename}")
        return

    # Get ground truth labels
    labels = get_label_from_image_id(img_id)

    if not labels:
        print(f"No label found for image ID: {img_id}")

        # Dự đoán trên toàn bộ ảnh nếu không có nhãn
        try:
            img = cv2.imread(img_path)
            if img is None:
                print(f"Error: Could not read image {img_path}")
                return

            # Hiển thị ảnh gốc
            plt.figure(figsize=(12, 12))
            plt.subplot(1, 1, 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img_rgb)
            plt.title(f"Image: {img_filename}")
            plt.axis("off")

            # Dự đoán
            image_tensor = preprocess_image(img_path)
            pred_text = greedy_decode(model, image_tensor)
            print(f"Prediction: {pred_text}")

            plt.show()

        except Exception as e:
            print(f"Error processing image: {e}")

        return

    # Đọc ảnh
    try:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Could not read image {img_path}")
            return

        # Hiển thị kết quả cho từng vùng chữ
        num_labels = len(labels)
        cols = min(3, num_labels)
        rows = (num_labels + cols - 1) // cols

        plt.figure(figsize=(15, 5 * rows))

        for i, label_data in enumerate(labels):
            text_gt = label_data["text"]
            x1, y1, x2, y2 = label_data["box"]

            # Đảm bảo tọa độ hợp lệ
            height, width = img.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)

            # Kiểm tra kích thước vùng cắt
            if x1 >= x2 or y1 >= y2:
                print(f"Warning: Invalid crop region: {x1},{y1},{x2},{y2}")
                continue

            # Cắt vùng chữ
            img_crop = img[y1:y2, x1:x2]
            if img_crop.size == 0:
                continue

            img_rgb = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)

            # Dự đoán
            image_tensor = preprocess_image(img_path, (x1, y1, x2, y2))
            pred_text = greedy_decode(model, image_tensor)

            # Tính metrics
            metrics = calculate_metrics(text_gt, pred_text)

            # Hiển thị kết quả
            plt.subplot(rows, cols, i + 1)
            plt.imshow(img_rgb)
            plt.title(
                f"GT: {text_gt}\nPred: {pred_text}\nCER: {metrics['cer']:.4f}, Acc: {metrics['char_accuracy']:.4f}"
            )
            plt.axis("off")

        plt.tight_layout()
        plt.show()

        print(f"Image: {img_filename}, ID: {img_id}")
        print(f"Found {num_labels} text regions")

    except Exception as e:
        print(f"Error processing image: {e}")


# --- Main ---
if __name__ == "__main__":
    while True:
        img_path = input("\nNhập đường dẫn đến ảnh (hoặc nhập 'q' để thoát): ")
        if img_path.lower() == "q":
            break

        if not os.path.exists(img_path):
            print(f"Lỗi: Không tìm thấy file {img_path}")
            continue

        predict_image(img_path)

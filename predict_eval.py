import os

import cv2
import Levenshtein
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from dataset_polygon import char2idx, encode_text, idx2char
from model_cnn_transformer import OCRModel

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VOCAB_SIZE = len(char2idx)
MODEL_PATH = "best_ocr_model.pth"
IMAGE_DIR = os.path.join("vietnamese", "test_image")
UNSEEN_IMAGE_DIR = os.path.join("vietnamese", "unseen_test_images")
LABEL_DIR = os.path.join("vietnamese", "labels")
MAX_LEN = 36
SAVE_RESULTS = True
RESULTS_DIR = "results"

# Xác định token SOS chính xác từ char2idx để đảm bảo nhất quán
SOS_TOKEN = None
for token in char2idx.keys():
    if "SOS" in token:
        SOS_TOKEN = token
        print(f"Found SOS token: '{SOS_TOKEN}'")
        break

if SOS_TOKEN is None:
    raise ValueError("SOS token not found in vocabulary!")

if SAVE_RESULTS and not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)


# --- Utility functions ---
def preprocess_image(img_path, box=None):
    """Preprocess an image for model input"""
    image = Image.open(img_path).convert("RGB")

    # If bounding box is provided, crop the image
    if box:
        x1, y1, x2, y2 = box
        image = image.crop((x1, y1, x2, y2))

    transform = transforms.Compose(
        [
            transforms.Resize((32, 128)),  # Match training transform
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform(image).unsqueeze(0)


def decode_sequence(indices):
    """Convert model output indices to characters"""
    chars = []
    for idx in indices:
        ch = idx2char.get(idx, "")
        if ch == "<EOS>":
            break
        if ch not in ("<PAD>", SOS_TOKEN):
            chars.append(ch)
    return "".join(chars)


def greedy_decode(model, image_tensor):
    """Perform greedy decoding using the model"""
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(DEVICE)
        memory = model.encoder(image_tensor)
        ys = torch.tensor(
            [[char2idx[SOS_TOKEN]]], device=DEVICE
        )  # Use dynamic SOS token

        # Initialize attention maps list (empty by default)
        attention_maps = []

        for _ in range(MAX_LEN):
            # Get decoder output and attention weights
            out = model.decoder(
                ys,
                memory,
                tgt_mask=model.generate_square_subsequent_mask(ys.size(1)).to(DEVICE),
            )

            # Note: In standard PyTorch implementation, attention weights are not accessible
            # directly from MultiheadAttention. We'll skip this part but keep the structure
            # for possible future modifications.

            prob = out[:, -1, :]
            _, next_word = torch.max(prob, dim=1)
            ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1)
            if next_word.item() == char2idx["<EOS>"]:
                break

        return decode_sequence(ys.squeeze(0).tolist()), attention_maps


def calculate_metrics(ground_truth, prediction):
    """Calculate evaluation metrics between ground truth and prediction"""
    # Character Error Rate (CER)
    edit_distance = Levenshtein.distance(ground_truth, prediction)
    cer = edit_distance / max(len(ground_truth), 1)

    # Word Accuracy - exact match between ground truth and prediction
    word_accuracy = 1.0 if ground_truth == prediction else 0.0

    # Accuracy at character level
    min_len = min(len(ground_truth), len(prediction))
    char_matches = sum(1 for i in range(min_len) if ground_truth[i] == prediction[i])
    char_accuracy = char_matches / max(len(ground_truth), len(prediction), 1)

    return {
        "edit_distance": edit_distance,
        "cer": cer,
        "word_accuracy": word_accuracy,
        "char_accuracy": char_accuracy,
    }


def visualize_attention(image, text, attention_map=None, save_path=None):
    """Visualize input image, prediction, and attention heatmap"""
    plt.figure(figsize=(12, 6))

    # Plot original image
    plt.subplot(1, 2 if attention_map is not None else 1, 1)
    plt.imshow(image)
    plt.title(f"Text: {text}")
    plt.axis("off")

    # Plot attention heatmap if available
    if attention_map is not None:
        plt.subplot(1, 2, 2)
        att_map = attention_map.mean(dim=0)[0]  # Average over attention heads
        plt.imshow(image)

        # Reshape attention to image dimensions
        h, w = image.shape[:2]
        att_h, att_w = int(np.sqrt(att_map.shape[0])), int(np.sqrt(att_map.shape[0]))
        att_map = att_map.reshape(att_h, att_w)
        att_map = cv2.resize(att_map.numpy(), (w, h))

        plt.imshow(att_map, alpha=0.6, cmap=cm.jet)
        plt.title("Attention Heatmap")
        plt.axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


# --- Load model ---
model = OCRModel(vocab_size=VOCAB_SIZE).to(DEVICE)

# Try loading the best model, fall back to regular model if not found
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print(f"Loaded model from {MODEL_PATH}")
except FileNotFoundError:
    alt_model_path = "ocr_model.pth"
    model.load_state_dict(torch.load(alt_model_path, map_location=DEVICE))
    print(f"Could not find {MODEL_PATH}, loaded model from {alt_model_path} instead")

model.eval()


# --- Evaluate on test set ---
def evaluate_dataset(image_dir, label_dir, num_samples=None, save_prefix="test"):
    print(f"💬 Evaluating on {image_dir}...")

    # Get image files from directory
    image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
    if num_samples:
        image_files = image_files[:num_samples]

    results = []
    all_metrics = {
        "total": 0,
        "edit_distance": 0,
        "cer": 0,
        "word_accuracy": 0,
        "char_accuracy": 0,
    }

    for img_file in tqdm(image_files):
        img_id = int(img_file.replace("im", "").replace(".jpg", ""))
        img_path = os.path.join(image_dir, img_file)
        label_path = os.path.join(label_dir, f"gt_{img_id}.txt")

        if not os.path.exists(label_path):
            print(f"Warning: Label file not found for {img_file}")
            continue

        with open(label_path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 9 or parts[8] == "###":
                    continue

                try:
                    polygon = list(map(int, parts[:8]))
                    text_gt = parts[8]

                    x_coords, y_coords = polygon[::2], polygon[1::2]
                    x1, y1, x2, y2 = (
                        min(x_coords),
                        min(y_coords),
                        max(x_coords),
                        max(y_coords),
                    )

                    # Kiểm tra tọa độ là hợp lệ
                    if x1 >= x2 or y1 >= y2:
                        print(
                            f"Warning: Invalid coordinates for {img_file}: {x1},{y1},{x2},{y2}"
                        )
                        continue

                    image_tensor = preprocess_image(img_path, (x1, y1, x2, y2))
                    pred_text, attention_maps = greedy_decode(model, image_tensor)

                    # Calculate metrics
                    metrics = calculate_metrics(text_gt, pred_text)
                    for key in metrics:
                        all_metrics[key] += metrics[key]
                    all_metrics["total"] += 1

                    # Store result
                    result = {
                        "img_id": img_id,
                        "gt": text_gt,
                        "pred": pred_text,
                        "metrics": metrics,
                    }
                    results.append(result)

                    # Visualize and save sample images
                    if SAVE_RESULTS and (len(results) % 10 == 0 or len(results) <= 10):
                        img_np = cv2.imread(img_path)

                        # Kiểm tra xem đọc ảnh có thành công không
                        if img_np is None:
                            print(f"Warning: Could not read image {img_path}")
                            continue

                        # Đảm bảo tọa độ nằm trong phạm vi ảnh
                        height, width = img_np.shape[:2]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(width, x2), min(height, y2)

                        # Kiểm tra kích thước crop phải > 0
                        if x1 >= x2 or y1 >= y2:
                            print(
                                f"Warning: Invalid crop region for {img_file}: {x1},{y1},{x2},{y2}"
                            )
                            continue

                        img_crop = img_np[y1:y2, x1:x2]

                        # Kiểm tra xem crop có thành công không
                        if img_crop.size == 0:
                            print(
                                f"Warning: Empty crop for {img_file} at {x1},{y1},{x2},{y2}"
                            )
                            continue

                        img_rgb = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)

                        # Get last attention map if available
                        attention_map = None
                        if attention_maps and len(attention_maps) > 0:
                            attention_map = attention_maps[-1]  # Last timestep

                        save_path = os.path.join(
                            RESULTS_DIR, f"{save_prefix}_{img_id}.png"
                        )
                        try:
                            visualize_attention(
                                img_rgb,
                                f"GT: {text_gt} | Pred: {pred_text}",
                                attention_map,
                                save_path,
                            )
                        except Exception as e:
                            print(
                                f"Warning: Failed to visualize results for {img_file}: {e}"
                            )
                except Exception as e:
                    print(f"Error processing {img_file}, line: {line.strip()}: {e}")
                    continue

    # Kiểm tra nếu không có kết quả nào
    if all_metrics["total"] == 0:
        print("No valid samples were evaluated.")
        return [], {}

    # Calculate average metrics
    avg_metrics = {
        k: v / all_metrics["total"] for k, v in all_metrics.items() if k != "total"
    }

    # Print summary
    print("\n===== Evaluation Summary =====")
    print(f"Total samples evaluated: {all_metrics['total']}")
    print(f"Average Edit Distance: {avg_metrics['edit_distance']:.4f}")
    print(f"Average Character Error Rate (CER): {avg_metrics['cer']:.4f}")
    print(f"Word Accuracy: {avg_metrics['word_accuracy']:.4f}")
    print(f"Character Accuracy: {avg_metrics['char_accuracy']:.4f}")
    print("=============================\n")

    return results, avg_metrics


# --- Run evaluation ---
print("🔍 Running evaluation on test set...")
test_results, test_metrics = evaluate_dataset(
    IMAGE_DIR, LABEL_DIR, num_samples=50, save_prefix="test"
)

# --- Run evaluation on unseen test set (if available) ---
if os.path.exists(UNSEEN_IMAGE_DIR):
    print("\n🔍 Running evaluation on unseen test set...")
    unseen_results, unseen_metrics = evaluate_dataset(
        UNSEEN_IMAGE_DIR, LABEL_DIR, num_samples=20, save_prefix="unseen"
    )

    # Compare metrics between test set and unseen test set
    print("\n===== Test vs Unseen Test =====")
    for metric in test_metrics:
        print(
            f"{metric}: {test_metrics[metric]:.4f} (test) vs {unseen_metrics[metric]:.4f} (unseen)"
        )
    print("==============================\n")


# --- Interactive demo ---
def demo():
    print("\n🔮 Interactive Demo Mode")
    print("Enter image path (relative to working directory) or 'q' to quit:")

    while True:
        user_input = input("> ")
        if user_input.lower() == "q":
            break

        if not os.path.exists(user_input):
            print(f"Error: File '{user_input}' not found.")
            continue

        try:
            # Process full image without cropping
            image_tensor = preprocess_image(user_input)
            pred_text, _ = greedy_decode(model, image_tensor)

            # Display result
            img = cv2.imread(user_input)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            plt.figure(figsize=(10, 6))
            plt.imshow(img_rgb)
            plt.title(f"Prediction: {pred_text}")
            plt.axis("off")
            plt.show()

        except Exception as e:
            print(f"Error processing image: {e}")


# Uncomment to run interactive demo
# demo()

print("✅ Evaluation completed. Results saved in:", RESULTS_DIR)

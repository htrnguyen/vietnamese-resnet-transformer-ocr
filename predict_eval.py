import os
from collections import Counter

import cv2
import Levenshtein
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix
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
EVAL_CHARTS_DIR = os.path.join(RESULTS_DIR, "evaluation_charts")

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

if SAVE_RESULTS and not os.path.exists(EVAL_CHARTS_DIR):
    os.makedirs(EVAL_CHARTS_DIR)


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

    # For confusion matrix
    all_gt_chars = []
    all_pred_chars = []

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

                    # Collect characters for confusion matrix
                    min_len = min(len(text_gt), len(pred_text))
                    for i in range(min_len):
                        all_gt_chars.append(text_gt[i])
                        all_pred_chars.append(pred_text[i])

                    # Add extra characters (either gt or pred may be longer)
                    if len(text_gt) > len(pred_text):
                        for i in range(min_len, len(text_gt)):
                            all_gt_chars.append(text_gt[i])
                            all_pred_chars.append("")  # Represents deletion
                    elif len(pred_text) > len(text_gt):
                        for i in range(min_len, len(pred_text)):
                            all_gt_chars.append("")  # Represents insertion
                            all_pred_chars.append(pred_text[i])

                    # Store result
                    result = {
                        "img_id": img_id,
                        "gt": text_gt,
                        "pred": pred_text,
                        "metrics": metrics,
                    }
                    results.append(result)

                    # Visualize and save sample images
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

                    save_path = os.path.join(RESULTS_DIR, f"{save_prefix}_{img_id}.png")
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
        return [], {}, [], []

    # Calculate average metrics
    avg_metrics = {
        k: v / all_metrics["total"] for k, v in all_metrics.items() if k != "total"
    }

    # Print summary
    print("\n===== Evaluation Summary =====")
    print(f"Total samples evaluated: {all_metrics['total']}")
    print(f"Average Edit Distance: {avg_metrics['edit_distance']:.4f}")
    print(f"Character Error Rate (CER): {avg_metrics['cer']:.4f}")
    print(f"Word Accuracy: {avg_metrics['word_accuracy']:.4f}")
    print(f"Character Accuracy: {avg_metrics['char_accuracy']:.4f}")
    print("=============================\n")

    # Create and save detailed metrics visualization
    if SAVE_RESULTS:
        create_metrics_visualization(results, save_prefix)

    # Create character error visualization
    if SAVE_RESULTS and len(all_gt_chars) > 0:
        create_character_error_visualization(all_gt_chars, all_pred_chars, save_prefix)

    return results, avg_metrics, all_gt_chars, all_pred_chars


def create_metrics_visualization(results, save_prefix):
    """Create and save detailed metrics visualization"""
    # Extract per-sample metrics
    sample_metrics = {
        "cer": [r["metrics"]["cer"] for r in results],
        "edit_distance": [r["metrics"]["edit_distance"] for r in results],
        "word_accuracy": [r["metrics"]["word_accuracy"] for r in results],
        "char_accuracy": [r["metrics"]["char_accuracy"] for r in results],
        "gt_length": [len(r["gt"]) for r in results],
    }

    # Create a figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot CER distribution
    axes[0, 0].hist(sample_metrics["cer"], bins=20, color="skyblue", edgecolor="black")
    axes[0, 0].set_title("Character Error Rate Distribution")
    axes[0, 0].set_xlabel("CER")
    axes[0, 0].set_ylabel("Number of samples")

    # Plot edit distance vs text length
    axes[0, 1].scatter(
        sample_metrics["gt_length"], sample_metrics["edit_distance"], alpha=0.6
    )
    axes[0, 1].set_title("Edit Distance vs. Text Length")
    axes[0, 1].set_xlabel("Ground Truth Length")
    axes[0, 1].set_ylabel("Edit Distance")

    # Plot Character Accuracy distribution
    axes[1, 0].hist(
        sample_metrics["char_accuracy"], bins=20, color="lightgreen", edgecolor="black"
    )
    axes[1, 0].set_title("Character Accuracy Distribution")
    axes[1, 0].set_xlabel("Character Accuracy")
    axes[1, 0].set_ylabel("Number of samples")

    # Plot Word Accuracy distribution
    axes[1, 1].hist(
        sample_metrics["word_accuracy"], bins=3, color="coral", edgecolor="black"
    )
    axes[1, 1].set_title("Word Accuracy Distribution")
    axes[1, 1].set_xlabel("Word Accuracy")
    axes[1, 1].set_ylabel("Number of samples")

    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_CHARTS_DIR, f"{save_prefix}_metrics_detailed.png"))
    plt.close()


def create_character_error_visualization(gt_chars, pred_chars, save_prefix):
    """Create and save character error visualizations"""

    # Count character occurrences
    gt_counter = Counter(gt_chars)
    pred_counter = Counter(pred_chars)

    # Get unique characters from both gt and pred
    all_chars = sorted(list(set(gt_chars + pred_chars)))

    # Filter out empty string if present
    if "" in all_chars:
        all_chars.remove("")

    # If too many characters, limit to top 30
    if len(all_chars) > 30:
        # Get top characters by frequency
        top_chars_dict = {
            char: gt_counter.get(char, 0) + pred_counter.get(char, 0)
            for char in all_chars
        }
        all_chars = [
            char
            for char, _ in sorted(
                top_chars_dict.items(), key=lambda x: x[1], reverse=True
            )[:30]
        ]

    # Create confusion matrix for character prediction
    cm = confusion_matrix(
        [c if c in all_chars else "OTHER" for c in gt_chars],
        [c if c in all_chars else "OTHER" for c in pred_chars],
        labels=all_chars + ["OTHER"],
    )

    # Normalize by row (ground truth)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)  # Replace NaN with 0

    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm_norm,
        annot=False,
        fmt=".2f",
        cmap="Blues",
        xticklabels=all_chars + ["OTHER"],
        yticklabels=all_chars + ["OTHER"],
    )
    plt.title("Character Prediction Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.tight_layout()
    plt.savefig(
        os.path.join(EVAL_CHARTS_DIR, f"{save_prefix}_char_confusion_matrix.png")
    )
    plt.close()

    # Character frequency comparison
    char_freq_gt = [gt_counter.get(c, 0) for c in all_chars]
    char_freq_pred = [pred_counter.get(c, 0) for c in all_chars]

    # Plot character frequency comparison
    plt.figure(figsize=(14, 6))
    width = 0.35
    x = np.arange(len(all_chars))
    plt.bar(x - width / 2, char_freq_gt, width, label="Ground Truth")
    plt.bar(x + width / 2, char_freq_pred, width, label="Predicted")
    plt.xlabel("Character")
    plt.ylabel("Frequency")
    plt.title("Character Frequency Comparison")
    plt.xticks(x, all_chars, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_CHARTS_DIR, f"{save_prefix}_char_frequency.png"))
    plt.close()

    # Identify most problematic characters
    error_rates = {}
    for char in all_chars:
        gt_indices = [i for i, c in enumerate(gt_chars) if c == char]
        if not gt_indices:
            continue

        errors = sum(1 for i in gt_indices if pred_chars[i] != char)
        error_rate = errors / len(gt_indices)
        error_rates[char] = (error_rate, len(gt_indices))

    # Plot character error rates
    chars, rates = zip(
        *[
            (c, r)
            for c, (r, _) in sorted(
                error_rates.items(), key=lambda x: x[1][0], reverse=True
            )[:20]
        ]
    )
    plt.figure(figsize=(14, 6))
    plt.bar(chars, rates, color="salmon")
    plt.xlabel("Character")
    plt.ylabel("Error Rate")
    plt.title("Top 20 Characters with Highest Error Rates")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_CHARTS_DIR, f"{save_prefix}_char_error_rates.png"))
    plt.close()


# --- Run evaluation ---
print("🔍 Running evaluation on test set...")
test_results, test_metrics, test_gt_chars, test_pred_chars = evaluate_dataset(
    IMAGE_DIR, LABEL_DIR, num_samples=50, save_prefix="test"
)

# --- Run evaluation on unseen test set (if available) ---
unseen_results = []
unseen_metrics = {}
unseen_gt_chars = []
unseen_pred_chars = []

if os.path.exists(UNSEEN_IMAGE_DIR):
    print("\n🔍 Running evaluation on unseen test set...")
    unseen_results, unseen_metrics, unseen_gt_chars, unseen_pred_chars = (
        evaluate_dataset(
            UNSEEN_IMAGE_DIR, LABEL_DIR, num_samples=20, save_prefix="unseen"
        )
    )

    # Compare metrics between test set and unseen test set
    print("\n===== Test vs Unseen Test =====")
    for metric in test_metrics:
        print(
            f"{metric}: {test_metrics[metric]:.4f} (test) vs {unseen_metrics[metric]:.4f} (unseen)"
        )
    print("==============================\n")

    # Create comparison chart
    if SAVE_RESULTS:
        plt.figure(figsize=(10, 6))
        metrics = ["cer", "word_accuracy", "char_accuracy"]
        x = np.arange(len(metrics))
        width = 0.35

        test_values = [test_metrics[m] for m in metrics]
        unseen_values = [unseen_metrics[m] for m in metrics]

        plt.bar(x - width / 2, test_values, width, label="Test Set")
        plt.bar(x + width / 2, unseen_values, width, label="Unseen Test Set")

        plt.xlabel("Metrics")
        plt.ylabel("Value")
        plt.title("Performance Comparison: Test vs. Unseen Test")
        plt.xticks(x, ["CER (lower is better)", "Word Accuracy", "Character Accuracy"])
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(EVAL_CHARTS_DIR, "test_vs_unseen_comparison.png"))
        plt.close()

        # Create a summary of results in CSV format
        summary_df = pd.DataFrame(
            {
                "Metric": [
                    "Character Error Rate (CER)",
                    "Word Accuracy",
                    "Character Accuracy",
                    "Average Edit Distance",
                    "Sample Count",
                ],
                "Test Set": [
                    test_metrics["cer"],
                    test_metrics["word_accuracy"],
                    test_metrics["char_accuracy"],
                    test_metrics["edit_distance"],
                    len(test_results),
                ],
                "Unseen Test Set": [
                    unseen_metrics["cer"],
                    unseen_metrics["word_accuracy"],
                    unseen_metrics["char_accuracy"],
                    unseen_metrics["edit_distance"],
                    len(unseen_results),
                ],
            }
        )

        summary_df.to_csv(
            os.path.join(RESULTS_DIR, "evaluation_summary.csv"), index=False
        )
        print(
            f"Saved evaluation summary to {os.path.join(RESULTS_DIR, 'evaluation_summary.csv')}"
        )


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
print("📊 Evaluation charts saved in:", EVAL_CHARTS_DIR)

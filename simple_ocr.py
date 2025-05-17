import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from dataset_polygon import char2idx, idx2char
from model_cnn_transformer import OCRModel


def load_model(model_path="models/best_ocr_model.pth"):
    """Load the trained OCR model"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = OCRModel(vocab_size=len(char2idx)).to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()
    return model, device


def preprocess_image(image):
    """Preprocess image for model input"""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    transform = transforms.Compose(
        [
            transforms.Resize((32, 128)),
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
        if ch not in ("<PAD>", "<SOS>"):
            chars.append(ch)
    return "".join(chars)


def detect_text_regions(image):
    """Detect text regions in the image using OpenCV"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter and process contours
    text_regions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Filter based on size and aspect ratio
        if w < 20 or h < 20:  # Skip too small regions
            continue
        aspect_ratio = w / float(h)
        if aspect_ratio < 0.1 or aspect_ratio > 10:  # Skip extreme aspect ratios
            continue

        text_regions.append((x, y, x + w, y + h))

    return text_regions


def process_image(image_path, model, device):
    """Process image and return OCR results with visualization"""
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")

    # Make a copy for visualization
    vis_image = image.copy()

    # Detect text regions
    text_regions = detect_text_regions(image)

    # Process each text region
    results = []
    for x1, y1, x2, y2 in text_regions:
        # Crop and preprocess region
        region = image[y1:y2, x1:x2]
        if region.size == 0:
            continue

        # Get prediction
        with torch.no_grad():
            image_tensor = preprocess_image(region).to(device)
            memory = model.encoder(image_tensor)

            # Initialize with SOS token
            ys = torch.tensor([[char2idx["<SOS>"]]], device=device)

            # Decode sequence
            for _ in range(36):  # max length
                out = model.decoder(
                    ys,
                    memory,
                    tgt_mask=model.generate_square_subsequent_mask(ys.size(1)).to(
                        device
                    ),
                )
                prob = out[:, -1, :]
                _, next_word = torch.max(prob, dim=1)
                ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1)
                if next_word.item() == char2idx["<EOS>"]:
                    break

            pred_text = decode_sequence(ys.squeeze(0).tolist())

        if pred_text:  # Only add non-empty predictions
            results.append({"box": (x1, y1, x2, y2), "text": pred_text})

            # Draw bounding box and text
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                vis_image,
                pred_text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

    return vis_image, results


def ocr_image(image_path, model_path="models/best_ocr_model.pth", save_path=None):
    """
    Process an image with OCR and return/save the result

    Args:
        image_path (str): Path to input image
        model_path (str): Path to model weights
        save_path (str, optional): Path to save result image. If None, only returns the result

    Returns:
        tuple: (result_image, list of detected texts)
    """
    # Load model
    model, device = load_model(model_path)

    # Process image
    result_image, results = process_image(image_path, model, device)

    # Save result if path provided
    if save_path:
        cv2.imwrite(save_path, result_image)
        print(f"Result saved to {save_path}")

    # Print detected texts
    print("\nDetected texts:")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['text']}")

    return result_image, results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python simple_ocr.py <image_path> [save_path]")
        sys.exit(1)

    image_path = sys.argv[1]
    save_path = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        result_image, results = ocr_image(image_path, save_path=save_path)

        # Show result
        cv2.imshow("OCR Result", result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from model_cnn_transformer import OCRModel
from dataset_polygon import OCRDataset, char2idx, idx2char
import torchvision.transforms as transforms

# --- Hyperparameters ---
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
VOCAB_SIZE = len(char2idx)
MODEL_SAVE_PATH = 'ocr_model.pth'
BEST_MODEL_PATH = 'best_ocr_model.pth'

print(f"Using device: {DEVICE}")
print(f"Vocabulary size: {VOCAB_SIZE}")

# --- Data transforms ---
transform = transforms.Compose([
    transforms.Resize((32, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- Collate function ---
def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images)
    targets_input = [t[:-1] for t in targets]
    targets_output = [t[1:] for t in targets]
    targets_input = pad_sequence(targets_input, batch_first=True, padding_value=char2idx['<PAD>'])
    targets_output = pad_sequence(targets_output, batch_first=True, padding_value=char2idx['<PAD>'])
    return images, targets_input, targets_output

# --- Datasets ---
train_dataset = OCRDataset(
    image_dir=os.path.join("vietnamese", "train_images"),
    label_dir=os.path.join("vietnamese", "labels"),
    transform=transform
)

test_dataset = OCRDataset(
    image_dir=os.path.join("vietnamese", "test_image"),
    label_dir=os.path.join("vietnamese", "labels"),
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

# --- Model ---
model = OCRModel(vocab_size=VOCAB_SIZE).to(DEVICE)
criterion = nn.CrossEntropyLoss(ignore_index=char2idx['<PAD>'])
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

# --- Validation function ---
def validate(model, dataloader):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, tgt_input, tgt_output in dataloader:
            images, tgt_input, tgt_output = images.to(DEVICE), tgt_input.to(DEVICE), tgt_output.to(DEVICE)
            
            output = model(images, tgt_input)
            output = output.view(-1, VOCAB_SIZE)
            tgt_output = tgt_output.view(-1)
            
            loss = criterion(output, tgt_output)
            val_loss += loss.item()
            
    return val_loss / len(dataloader)

# --- Training loop ---
best_val_loss = float('inf')
for epoch in range(EPOCHS):
    # Training
    model.train()
    epoch_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for images, tgt_input, tgt_output in pbar:
        images, tgt_input, tgt_output = images.to(DEVICE), tgt_input.to(DEVICE), tgt_output.to(DEVICE)

        output = model(images, tgt_input)  # (B, T, Vocab)
        output = output.view(-1, VOCAB_SIZE)
        tgt_output = tgt_output.view(-1)

        loss = criterion(output, tgt_output)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()

        epoch_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    avg_train_loss = epoch_loss / len(train_loader)
    
    # Validation
    val_loss = validate(model, test_loader)
    
    print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")
    
    # Save model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"Best model saved with val_loss: {val_loss:.4f}")
    
    # Update learning rate
    scheduler.step(val_loss)

print("Training completed!")

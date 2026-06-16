import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import glob
import os

from dataset import RadermeckerDataset
from model import RadermeckerTransformer

# ==========================================
# FOCAL LOSS IMPLEMENTATION
# ==========================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ==========================================
# 1. Configuration & Hyperparameters
# ==========================================
EPOCHS = 50
BATCH_SIZE = 32 
LEARNING_RATE = 1e-4

# Aggressive punishment multiplier for patient memorization
MAX_GRL_ALPHA = 2.5 

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Training on device: {device}")

# ==========================================
# 2. Dynamic Data Loading & Balancing
# ==========================================
print("\nScanning training directory for patient files...")
raw_files = glob.glob("data/train/*.edf") + glob.glob("data/train/*.EDF")
train_files = list(set(raw_files))
train_files.sort()

if len(train_files) == 0:
    raise FileNotFoundError("No EDF files found in 'data/train/'. Please ensure files are placed correctly.")

patient_ids = list(range(len(train_files)))
print(f"-> Found {len(train_files)} patient file(s): {train_files}")

# Note: Your dataset.py MUST be updated to return the 6 physical features!
dataset = RadermeckerDataset(
    edf_files=train_files, 
    event_dict={
        'SSPE': 0,          
        'Background': 1,        
        'sedated sleep': 1,     
        'trying to awake': 1,   
        'ELECT. ARTIFACT': 2    
    }, 
    patient_ids=patient_ids
)

if len(dataset) == 0:
    raise ValueError("Dataset is empty. Check your EDF files and TSV annotations.")

unique_pids = np.unique(dataset.patient_labels)
pid_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_pids)}
dataset.patient_labels = np.array([pid_mapping[pid] for pid in dataset.patient_labels])

class_counts = np.bincount(dataset.labels)
class_weights = 1.0 / np.maximum(class_counts, 1) 
sample_weights = class_weights[dataset.labels]

sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler, drop_last=False)

# ==========================================
# 3. Model & Optimizer Initialization
# ==========================================
print("\nBuilding Multimodal Transformer Architecture...")
num_channels = len(dataset.STANDARD_10_20)  
num_clinical_classes = len(np.unique(dataset.labels))
num_patients = len(np.unique(dataset.patient_labels))

model = RadermeckerTransformer(
    num_channels=num_channels,
    d_model=64, 
    nhead=4, 
    num_layers=3, 
    num_clinical_classes=num_clinical_classes, 
    num_patients=num_patients
).to(device)

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

loss_weights = torch.tensor([3.0, 1.0, 1.0], dtype=torch.float32).to(device)
criterion_clinical = FocalLoss(alpha=loss_weights, gamma=2.0)
criterion_patient = nn.CrossEntropyLoss()

# ==========================================
# 4. The Adversarial Training Loop
# ==========================================
print("\nInitiating Multimodal Adversarial Training Phase...")

for epoch in range(EPOCHS):
    model.train()
    
    p = float(epoch) / EPOCHS
    
    # Overdrive the Alpha schedule to hit MAX_GRL_ALPHA
    alpha = MAX_GRL_ALPHA * (2. / (1. + np.exp(-10 * p)) - 1)
    
    running_clinical_loss = 0.0
    running_patient_loss = 0.0
    correct_clinical = 0
    total_samples = 0
    
    if len(train_loader) == 0:
         print("Warning: Train loader is empty!")
         break

    for batch_idx, (x, physics_tensor, y_clinical, y_patient) in enumerate(train_loader):
        
        x = x.to(device)
        
        # -> THE FIX: We now expect and pass ALL 6 features to the FiLM layer
        # Ensure your model.py FiLMLayer is initialized with cond_dim=6
        physics_tensor = physics_tensor[:, :6].to(device) 
        
        y_clinical = y_clinical.to(device)
        y_patient = y_patient.to(device)
        
        # Biological Noise Injection (50% chance per batch)
        if torch.rand(1).item() < 0.5:
            jitter = torch.randn_like(x) * 0.05
            x = x + jitter
        
        optimizer.zero_grad()
        
        clinical_preds, patient_preds, latent_features, _ = model(x, physics_tensor, alpha=alpha)
        
        loss_clinical = criterion_clinical(clinical_preds, y_clinical)
        loss_patient = criterion_patient(patient_preds, y_patient)
        
        total_loss = loss_clinical + loss_patient
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_clinical_loss += loss_clinical.item()
        running_patient_loss += loss_patient.item()
        
        _, predicted = torch.max(clinical_preds.data, 1)
        total_samples += y_clinical.size(0)
        correct_clinical += (predicted == y_clinical).sum().item()

    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]

    epoch_acc = 100 * correct_clinical / total_samples
    avg_clin_loss = running_clinical_loss / len(train_loader)
    avg_pat_loss = running_patient_loss / len(train_loader)
    
    print(f"Epoch [{epoch+1:02d}/{EPOCHS}] "
          f"| Alpha: {alpha:.3f} "
          f"| LR: {current_lr:.6f} "
          f"| Clinical Loss: {avg_clin_loss:.4f} "
          f"| Patient Loss: {avg_pat_loss:.4f} "
          f"| Clinical Acc: {epoch_acc:.2f}%")

torch.save(model.state_dict(), "sspe_transformer_weights.pth")
print("\nTraining Complete! Multimodal Model weights saved to 'sspe_transformer_weights.pth'.")
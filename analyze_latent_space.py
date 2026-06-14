import torch
import numpy as np
import pandas as pd
from scipy.signal import welch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from dataset import RadermeckerDataset 
from model import RadermeckerTransformer 

# ==========================================
# Configuration
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load your labeled dataset to see how the AI categorizes the known data
dataset = RadermeckerDataset(
    edf_files=["data/train/SUB_1.edf"], # Point to the healthy labeled file
    event_dict={'SSPE': 0, 'Background': 1, 'ELECT. ARTIFACT': 2}, # Updated to your exact labels
    patient_ids=[0]
)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# Initialize model to match exactly what you trained
model = RadermeckerTransformer(num_channels=19, d_model=64, nhead=4, num_layers=2, num_clinical_classes=3, num_patients=2).to(device)

# Load the frozen brain!
model.load_state_dict(torch.load("sspe_transformer_weights.pth", map_location=device, weights_only=True))
model.eval() # Turn off learning

# Data storage lists
extracted_data = []
latent_vectors = []

print("Extracting Clinical Metrics & AI Hidden States...")

with torch.no_grad():
    for idx, (x, y_clinical, y_patient) in enumerate(loader):
        x = x.to(device)
        
        # 1. Forward Pass
        # Our updated model returns 4 things: clinical, patient, latent_features, and None
        clinical_preds, _, latent_features, _ = model(x, alpha=0.0)
        
        # 2. Compute Signal Energy 
        raw_signal = x.cpu().numpy()[0] # Shape: (19, time_steps)
        signal_energy = np.sum(raw_signal ** 2) / raw_signal.shape[1] 
        
        # 3. Compute Dominant Frequency (Cz channel is index 9)
        freqs, psd = welch(raw_signal[9], fs=256, nperseg=256)
        dominant_frequency = freqs[np.argmax(psd)]
        
        # 4. Compute Temporal Chaos (Sequence Variance)
        # We calculate variance of the latent feature vector
        sequence_variance = torch.var(latent_features).item()
        
        # Store the latent feature vector for t-SNE
        latent_vectors.append(latent_features.cpu().numpy()[0])
        
        # Compile row data
        extracted_data.append({
            "Window_ID": idx,
            "Patient_ID": y_patient.item(),
            "True_Label": y_clinical.item(),
            "Predicted_Prob_SSPE": torch.softmax(clinical_preds, dim=1)[0][0].item(), 
            "Signal_Energy_uV2": signal_energy,
            "Dominant_Freq_Hz": dominant_frequency,
            "Chaos_Index": sequence_variance
        })

# ==========================================
# Discovering Commonalities (t-SNE)
# ==========================================
print("Mapping Hidden Commonalities...")
latent_matrix = np.array(latent_vectors) 

# t-SNE compresses the 64-dimensional Transformer understanding into 2D coordinates
# Perplexity should be smaller than your number of samples (We have 131, so 30 is great)
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
latent_2d = tsne.fit_transform(latent_matrix)

# Append the 2D coordinates to our data
for i, row in enumerate(extracted_data):
    row["Latent_X"] = latent_2d[i, 0]
    row["Latent_Y"] = latent_2d[i, 1]

# ==========================================
# Export to CSV
# ==========================================
df = pd.DataFrame(extracted_data)
df.to_csv("sspe_latent_analysis.csv", index=False)
print("\nExtraction Complete! Data saved to 'sspe_latent_analysis.csv'")
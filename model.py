import torch
import torch.nn as nn
import math
from torch.autograd import Function

# ==============================================================================
# GRADIENT REVERSAL LAYER (For Adversarial Domain Adaptation)
# ==============================================================================
class GradientReversalLayer(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone() 

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

# ==============================================================================
# MULTISCALE 1D CNN FRONT-END (The "Eyes" - Morphology Extractor)
# ==============================================================================
class MultiscaleCNNBlock(nn.Module):
    def __init__(self, in_channels, d_model):
        super(MultiscaleCNNBlock, self).__init__()
        
        # We split the target d_model (e.g., 64) into 4 equal branches (16 filters each)
        branch_dim = d_model // 4 
        
        # Branch 1: Small Kernel (Hunts for the sharp, rapid SSPE onset spike)
        self.branch1 = nn.Conv1d(in_channels, branch_dim, kernel_size=15, padding=7)
        
        # Branch 2: Medium Kernel (Captures transitional morphologies)
        self.branch2 = nn.Conv1d(in_channels, branch_dim, kernel_size=31, padding=15)
        
        # Branch 3: Large Kernel (Hunts for the massive 1-3 Hz rolling delta wave)
        self.branch3 = nn.Conv1d(in_channels, branch_dim, kernel_size=63, padding=31)
        
        # Branch 4: Max Pooling + 1x1 Conv (Preserves the highest-energy local peaks)
        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, branch_dim, kernel_size=1)
        )
        
        # Project the concatenated branches back to exact d_model dimensions
        self.project = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.bn = nn.BatchNorm1d(d_model)
        self.relu = nn.ReLU()
        
        # Decimate the time sequence by a factor of 8 to feed the Transformer 
        # (e.g., 2048 samples becomes a highly dense 256-token sequence)
        self.pool = nn.MaxPool1d(kernel_size=8, stride=8)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        
        # Stack the different perspectives on top of each other
        out = torch.cat([b1, b2, b3, b4], dim=1)
        
        out = self.project(out)
        out = self.bn(out)
        out = self.relu(out)
        out = self.pool(out)
        return out

# ==============================================================================
# FEATURE LINEAR MODULATION (FiLM) LAYER
# ==============================================================================
class FiLMLayer(nn.Module):
    def __init__(self, cond_dim, feature_dim):
        super(FiLMLayer, self).__init__()
        # Two linear layers to generate the scaling (gamma) and shifting (beta) vectors
        self.gamma = nn.Linear(cond_dim, feature_dim)
        self.beta = nn.Linear(cond_dim, feature_dim)
        
        # Initialize weights to zero so that at the very start of training, 
        # the network acts as a standard Transformer before learning to modulate.
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x, condition):
        # condition is the 4D physics token
        # x is the [Batch, Seq, d_model] brainwave sequence
        
        # Generate the modulation parameters
        g = self.gamma(condition).unsqueeze(1)  # Shape: [Batch, 1, d_model]
        b = self.beta(condition).unsqueeze(1)   # Shape: [Batch, 1, d_model]
        
        # Modulate the sequence: y = (1 + gamma) * x + beta
        return (1.0 + g) * x + b

# ==============================================================================
# TRANSFORMER ARCHITECTURE (The "Brain" - Periodicity Extractor)
# ==============================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class RadermeckerTransformer(nn.Module):
    def __init__(self, num_channels, d_model, nhead, num_layers, num_clinical_classes, num_patients):
        super(RadermeckerTransformer, self).__init__()
        
        self.multiscale_cnn = MultiscaleCNNBlock(num_channels, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # -> UPGRADE: Replace the simple physics embedder with the FiLM conditioning block
        self.film_gating = FiLMLayer(cond_dim=4, feature_dim=d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, dropout=0.2, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Branch 1: The Main Objective (Diagnose SSPE)
        self.clinical_classifier = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, num_clinical_classes))
        
        # Branch 2: The Adversarial Objective (Forget Patient ID)
        self.patient_discriminator = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, num_patients))

    def forward(self, x, physics_features, alpha=1.0):
        # 1. Feature Extraction (Morphology & Scale)
        wave_tokens = self.multiscale_cnn(x)
        
        # Permute for Transformer: (Batch, Sequence_Length, Features)
        wave_tokens = wave_tokens.permute(0, 2, 1)
        
        # Apply Positional Encoding to the time-series brainwaves
        wave_tokens = self.pos_encoder(wave_tokens)
        
        # 2. -> FiLM GATING FUSION
        # Adaptively scale and shift the morphology tokens using the physical reality
        gated_sequence = self.film_gating(wave_tokens, physics_features)
        
        # 3. Transformer Processing (Periodicity & Global Context)
        encoded_sequence = self.transformer_encoder(gated_sequence)
        latent_features = torch.mean(encoded_sequence, dim=1) # Global Average Pooling over modulated shape
        
        # 4. Clinical Prediction (Normal Flow)
        clinical_preds = self.clinical_classifier(latent_features)
        
        # 5. Adversarial Patient Prediction (Reversed Flow)
        reversed_features = GradientReversalLayer.apply(latent_features, alpha)
        patient_preds = self.patient_discriminator(reversed_features)

        return clinical_preds, patient_preds, latent_features, None

if __name__ == "__main__":
    print("Initializing Multimodal FiLM-Gated Conformer test configuration...")
    
    # Simulated batch: 4 windows, 19 Standard EEG channels, 2048 time samples (8 seconds at 256Hz)
    simulated_eeg = torch.randn(4, 19, 2048)
    
    # Simulated physical features (Absolute V, SNR, Delta, Entropy) for the 4 windows
    simulated_physics = torch.randn(4, 4) 
    
    model = RadermeckerTransformer(num_channels=19, d_model=64, nhead=4, num_layers=2, num_clinical_classes=3, num_patients=2)
    
    c_preds, p_preds, features, _ = model(simulated_eeg, simulated_physics, alpha=0.5)
    
    print("Test forward pass successful!")
    print(f"Clinical Output Shape: {c_preds.shape} (Expected: 4 windows, 3 classes)")
    print(f"Patient Output Shape: {p_preds.shape} (Expected: 4 windows, 2 patients)")
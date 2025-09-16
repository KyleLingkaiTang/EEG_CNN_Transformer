import numpy as np
import torch
import torch.nn as nn
torch.cuda.empty_cache()

import math
import scipy
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, matthews_corrcoef

class EEGTransformer(nn.Module):
    def __init__(self, n_channels=125, grid_size=(5, 5), num_heads=5, num_layers=3, dim_feedforward=2048, dropout=0.5,
                 device=None):
        super(EEGTransformer, self).__init__()

        # Set device (GPU if available, otherwise CPU)
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.grid_size = grid_size
        self.n_channels = n_channels

        # 2D CNN layers for spatial feature extraction
        self.cnn_layers = nn.Sequential(
            # First conv block
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=2, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),
            # nn.MaxPool2d(2, 2),

            # Second conv block
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            # nn.MaxPool2d(2, 2),

            # Third conv block
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU()
        )

        # Calculate the size after CNN
        self.cnn_output_size = self._get_cnn_output_size()
        print(self.cnn_output_size)
        # Linear layer to match transformer input dimension
        self.cnn_projection = nn.Linear(2048, n_channels)

        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_channels,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        # Stack multiple encoder layers
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Positional encoding
        self.pos_encoder = PositionalEncoding(n_channels, dropout)

        # Classification head
        self.classification_head = nn.Sequential(
            nn.Linear(n_channels, 64),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # Move model to device
        self.to(self.device)

    def _get_cnn_output_size(self):
        # Helper function to calculate CNN output size
        h, w = self.grid_size
        # After two max pooling layers
        h, w = h // 4, w // 4
        return 25 * h * w

    def forward(self, x):
        batch_size, seq_len, n_channels, n_channels2 = x.shape

        # Reshape input to process each timepoint with 2D CNN
        # Check if number of channels matches grid size
        # assert n_channels == self.grid_size[0] * self.grid_size[1], \
        #     f"Number of channels ({n_channels}) must match grid size ({self.grid_size[0]}x{self.grid_size[1]})"

        # Reshape from [batch, seq_len, channels] -> [batch * seq_len, 1, height, width]
        x = x.reshape(batch_size * seq_len, 1, self.grid_size[0], self.grid_size[1])

        # Apply CNN
        x = self.cnn_layers(x)

        # Flatten CNN output and project to transformer dimension
        x = x.view(batch_size * seq_len, -1)
        x = self.cnn_projection(x)

        # Reshape back to transformer input shape [batch, seq_len, channels]
        x = x.view(batch_size, seq_len, -1)

        # Transformer processing
        x = self.pos_encoder(x)
        transformer_output = self.transformer_encoder(x)
        pooled_output = torch.mean(transformer_output, dim=1)
        logits = self.classification_head(pooled_output)
        return logits


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.5, max_len=200):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))  # [(d_model+1)//2]

        pe = torch.zeros(max_len, 1, d_model)  # [max_len, 1, d_model]

        # Calculate number of even and odd positions
        n_even = (d_model + 1) // 2  # For sine
        n_odd = d_model // 2  # For cosine

        # Create appropriate sized div_terms
        div_term_even = div_term[:n_even]  # For sine terms
        div_term_odd = div_term[:n_odd]  # For cosine terms

        pe[:, 0, 0::2] = torch.sin(position * div_term_even)
        pe[:, 0, 1::2] = torch.cos(position * div_term_odd)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


# Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
grid_size = (5, 5)
transformer = EEGTransformer(n_channels=125, grid_size=grid_size, device=device)

# Training parameters
n_epochs = 80
sequence_length = 5000

# Load and split data
# Data is prepared within MATLAB. Feature matrix should be a 4D matrix (n_sub * n_timepoints * 5 * 5 [10-20 setup with zero padding])
mat = scipy.io.loadmat('data.mat')
X = np.float32(mat['feat'])
y = np.float32(mat['label'])

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for i, (train_index, test_index) in enumerate(kf.split(X)):
    # Split data into train and validation sets (80-20 split)

    train_ind = np.where(np.isin(aug_ind, train_index) == True)
    X_train = X1[train_ind[0]]
    X_val = X[test_index]

    y_train = y1[train_ind[0]]
    y_val = y[test_index]

    # Create DataLoaders
    train_loader = DataLoader(list(zip(X_train, y_train)), shuffle=True, batch_size=16)
    val_loader = DataLoader(list(zip(X_val, y_val)), shuffle=False, batch_size=100)

    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001)
    # Training loop with validation
    for epoch in range(n_epochs):
        # Training phase
        transformer.train()

        train_loss = 0
        train_correct = 0
        train_total = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            predictions = transformer(X_batch)
            loss = criterion(predictions.squeeze(), y_batch.squeeze())
            loss.backward()
            optimizer.step()



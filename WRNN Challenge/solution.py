import os
import sys
import io
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import lightning as L

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Add project root folder to path for importing utils
sys.path.append(f"{CURRENT_DIR}/../..")

from my_utils import DataPoint, ScorerStepByStep


# -----------------------------------------------------
# 1. LSTM Model (same as training)
# -----------------------------------------------------
class LSTMModel(L.LightningModule):
    def __init__(self, input_size=32, hidden_size=64, num_layers=1,
                 lr=0.001, output_size=32):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lr = lr

        self.lstm = nn.LSTM(
            input_size,
            self.hidden_size,
            self.num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(self.hidden_size, output_size)

    def forward(self, x):
        batch = x.size(0)
        h0 = torch.zeros(self.num_layers, batch, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch, self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out)


# -----------------------------------------------------
# 2. Prediction Wrapper (submission requires NO ARGS)
# -----------------------------------------------------
class PredictionModel:
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "ultra_final_parameters.pt")

        # Load model safely even if file is inside a ZIP (non-seekable)
        with open(model_path, "rb") as f:
            buffer = io.BytesIO(f.read())

        state = torch.load(buffer, map_location="cpu")

        self.model = LSTMModel()
        self.model.load_state_dict(state)
        self.model.eval()

        # Internal state
        self.current_seq_ix = None
        self.sequence_history = []

    def predict(self, data_point: DataPoint) -> np.ndarray:

        # Reset for new sequence
        if data_point.seq_ix != self.current_seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.sequence_history = []

        # Build input sequence
        seq = self.sequence_history + [data_point.state]
        x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            out = self.model(x)

        # Only return prediction when required
        if not data_point.need_prediction:
            return None

        pred = out[0, -1, :].numpy()

        # Append AFTER prediction
        self.sequence_history.append(data_point.state)

        return pred


# -----------------------------------------------------
# 3. Local evaluation (ignored by submission server)
# -----------------------------------------------------
if __name__ == "__main__":
    test_file = f"{CURRENT_DIR}/../../datasets/train.parquet"

    model = PredictionModel()
    scorer = ScorerStepByStep(test_file)

    print("Evaluating LSTM Model...")
    results = scorer.score(model)

    print("\n================ Results ================")
    print(f"Mean R²: {results['mean_r2']:.6f}")
    print("=========================================")

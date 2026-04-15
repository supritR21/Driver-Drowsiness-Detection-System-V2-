import torch
from app.services.model_arch import DrowsinessBiLSTM

checkpoint = torch.load("../storage/models/drowsiness_bilstm.pt", map_location="cpu")
model = DrowsinessBiLSTM()
model.load_state_dict(checkpoint["model_state_dict"])
print("Model loaded successfully")
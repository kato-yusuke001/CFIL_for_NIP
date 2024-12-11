from CFIL_for_NIP.network import ABN
from CFIL_for_NIP.memory import ApproachMemory
import os

class CFIL:
    def __init__(self):
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu" 
        self.approach_model = ABN()
        self.approach_model.to(self.device)

        self.file_path = "CFIL_for_NIP\\train_data\\20240719150226"

    def loadTrainedModel(self):
        import torch
        self.approach_model.load_state_dict(torch.load(os.path.join(self.file_path, "approach_model_final.pth"), map_location=torch.device(self.device), weights_only=True))


if __name__ == "__main__":
    cfil = CFIL()
    cfil.loadTrainedModel()
    print("Model loaded")
    
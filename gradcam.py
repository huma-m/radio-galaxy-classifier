import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt


CLASS_NAMES = ["FRI", "FRII"]

class GradCAM_ECNN:
    def __init__(self, model):
        self.model = model

    def generate(self, img_tensor, class_idx=None):
        self.model.eval()

        output = self.model(img_tensor)
        probs = torch.softmax(output, dim=1)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        if self.model.feature_maps is None:
            raise RuntimeError("feature_maps is None — forward() may not have run")
        
        feature_maps = self.model.feature_maps

        gradients = torch.autograd.grad(
            outputs=output[:, class_idx],
            inputs=feature_maps,
            retain_graph=False,
        )[0]

        weights = gradients.mean(dim=(2, 3), keepdim=True)

        cam = (weights * feature_maps).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = cam.detach().squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam = cv2.resize(cam, (img_tensor.shape[-1], img_tensor.shape[-2]))
        self.model.feature_maps = None
        
        return cam, class_idx, probs.squeeze().detach().cpu().numpy()
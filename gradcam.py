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


def visualize_gradcam_ecnn(model, test_dataset, indices, device):
    gcam = GradCAM_ECNN(model)
    n = len(indices)
    fig, axes = plt.subplots(n, 3, figsize=(9, n * 3))

    for row, idx in enumerate(indices):
        img_tensor, true_label = test_dataset[idx]
        img_tensor = img_tensor.unsqueeze(0).to(device)
        img_tensor.requires_grad_(True)

        cam, pred_class, probs = gcam.generate(img_tensor)

        orig = img_tensor[0, 0].detach().cpu().numpy()
        orig_norm = (orig - orig.min()) / (orig.max() - orig.min() + 1e-8)
        orig_uint8 = (orig_norm * 255).astype(np.uint8)

        heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        orig_rgb = cv2.cvtColor(orig_uint8, cv2.COLOR_GRAY2RGB)
        overlay = cv2.addWeighted(orig_rgb, 0.5, heatmap_rgb, 0.5, 0)

        axes[row, 0].imshow(orig, cmap="inferno")
        axes[row, 0].set_title(f"True: {CLASS_NAMES[true_label]}")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(cam, cmap="jet")
        axes[row, 1].set_title(f"Pred: {CLASS_NAMES[pred_class]} ({probs[pred_class]:.1%})")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(overlay)
        axes[row, 2].set_title("Overlay")
        axes[row, 2].axis("off")

    plt.tight_layout()
    plt.savefig("visualisations/gradcam/gradcam_results.png", dpi=150, bbox_inches="tight")
    plt.show()
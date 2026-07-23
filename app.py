import gradio as gr
import torch
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from torchvision.transforms import transforms

from models import DNSteerableLeNet, VanillaLeNet
from gradcam import GradCAM_ECNN


CLASS_NAMES  = ["FRI", "FRII", "Compact", "Bent"]
IMSIZE       = 150
DEVICE       = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
DATAMEAN     = 0.0026
DATASTD      = 0.0328

MODEL_PATHS = {
    "DNSteerableLeNet (G-CNN)": "models/first_dnlenet.pt",
    "VanillaLeNet (baseline)": "models/first_lenet.pt",
}

CLASS_DESC = {
    "FRI": (
        "**FR-I** — Jets are brightest near the core and fade outward. "
        "Typically associated with lower radio luminosity."
    ),
    "FRII": (
        "**FR-II** — Bright hotspots at the ends of extended lobes with an edge-brightened morphology."
    ),
    "Compact": (
        "**Compact** — Radio emission is concentrated into a compact unresolved or barely resolved source with no prominent jet or lobe structure."
    ),
    "Bent": (
        "**Bent** — Radio jets or lobes are curved due to interactions with the surrounding environment, commonly seen in cluster galaxies."
    ),
}

# ── preprocessing (matches training pipeline) ──────────────────────────────
def preprocess(image):
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    image = Image.fromarray(image.astype(np.uint8))
    
    image = transforms.CenterCrop(IMSIZE)(image)
    image = transforms.Pad((0, 0, 1, 1))(image)
    image = transforms.ToTensor()(image)
    image = transforms.Normalize((DATAMEAN,), (DATASTD,))(image)

    return image.unsqueeze(0)


# ── model loader (cached) ──────────────────────────────────────────────────
_model_cache = {}

def load_model(model_name):
    if model_name in _model_cache:
        return _model_cache[model_name]
    path = MODEL_PATHS[model_name]
    if "DNSteerable" in model_name:
        print("Loading model...")
        model = DNSteerableLeNet(1, 4, IMSIZE + 1, kernel_size=5, N=16)
    else:
        model = VanillaLeNet(1, 4, IMSIZE + 1, kernel_size=5)

    state = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    _model_cache[model_name] = model
    return model

_cam_cache = {}

def get_gradcam(model_name):
    if model_name not in _cam_cache:
        _cam_cache[model_name] = GradCAM_ECNN(load_model(model_name))
    return _cam_cache[model_name]

# ── inference ──────────────────────────────────────────────────────────────
def predict(image, model_name):
    if image is None:
        return None, None, "Upload an image to classify."

    model = load_model(model_name)
    gcam = get_gradcam(model_name)

    img_tensor = preprocess(image).to(DEVICE)
    cam, pred_idx, probs = gcam.generate(img_tensor)

    # ── confidence output ──
    pred_name = CLASS_NAMES[pred_idx]
    conf = probs[pred_idx] * 100

    prediction_html = f"""
    <div class="prediction-card">
        <h3 class="section-title">Prediction</h3>
        <div class="prediction-name">{pred_name}</div>
        <hr>
        <h3 class="section-title">Confidence</h3>
        <div class="confidence-value">{conf:.1f}%</div>
        <div class="confidence-bar">
            <div class="confidence-fill" style="width:{conf:.1f}%"></div>
        </div>
        <div class="confidence-scale">
            <span>0%</span>
            <span>50%</span>
            <span>100%</span>
        </div>
    </div>
    """

    # ── overlay ──
    orig = img_tensor[0, 0].detach().cpu().numpy()
    orig = (orig - orig.min()) / (orig.max() - orig.min() + 1e-8)
    orig_uint8  = (orig * 255).astype(np.uint8)
    heatmap     = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    orig_rgb    = cv2.cvtColor(orig_uint8, cv2.COLOR_GRAY2RGB)
    overlay     = cv2.addWeighted(orig_rgb, 0.55, heatmap_rgb, 0.45, 0)

    # ── explanation ──
    EXPLANATIONS = {
        "FRI": ["/gradio_api/file=assets/fri1.png", "/gradio_api/file=assets/fri2.png"],
        "FRII": ["/gradio_api/file=assets/frii1.png", "/gradio_api/file=assets/frii2.png"],
        "Compact": ["/gradio_api/file=assets/compact1.png", "/gradio_api/file=assets/compact2.png"],
        "Bent": ["/gradio_api/file=assets/bent1.png", "/gradio_api/file=assets/bent2.png"],
    }
    images_html = f"""
<div style="
    display:flex;
    gap:12px;
    justify-content:center;
    align-items:center;
    flex-wrap:wrap;
">
    {
        "".join(
            f'<img src="{img}" style="width:120px; border-radius:8px;">'
            for img in EXPLANATIONS[pred_name]
        )
    }
</div>
"""
    explanation = (
        f"### Predicted: {pred_name} ({conf:.1f}% confidence)\n\n"
        f"{CLASS_DESC[pred_name]}\n\n"
        f"The heatmap shows which pixels most influenced this decision — "
        f"brighter regions had greater weight.\n\n"
        f"{images_html}"
    )
    return (
        gr.update(value=prediction_html, visible=True),
        Image.fromarray(overlay),
        gr.update(value=explanation, visible=True),
    )


# ── example images ─────────────────────────────────────────────────────────
EXAMPLES = {
    "FRI": ["assets/eg_fri1.png", "assets/eg_fri2.png"],
    "FRII": ["assets/eg_frii1.png", "assets/eg_frii2.png"],
    "COMPACT": ["assets/eg_compact1.png", "assets/eg_compact2.png"],
    "BENT": ["assets/eg_bent1.png", "assets/eg_bent2.png"],
}

# ── UI ─────────────────────────────────────────────────────────────────────
css = Path("style.css").read_text()

with gr.Blocks(fill_width=True,) as demo:

    # ==========================================================
    # NAVBAR
    # ==========================================================
    gr.HTML("""
    <div class="navbar">
        <div class="title">
            <h2 style="margin: 0px;">Radio Galaxy Morphology Classifier</h2>
        </div>

        <div class="nav-links">
            <a href="https://github.com/huma-m" target="_blank" style="display: inline-flex; align-items: center; gap: 8px; color: white; padding: 10px 20px; font-size: 16px; font-weight: 600; font-family: sans-serif; text-decoration: none;">
                <svg height="20" width="20" viewBox="0 0 16 16" fill="white"><path d="M8 0c4.42 0 8 3.58 8 8a8.013 8.013 0 0 1-5.45 7.59c-.4.08-.55-.17-.55-.38 0-.27.01-1.13.01-2.2 0-.75-.25-1.23-.54-1.48 1.78-.2 3.65-.88 3.65-3.95 0-.88-.31-1.59-.82-2.15.08-.2.36-1.02-.08-2.12 0 0-.67-.22-2.2.82A7.48 7.48 0 0 0 8 3c-.68 0-1.36.09-2 .27-1.53-1.03-2.2-.82-2.2-.82-.44 1.1-.16 1.92-.08 2.12-.51.56-.82 1.28-.82 2.15 0 3.06 1.86 3.75 3.64 3.95-.23.2-.44.55-.51 1.07-.46.21-1.61.55-2.33-.66-.15-.24-.6-.83-1.23-.82-.67.01-.27.38.01.53.34.19.73.9.82 1.13.16.45.68 1.31 2.69.94 0 .67.01 1.3.01 1.49 0 .21-.15.45-.55.38A7.995 7.995 0 0 1 0 8c0-4.42 3.58-8 8-8Z"/></svg>
                GitHub
            </a>
        </div>
    </div>
    """, elem_id="navbar-block")

    # ==========================================================
    # MAIN CONTENT
    # ==========================================================
    with gr.Row(equal_height=True):

        # ======================================================
        # LEFT PANEL
        # ======================================================
        with gr.Column(scale=1, elem_classes="card"):

            gr.Markdown("### Upload Radio Galaxy Image (PNG, JPG, JPEG)", elem_classes="title-text")
            image_input = gr.Image(
                type="numpy",
                image_mode="L",
                show_label=False,
                height=320,
            )

            gr.Markdown("### Select a Model")
            with gr.Row(equal_height=True, elem_id="model_row"):
                model_dropdown = gr.Dropdown(
                    choices=list(MODEL_PATHS.keys()),
                    value="DNSteerableLeNet (G-CNN)",
                    show_label=False,
                    allow_custom_value=False,
                    filterable=False,
                    elem_id="model_dropdown",
                    scale=4
                )

                classify_btn = gr.Button(
                    "Classify",
                    elem_id="classify-btn",
                    min_width=100,
                    scale=1
                )

            
            if EXAMPLES:
                gr.Markdown("### Example Images")
                with gr.Row():
                    gr.Examples(
                        examples=EXAMPLES["FRI"],
                        inputs=[image_input,],
                        examples_per_page=2,
                        label="FRI",
                        elem_id="examples"
                    )
                    gr.Examples(
                        examples=EXAMPLES["FRII"],
                        inputs=[image_input,],
                        examples_per_page=2,
                        label="FRII",
                        elem_id="examples"
                    )
                with gr.Row():
                    gr.Examples(
                        examples=EXAMPLES["COMPACT"],
                        inputs=[image_input,],
                        examples_per_page=2,
                        label="Compact",
                        elem_id="examples"
                    )
                    gr.Examples(
                        examples=EXAMPLES["BENT"],
                        inputs=[image_input,],
                        examples_per_page=2,
                        label="Bent",
                        elem_id="examples"
                    )

        # ======================================================
        # RIGHT PANEL
        # ======================================================
        with gr.Column(scale=2, elem_classes="card"):
            
            gr.Markdown("### Classification Result", elem_classes="title-text")
                
            with gr.Row(equal_height=True, elem_classes="pred-row"):
                overlay_output = gr.Image(
                    label="Grad-CAM",
                    height=320,
                    scale=2
                )
                gr.Markdown("""### Grad-CAM Explanation
                Brighter (yellow/red) regions had the greatest influence on the model's prediction, while darker (blue) regions contributed less. The model focuses on the brightest regions of the radio galaxy, which correspond to the core and lobes in FR-I and FR-II morphologies, respectively.
                """, elem_classes="gradcam-desc", scale=1)
                
            with gr.Row(equal_height=True, elem_classes="pred-row"):
                prediction_output = gr.HTML(elem_id="prediction-card", visible=False)
                explain_output = gr.Markdown(elem_id="result-md", visible=False)
    # ==========================================================
    # FOOTER
    # ==========================================================
    with gr.Column(elem_id="footer"):

        with gr.Accordion("Project Notes", open=False, elem_id="project-notes"):
            gr.Markdown("""
## About the Project

This application classifies radio galaxies into four morphological classes using deep learning:

- **FRI** – Core-brightened galaxies with jets that fade away from the center.
- **FRII** – Edge-brightened galaxies with prominent hotspots at the ends of their radio lobes.
- **Compact** – Small unresolved radio sources without extended jet structures.
- **Bent** – Galaxies whose jets are curved due to interactions with their surrounding environment.

---

## Dataset

The model was trained using the RadioGalaxyDataset, a curated dataset containing 2,158 grayscale radio galaxy images from the FIRST (Faint Images of the Radio Sky at Twenty-Centimeters) survey. 
The dataset combines expert-labelled sources from six published catalogues: MiraBest, Gendre, FRICAT (Capetti et al., 2017b), FR0CAT (Capetti et al., 2017a), Baldi et al. (2018), and Proctor. 
Each image is labelled as one of four radio galaxy morphologies: FRI, FRII, Compact, or Bent.

---

## Model

Multiple convolutional neural network architectures were evaluated for radio galaxy morphology classification, including **ResNet-18, DenseNet-121, VanillaLeNet,** and **DNSteerableLeNet,** a Group Equivariant CNN based on the work of **Scaife & Porter (2021)**. The final application uses **DNSteerableLeNet**, which achieved the best performance with **81% test accuracy** and a **Macro F1-score of 0.81**.

The effect of input image resolution was also investigated. Increasing the image size from **150×150** to **225×225** improved the performance of pretrained CNNs, with **ResNet-18** increasing from **69% → 78%** accuracy and **DenseNet-121** from **73% → 79%**. However, the equivariant models showed performance degradation at the higher resolution, indicating that rotationally equivariant feature extraction contributed more to performance than simply increasing input resolution. Therefore, the final model uses the **150×150** input resolution.

---

## Error Analysis

While the model performs well overall, some morphologies remain challenging.

Typical confusion occurs between:

- **Bent ↔ FRI**, as bent jets can closely resemble the edge-darkened jet structures of FR-I galaxies.
- **Bent ↔ FRII**, when curved lobes or asymmetric hotspots resemble distorted FR-II morphologies.

The Bent class is the most challenging to classify due to its high morphological variability and similarity to both FRI and FRII galaxies. These errors reflect the inherent complexity of radio galaxy morphology rather than simple model failures.
  
---
  
## References

**Dataset**
- RadioGalaxyDataset (Zenodo): https://zenodo.org/records/7351724

**Model**
- Scaife & Porter (2021), *Fanaroff–Riley Classification of Radio Galaxies Using Group Equivariant Convolutional Neural Networks:*
  https://arxiv.org/abs/2102.08252
""")

    # ==========================================================
    # EVENTS
    # ==========================================================
    classify_btn.click(
        fn=predict,
        inputs=[
            image_input,
            model_dropdown
        ],
        outputs=[
            prediction_output,
            overlay_output,
            explain_output
        ]
    )
demo.launch(css=css, footer_links=[], allowed_paths=["assets"])
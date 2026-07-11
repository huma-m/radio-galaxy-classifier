import gradio as gr
import torch
import numpy as np
import cv2
from PIL import Image
import os
from pathlib import Path
import sys
from torchvision.transforms import transforms

from models import DNSteerableLeNet, VanillaLeNet
from gradcam import GradCAM_ECNN


CLASS_NAMES  = ["FRI", "FRII"]
IMSIZE       = 150
DEVICE       = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
DATAMEAN     = 0.0026
DATASTD      = 0.0328

MODEL_PATHS = {
    "DNSteerableLeNet (G-CNN, recommended)": "models/final_crumb_123_dnlenet.pt",
    "VanillaLeNet (baseline)":               "models/final_crumb_123_lenet.pt",
}

CLASS_DESC = {
    "FRI": (
        "**FR-I** — jets brightest near the core, fading outward. "
        "Associated with lower radio luminosity and denser environments."
    ),
    "FRII": (
        "**FR-II** — bright hotspots at the ends of extended lobes, "
        "faint core. Higher luminosity, lobes terminate in distinct shock regions."
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
        model = DNSteerableLeNet(1, 2, IMSIZE + 1, kernel_size=5, N=16)
    else:
        model = VanillaLeNet(1, 2, IMSIZE + 1, kernel_size=5)

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

    full_name = {
        "FRI": "Fanaroff–Riley Type I",
        "FRII": "Fanaroff–Riley Type II",
    }

    prediction_html = f"""
    <div class="prediction-card">
        <h3 class="section-title">Prediction</h3>
        <div class="prediction-name">{pred_name}</div>
        <div class="prediction-type">{full_name[pred_name]}</div>
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
    explanation = (
        f"### Predicted: {pred_name} ({conf:.1f}% confidence)\n\n"
        f"{CLASS_DESC[pred_name]}\n\n"
        f"The heatmap shows which pixels most influenced this decision — "
        f"brighter regions had greater weight. "
        + (
            "The model focused on a single dominant brightness region, "
            "consistent with an FR-I core-brightened morphology."
            if pred_name == "FRI" else
            "The model attended to separated bright regions at lobe extremities, "
            "consistent with FR-II edge-brightened morphology."
        )
    )
    return (
        gr.update(value=prediction_html, visible=True),
        Image.fromarray(overlay),
        gr.update(value=explanation, visible=True),
    )


# ── example images ─────────────────────────────────────────────────────────
EXAMPLES_FRI = []
for f in ["examples/fri_1.png", "examples/fri_2.png",]:
    if os.path.exists(f):
        EXAMPLES_FRI.append([f])
EXAMPLES_FRII = []
for f in ["examples/frii_1.png", "examples/frii_2.png"]:
    if os.path.exists(f):
        EXAMPLES_FRII.append([f])



# ── UI ─────────────────────────────────────────────────────────────────────
css = Path("style.css").read_text()

with gr.Blocks(fill_width=True,) as demo:

    # ==========================================================
    # NAVBAR
    # ==========================================================
    gr.HTML("""
    <div class="navbar">
        <div class="title">
            <h2 style="margin: 0px;">Radio Galaxy Classifier</h2>
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
            model_dropdown = gr.Dropdown(
                choices=list(MODEL_PATHS.keys()),
                value="DNSteerableLeNet (G-CNN, recommended)",
                show_label=False,
                allow_custom_value=False,
                filterable=False,  
            )

            classify_btn = gr.Button(
                "Classify",
                elem_id="classify-btn",
            )

            
            if EXAMPLES_FRI and EXAMPLES_FRII:
                gr.Markdown("### Example Images")
                with gr.Row():
                    gr.Examples(
                        examples=EXAMPLES_FRI,
                        inputs=[image_input, model_dropdown],
                        examples_per_page=2,
                        label="FRI",
                        elem_id="examples"
                    )
                    gr.Examples(
                        examples=EXAMPLES_FRII,
                        inputs=[image_input, model_dropdown],
                        examples_per_page=2,
                        label="FRII",
                        elem_id="examples"
                    )

        # ======================================================
        # RIGHT PANEL
        # ======================================================
        with gr.Column(scale=2, elem_classes="card"):
            
            gr.Markdown("### Classification Result", elem_classes="title-text")
                
            with gr.Row(equal_height=True, elem_classes="pred-row"):
                with gr.Column(scale=2):
                    overlay_output = gr.Image(
                        label="Grad-CAM",
                        height=320,
                    )
                with gr.Column(scale=1):
                    gr.Markdown("""### Grad-CAM Explanation
                    Brighter (yellow/red) regions had the greatest influence on the model's prediction, while darker (blue) regions contributed less. The model focuses on the brightest regions of the radio galaxy, which correspond to the core and lobes in FR-I and FR-II morphologies, respectively.
                    """, elem_classes="gradcam-desc")
                
            with gr.Row(equal_height=True, elem_classes="pred-row"):
                prediction_output = gr.HTML(elem_id="prediction-card", visible=False)
                explain_output = gr.Markdown(elem_id="result-md", visible=False)
    # ==========================================================
    # FOOTER
    # ==========================================================
    with gr.Column(elem_id="footer"):

        with gr.Accordion("Project Notes", open=False, elem_id="project-notes"):
            gr.Markdown("""
## How This Came Together

I started this project wanting to do radio source **segmentation** on LoTSS DR3 data —
drawing masks around blobs of radio emission. I built a preprocessing pipeline, downloaded
~300 cutouts, generated RMS-threshold masks. It worked, technically. But then I asked myself
what the model would actually be learning that a simple `pixel > 4σ` threshold wasn't already
doing. The answer was: not much. Segmentation without expert pixel-level annotations is just
teaching a model to replicate a threshold.

So I pivoted to **morphology classification** — FR-I vs FR-II — which is a problem where
a model can genuinely learn something a threshold can't tell you.

---

## The Standard CNN Wall

I tried ResNet-18, DenseNet-121, EfficientNet-B0. All landed around **64–67% accuracy**,
barely above chance on a binary task. This confused me for a while. These are strong
architectures — why were they failing so badly on ~1600 training images?

The answer turned out to be rotational invariance. Radio galaxies have no preferred orientation
on the sky — an FR-II pointing left and an FR-II pointing right are the same object. A standard
CNN has to *learn* this from data, which means it needs to see every galaxy at every rotation.
With 1600 samples, it never gets enough coverage. The model memorises orientations instead of
learning morphology.

This is the kind of thing that feels obvious in hindsight but took me an embarrassingly long
time to figure out empirically.

---

## Switching to Group-Equivariant CNNs

I found Scaife & Porter (2021), who trained a **Group-Equivariant CNN** (G-CNN) on the
MiraBest dataset — the same type of data, the same task. Their architecture, DNSteerableLeNet,
encodes rotational and reflectional symmetry *mathematically* into the convolution layers using
the `e2cnn` library. The model never has to learn that rotations are equivalent — it already knows.

Switching to this architecture and the CRUMB dataset (a cleaned, cross-matched combination of
MiraBest, FR-DEEP, and AT17) brought accuracy up to ~73%, with notably **lower variance across
random seeds** (std 0.025 vs 0.053 for VanillaLeNet). The stability improvement matters as much
as the accuracy improvement — it means the model is learning something consistent rather than
fitting whatever rotations happened to land in a particular split.

---

## The Dataset Split Problem

CRUMB ships with an official `test_batch`. When I evaluated on it, accuracy dropped to ~53% —
essentially random. Swapping val and test brought it back to ~80%. Something was wrong with the
official split.

After a lot of debugging (pixel statistics, label distributions, per-source-dataset breakdowns,
even testing if the labels were globally flipped — they weren't), I found the answer visually:
the official `test_batch` skews toward **morphologically compact sources**, where FR structure
simply isn't visible at 150×150 pixels. The model learned real morphological features but was
being tested on images where those features don't appear.

The fix was to combine all data and re-split with stratification by label × parent-dataset
origin. This distributes compact and extended sources evenly across train/val/test, giving a
more honest evaluation. 70–75% accuracy on this harder, full-distribution split is the number
I report.

---

## Current State

The model works. GradCAM shows it's attending to genuine morphological structure — cores,
jets, lobe separation — rather than noise or image artifacts. The failure cases are
scientifically interpretable: the model anchors on peak brightness and under-weights faint
secondary emission, which explains why visually ambiguous sources (the ones human experts also
disagree on) are where errors concentrate.

73% on the full distribution isn't 94%. The published 94% numbers use Confident-label subsets —
the easier half of the data. I think reporting honest numbers on the harder full distribution is
more useful than cherry-picking a subset to match a benchmark.

---

## What I'd Try Next

**Fine-grained subtype classification.** CRUMB's complete labels encode morphological subtypes
within FRI: Standard, Wide-Angle Tail (WAT), Head-Tail (HT). These are physically distinct
objects. A classifier that distinguishes WAT from standard FRI would be more scientifically
useful than the binary FRI/FRII split — and nobody has published this on CRUMB specifically.

**Confidence calibration.** The model is sometimes highly confident on wrong predictions.
Temperature scaling or conformal prediction would give better-calibrated uncertainty estimates,
which matters if this were ever used in an actual pipeline.

**Cross-survey generalisation.** CRUMB is all FIRST survey data. LoTSS (LOFAR) sees extended
emission that FIRST misses, which means a model trained on FIRST may systematically
misclassify sources that look different at 144 MHz. Testing cross-survey transfer would be
an honest stress test of what the model has actually learned.

**Better handling of the Uncertain label.** The Uncertain sources in MiraBest aren't mislabeled
— they're genuinely ambiguous, sometimes because the morphology is unclear and sometimes because
the source is at a redshift where the jets aren't resolved. Treating them the same as Confident
sources during training adds label noise. A noise-robust loss function or a reject option for
low-confidence predictions might handle this more honestly.
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
# with gr.Blocks(title="Radio Galaxy Classifier") as demo:

#     gr.Markdown("# Radio Galaxy Morphology Classifier", elem_id="title")
#     gr.Markdown(
#         "FR-I vs FR-II classification using a Group-Equivariant CNN "
#         "trained on the [CRUMB dataset](http://www.jb.man.ac.uk/research/MiraBest/CRUMB/) "
#         "— 2,100 FIRST survey radio galaxies.",
#         elem_id="subtitle"
#     )

#     # ── about section ────────────────────────────────────────────────────
#     with gr.Accordion("About this project", open=False):
#         gr.Markdown("""
# **Architecture**
# Two models are available:
# - *DNSteerableLeNet* — D8-equivariant G-CNN (Scaife & Porter 2021). Encodes rotational and
#   reflectional symmetry directly into the convolution layers. Radio galaxies have no preferred
#   sky orientation, so equivariance is a natural inductive bias.
# - *VanillaLeNet* — standard LeNet without equivariance, used as a baseline.

# **Dataset — CRUMB**
# 2,100 FIRST survey radio galaxies, cross-matched from MiraBest, FR-DEEP, AT17 and MiraBest
# Hybrid. Re-split with stratified 70/15/15 train/val/test to ensure uniform morphological
# difficulty distribution across splits.

# **Results** (3 seeds, full binary CRUMB)

# | Model | Mean Acc | Std |
# |---|---|---|
# | ResNet-18 / DenseNet-121 / EfficientNet-B0 | 0.64–0.67 | high |
# | VanillaLeNet | 0.722 | 0.053 |
# | DNSteerableLeNet | 0.730 | 0.025 |

# Pretrained ImageNet CNNs underperform small domain-appropriate architectures on single-channel
# 150×150 radio images. The G-CNN's main advantage is **stability** (lower variance across seeds),
# not raw accuracy — consistent with equivariance reducing sensitivity to random split composition.

# **GradCAM**
# Hooks onto the `GroupPooling` layer of DNSteerableLeNet (last spatial feature map before
# flattening). Reveals the model anchors on peak brightness and can under-weight faint secondary
# emission — explaining FRI/FRII confusion on visually ambiguous sources.

# **References**
# - Fanaroff & Riley (1974) — original FR classification
# - Scaife & Porter (2021) — G-CNN for radio galaxy classification
# - Porter & Scaife (2023) — MiraBest / CRUMB dataset paper
#         """)

# Radio Galaxy Morphology Classifier

Automated classification of radio galaxies into four morphological types — **FR-I, FR-II, Compact, and Bent** — using Group-Equivariant Convolutional Neural Networks trained on VLA FIRST survey images.

**Live demo:** [Hugging Face Spaces](https://huggingface.co/spaces/huma-03/radio-galaxy-classification)

---

## What This Does

Radio galaxies are classified by where their radio emission peaks along the jet structure:

| Class | Description |
|---|---|
| **FR-I** | Jets brightest near the core, fading outward |
| **FR-II** | Bright hotspots at lobe edges, fainter core |
| **Compact** | Unresolved point-like source |
| **Bent** | Curved jets deflected by cluster environment |

Manual classification at survey scale is infeasible as modern radio surveys contain hundreds of thousands of sources. This project automates the task and provides GradCAM visualisations to verify the model is learning genuine morphological features.

---

## Results

All models trained on 2,158 FIRST survey images (150×150 grayscale).

### 150×150 (native resolution)

| Model | Accuracy | Notes |
|---|---|---|
| EfficientNet-B0 | 63% | ImageNet pretraining hurts on single-channel radio data |
| ResNet-18 | 69% | |
| DenseNet-121 | 73% | Best pretrained baseline at this resolution |
| VanillaLeNet | 80% | No pretraining, domain-appropriate architecture |
| **DNSteerableLeNet** | **81%** | D8-equivariant, best accuracy and stability |

### 225×225 (pretrained CNN optimised resolution)

| Model | Accuracy | Notes |
|---|---|---|
| EfficientNet-B0 | 67% | +4% vs 150px |
| ResNet-18 | 78% | +9% vs 150px |
| DenseNet-121 | 79% | +6% vs 150px |

**Key finding:** DNSteerableLeNet at 150×150 outperforms all pretrained CNNs even at their optimal 225×225 resolution, despite having fewer parameters and no ImageNet pretraining. The architectural prior (rotational equivariance) outperforms data volume and transfer learning on this task.

**Bent galaxies** are the hardest class across all models — consistently confused with FR-I and FR-II but never with Compact. This is physically sensible: Bent sources are FR galaxies whose jets have been deflected by their environment, making the morphological boundary genuinely ambiguous.

---

## Why Group-Equivariant CNNs

Radio galaxies have no preferred orientation on the sky. A standard CNN must learn rotational invariance from data and with ~1500 training images, it memorises orientations instead of learning morphology, producing a hard ceiling around 63–73%.

DNSteerableLeNet (Scaife & Porter, 2021) encodes **D16 symmetry** (16-fold rotation × reflection) directly into its convolution layers via the `e2cnn` library. The model mathematically treats rotated versions of the same galaxy as identical, without needing to learn this from examples.

---

## GradCAM

Standard GradCAM backward hooks are incompatible with `e2cnn`'s `GeometricTensor` outputs, hooks fire but gradients are `None`. 

The fix: store intermediate feature maps directly during the forward pass (post-GroupPooling, after `.tensor` unwrap) and compute gradients through standard PyTorch autograd without hooks.

```python
# In DNSteerableLeNet.forward()
x = self.gpool(x)
x = x.tensor          # unwrap GeometricTensor → plain tensor
self.feature_maps = x  # store for GradCAM
x = x.view(x.size()[0], -1)
```

GradCAM is used to visualize the regions of each radio galaxy that contribute most to the model's prediction, helping verify that the network focuses on meaningful morphological structures such as jets, lobe separation, core brightness gradients, rather than image artifacts or background noise.

---

## References

**Dataset**
- RadioGalaxyDataset (Zenodo): https://zenodo.org/records/7351724

**Models**
- Scaife & Porter (2021), *Fanaroff–Riley Classification of Radio Galaxies Using Group Equivariant Convolutional Neural Networks:*
  https://arxiv.org/abs/2102.08252

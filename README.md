# ArtExtract — Task 1: Multi-Task Art Classification
## CNN-ReNet Architecture

**Dataset**: WikiArt via Kaggle (`steubk/wikiart`) — 80,042 paintings across 27 styles, 720 artists, 10 genres  
**Architecture**: EfficientNet-B0 backbone + ReNet spatial recurrence (Visin et al., 2015) + multi-task classification heads

### Overview
This notebook implements a convolutional-recurrent architecture for simultaneously classifying paintings by **style**, **artist**, and **genre**. 
The approach I used involves a ReNet model using BiGRUs.
```
EfficientNet-B0 backbone
        ↓
  (B, 1280, 7, 7) feature map
        ↓
  Project channels: 1280 → 256 via 1×1 conv   ← cheaper sequence to process
  (B, 256, 7, 7)
        ↓
  Horizontal pass: reshape to (B*7, 7, 256)
  BiGRU(input=256, hidden=256) → (B*7, 7, 512)
  Reshape back to (B, 7, 7, 512)
        ↓
  Vertical pass: reshape to (B*7, 7, 512)
  BiGRU(input=512, hidden=256) → (B*7, 7, 512)
  Reshape back to (B, 7, 7, 512)
        ↓
  Global average pool → (B, 512)
        ↓
  Shared embedding (dropout + linear) → (B, 256)
        ↓
  ┌──────────────────────────────────────┐
  │  Style head:  Linear(256 → 27)          │
  │  Artist head: Linear(256 → 726)         │
  │  Genre head:  Linear(256 → 10)          │
  └──────────────────────────────────────┘

```
**Files** <br>
Due to Kaggle session constraints, training and analysis are split across two notebooks. 
- `cnn_rnn.ipynb`: contains the full training run with curves and epoch logs.
- `cnn-rnn_analysis.ipynb`: contains evaluation metrics, confusion matrices, and outlier analysis using the saved checkpoint

**References**:
- Visin et al. (2015). *ReNet: A Recurrent Neural Network Based Alternative to Convolutional Networks*. arXiv:1505.00393
- Tan & Le (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*. ICML 2019
- Cho et al. (2014). *Learning Phrase Representations using RNN Encoder-Decoder*. arXiv:1406.1078



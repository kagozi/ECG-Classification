# AI ECG classification

> **Note:** All the commands are based on a Unix based system such as _OSX_.
> For a different system look for similar commands for it.


## Setup

We are using Python version 3.11.9

```bash
$ python --version
Python 3.11.9
```

### Python virtual environment

**Create** a virtual environment:

```bash
$ python3 -m venv .ecg
```

`.ecg` is the name of the folder that would contain the virtual environment.

**Activate** the virtual environment:

```bash
$ source .ecg/bin/activate
```

**Windows**
```bash
source .ecg/Scripts/activate
```
### Requirements

```bash
(.venv) $ pip install -r requirements.txt
```

```
    # ============================================================================
    # TIPS FOR HIGH SCORES WITH FUSION
    # ============================================================================
    1. Data Strategy:
    ✓ Scalograms capture magnitude/energy information
    ✓ Phasograms capture timing/synchronization information
    ✓ Fusion combines complementary features
    ✓ Try both composite (all leads) and Lead II variants

    2. Fusion Strategy:
    - Early Fusion: Better when modalities are highly correlated
    - Late Fusion: Better when modalities provide independent information
    - Start with early fusion (faster, fewer parameters)

    3. Model Selection:
    - Swin Transformers: Best for capturing long-range patterns
    - swin_base: Good balance of performance and speed
    - swin_large: Best performance but slower
    - swin_small/tiny: For faster experiments

    4. Training Best Practices:
    ✓ Lower learning rate for fusion models (3e-5 to 5e-5)
    ✓ More epochs may help (30-40)
    ✓ Watch for overfitting with 6-channel inputs
    ✓ Consider gradient accumulation for stability

    5. Evaluation:
    ✓ Always compare fusion vs single modality
    ✓ Check if fusion actually improves scores
    ✓ Some classes may benefit more from fusion
    ✓ Ensemble fusion + single modality models
    ✓ Try weighted ensemble based on validation performance

    6. Ensemble Strategies:
    - Simple Average: Equal weight to all models
    - Top-K Average: Only best performing models
    - Weighted Average: Weight by validation F1 score
    - Diversity matters: Mix CNNs + Transformers, scalograms + phasograms

    7. Expected Performance Hierarchy:
    Single Modality (3ch) < Early Fusion (6ch) < Late Fusion (6ch) < Ensemble of Best
    
    8. Typical Improvements:
    - Fusion over single: +1-3% F1
    - Ensemble over best single: +2-5% F1
    - Weighted ensemble: +0.5-1% over simple average

    9. OPTIMIZATION:
    - Use test-time augmentation (use_tta=True)
    - Try ensemble of best models
    - Experiment with different learning rates
    - Adjust batch size based on GPU memory

    10. ANALYSIS:
    - Review training curves for overfitting
    - Analyze per-class performance
    - Check which modality helps which class
    - Consider class-specific thresholding
    """
```

# preprocessing.py
import wfdb
import os
import numpy as np
from scipy.signal import butter, lfilter
import tqdm
from constants import TARGET_FS, SIGNAL_LEN_SECONDS, LOWCUT, HIGHCUT, OUTPUT_PATH, IMAGE_SIZE, TARGET_FS
from scalogram_phasogram import ScalogramGenerator, generate_and_save_scalograms, LeadIIGenerator, generate_and_save_lead2_scalograms, generate_and_save_composite_phasograms, generate_and_save_lead2_phasograms
from preprocessing.create_superclass import create_superclass_labels
from constants import DATA_PATH

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, fs, order=5):
    b, a = butter_bandpass(LOWCUT, HIGHCUT, fs, order=order)
    y = lfilter(b, a, data)
    return y

def preprocess_signals(signals):
    """Preprocesses 10-second ECG signals at 100Hz by filtering and normalizing."""
    if signals.ndim != 3 or signals.shape[0] == 0:
        raise ValueError(f"Invalid input signals shape: {signals.shape}")
    filtered_signals = np.zeros_like(signals)
    for i in range(signals.shape[0]):
        for j in range(signals.shape[1]):
            filtered_signals[i, j, :] = bandpass_filter(signals[i, j, :], fs=TARGET_FS)
    max_val = np.abs(filtered_signals).max(axis=(1, 2), keepdims=True)
    max_val[max_val == 0] = 1
    scaled_signals = filtered_signals / max_val
    return scaled_signals

def load_raw_signals(df, path):
    """Loads raw ECG signals without preprocessing."""
    if df.empty:
        return np.array([]).reshape(0, 12, SIGNAL_LEN_SECONDS * TARGET_FS)

    raw_signals = []
    for f in tqdm(df.filename_lr, desc=f"Loading {len(df)} signals"):
        signal, _ = wfdb.rdsamp(os.path.join(path, f))
        raw_signals.append(signal.T)

    raw_signals = np.array(raw_signals)
    return raw_signals



# ============================================================================
# USAGE EXAMPLE
# ============================================================================
if __name__ == '__main__':
    data, superclass_labels, mlb = create_superclass_labels(DATA_PATH, weight_threshold=0.5, min_count=10)
    category_counts = data['superclass_labels'].value_counts()
    print(category_counts)

    raw_signals = load_raw_signals(data, DATA_PATH)
    processed_signals = preprocess_signals(raw_signals)

    # 1. Generate RGB composite scalograms
    print("\n" + "="*80)
    print("GENERATING RGB COMPOSITE SCALOGRAMS")
    print("="*80 + "\n")
    SCALOGRAM_DIR = OUTPUT_PATH + '/composite_scalograms'
    generator = ScalogramGenerator(fs=TARGET_FS, image_size=IMAGE_SIZE)
    scalogram_paths = generate_and_save_scalograms(
        data, 
        processed_signals, 
        SCALOGRAM_DIR, 
        generator
    )

    # 2. Generate RGB composite phasograms
    print("\n" + "="*80)
    print("GENERATING RGB COMPOSITE PHASOGRAMS")
    print("="*80 + "\n")
    COMPOSITE_PHASOGRAM_DIR = OUTPUT_PATH + '/composite_phasograms'
    composite_phasogram_paths = generate_and_save_composite_phasograms(
        data, 
        processed_signals, 
        COMPOSITE_PHASOGRAM_DIR, 
        generator
    )

    # 3. Generate Lead II scalograms
    print("\n" + "="*80)
    print("GENERATING LEAD II SCALOGRAMS")
    print("="*80 + "\n")
    LEAD2_SCALOGRAM_DIR = OUTPUT_PATH + '/lead2_scalograms'
    lead2_generator = LeadIIGenerator(fs=TARGET_FS, image_size=IMAGE_SIZE)
    lead2_scalogram_paths = generate_and_save_lead2_scalograms(
        data, 
        processed_signals, 
        LEAD2_SCALOGRAM_DIR, 
        lead2_generator
    )

    # 4. Generate Lead II phasograms
    print("\n" + "="*80)
    print("GENERATING LEAD II PHASOGRAMS")
    print("="*80 + "\n")
    LEAD2_PHASOGRAM_DIR = OUTPUT_PATH + '/lead2_phasograms'
    lead2_phasogram_paths = generate_and_save_lead2_phasograms(
        data, 
        processed_signals, 
        LEAD2_PHASOGRAM_DIR, 
        lead2_generator
    )
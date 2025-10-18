import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pywt
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import os

# ============================================================================
# SCALOGRAM GENERATOR - CWT + RGB Composite
# ============================================================================

class ScalogramGenerator:
    """
    CWT-based scalogram generator for ECG signals.
    Uses Morlet wavelet with RGB composite layout.
    """
    
    def __init__(self, fs=100, image_size=224):
        self.fs = fs
        self.image_size = image_size
        
    def cwt_scalogram(self, signal, wavelet='cmor2.0-1.0'):
        """
        Continuous Wavelet Transform scalogram with Complex Morlet wavelet.
        
        Args:
            signal: 1D ECG signal
            wavelet: Wavelet type (default: 'cmor2.0-1.0' - optimized for ECG)
        
        Returns:
            scalogram: 2D scalogram array
        """
        # Optimized scales for ECG (0.5-50 Hz range)
        scales = np.arange(1, 128)
        
        coefficients, frequencies = pywt.cwt(signal, scales, wavelet, 
                                              sampling_period=1/self.fs)
        
        # Convert to power (magnitude squared)
        scalogram = np.abs(coefficients) ** 2
        
        # Log transform for better visualization
        scalogram = np.log1p(scalogram)
        
        return scalogram
    
    def cwt_phasogram(self, signal, wavelet='cmor2.0-1.0'):
        """
        Generate phasogram (phase information) from CWT coefficients.
        
        Args:
            signal: 1D ECG signal
            wavelet: Wavelet type (default: 'cmor2.0-1.0' - optimized for ECG)
        
        Returns:
            phasogram: 2D phase array
        """
        # Optimized scales for ECG
        scales = np.arange(1, 128)
        
        coefficients, frequencies = pywt.cwt(signal, scales, wavelet, 
                                              sampling_period=1/self.fs)
        
        # Extract phase information
        phasogram = np.angle(coefficients)
        
        return phasogram
    
    def generate_rgb_composite(self, signals_12lead):
        """
        Generate 3-channel RGB composite scalogram from 12-lead ECG.
        
        Args:
            signals_12lead: (12, time_steps) array
        
        Returns:
            image: RGB image array (image_size, image_size, 3) as uint8
        """
        # Group leads into 3 channels
        channel_groups = [
            [0, 1, 2],              # Limb leads (I, II, III) -> R
            [3, 4, 5],              # Augmented leads (aVR, aVL, aVF) -> G
            [6, 7, 8, 9, 10, 11]    # Chest leads (V1-V6) -> B
        ]
        
        channels = []
        for group in channel_groups:
            # Average signals in group
            avg_signal = np.mean(signals_12lead[group], axis=0)
            
            # Generate scalogram using cmor2.0-1.0 for better ECG analysis
            scalo = self.cwt_scalogram(avg_signal, wavelet='cmor2.0-1.0')
            
            # Normalize to 0-1
            scalo = (scalo - scalo.min()) / (scalo.max() - scalo.min() + 1e-8)
            channels.append(scalo)
        
        # Resize all channels to target size
        target_shape = (self.image_size, self.image_size)
        channels = [cv2.resize(c, target_shape) for c in channels]
        
        # Stack as RGB and convert to uint8
        image = np.stack(channels, axis=-1)
        image = (image * 255).astype(np.uint8)
        
        return image
    
    def generate_rgb_composite_phasogram(self, signals_12lead):
        """
        Generate 3-channel RGB composite phasogram from 12-lead ECG.
        
        Args:
            signals_12lead: (12, time_steps) array
        
        Returns:
            image: RGB phasogram image array (image_size, image_size, 3) as uint8
        """
        # Group leads into 3 channels
        channel_groups = [
            [0, 1, 2],              # Limb leads (I, II, III) -> R
            [3, 4, 5],              # Augmented leads (aVR, aVL, aVF) -> G
            [6, 7, 8, 9, 10, 11]    # Chest leads (V1-V6) -> B
        ]
        
        channels = []
        for group in channel_groups:
            # Average signals in group
            avg_signal = np.mean(signals_12lead[group], axis=0)
            
            # Generate phasogram using cmor2.0-1.0 for better ECG analysis
            phaso = self.cwt_phasogram(avg_signal, wavelet='cmor2.0-1.0')
            
            # Normalize phase from [-pi, pi] to [0, 1]
            phaso = (phaso + np.pi) / (2 * np.pi)
            channels.append(phaso)
        
        # Resize all channels to target size
        target_shape = (self.image_size, self.image_size)
        channels = [cv2.resize(c, target_shape) for c in channels]
        
        # Stack as RGB and convert to uint8
        image = np.stack(channels, axis=-1)
        image = (image * 255).astype(np.uint8)
        
        return image


# ============================================================================
# LEAD II GENERATOR - Scalograms and Phasograms
# ============================================================================

class LeadIIGenerator:
    """
    Generator for Lead II scalograms and phasograms.
    """
    
    def __init__(self, fs=100, image_size=224):
        self.fs = fs
        self.image_size = image_size
        
    def cwt_scalogram(self, signal, wavelet='cmor2.0-1.0'):
        """
        Continuous Wavelet Transform scalogram with Complex Morlet wavelet.
        
        Args:
            signal: 1D ECG signal
            wavelet: Wavelet type (default: 'cmor2.0-1.0' - optimized for ECG)
        
        Returns:
            scalogram: 2D scalogram array
        """
        # Optimized scales for ECG (0.5-50 Hz range)
        scales = np.arange(1, 128)
        
        coefficients, frequencies = pywt.cwt(signal, scales, wavelet, 
                                              sampling_period=1/self.fs)
        
        # Convert to power (magnitude squared)
        scalogram = np.abs(coefficients) ** 2
        
        # Log transform for better visualization
        scalogram = np.log1p(scalogram)
        
        return scalogram
    
    def cwt_phasogram(self, signal, wavelet='cmor2.0-1.0'):
        """
        Generate phasogram (phase information) from CWT coefficients.
        
        Args:
            signal: 1D ECG signal
            wavelet: Wavelet type (default: 'cmor2.0-1.0' - optimized for ECG)
        
        Returns:
            phasogram: 2D phase array
        """
        # Optimized scales for ECG
        scales = np.arange(1, 128)
        
        coefficients, frequencies = pywt.cwt(signal, scales, wavelet, 
                                              sampling_period=1/self.fs)
        
        # Extract phase information
        phasogram = np.angle(coefficients)
        
        return phasogram
    
    def generate_lead2_scalogram(self, signals_12lead):
        """
        Generate scalogram from Lead II (index 1).
        
        Args:
            signals_12lead: (12, time_steps) array
        
        Returns:
            image: Grayscale scalogram as RGB (image_size, image_size, 3) uint8
        """
        # Extract Lead II (index 1)
        lead2_signal = signals_12lead[1]
        
        # Generate scalogram using cmor2.0-1.0 for better ECG analysis
        scalo = self.cwt_scalogram(lead2_signal, wavelet='cmor2.0-1.0')
        
        # Normalize to 0-1
        scalo = (scalo - scalo.min()) / (scalo.max() - scalo.min() + 1e-8)
        
        # Resize to target size
        scalo = cv2.resize(scalo, (self.image_size, self.image_size))
        
        # Convert to RGB (grayscale replicated across channels)
        image = np.stack([scalo, scalo, scalo], axis=-1)
        image = (image * 255).astype(np.uint8)
        
        return image
    
    def generate_lead2_phasogram(self, signals_12lead):
        """
        Generate phasogram from Lead II (index 1).
        
        Args:
            signals_12lead: (12, time_steps) array
        
        Returns:
            image: Phase image as RGB (image_size, image_size, 3) uint8
        """
        # Extract Lead II (index 1)
        lead2_signal = signals_12lead[1]
        
        # Generate phasogram using cmor2.0-1.0 for better ECG analysis
        phaso = self.cwt_phasogram(lead2_signal, wavelet='cmor2.0-1.0')
        
        # Normalize phase from [-pi, pi] to [0, 1]
        phaso = (phaso + np.pi) / (2 * np.pi)
        
        # Resize to target size
        phaso = cv2.resize(phaso, (self.image_size, self.image_size))
        
        # Convert to RGB (grayscale replicated across channels)
        image = np.stack([phaso, phaso, phaso], axis=-1)
        image = (image * 255).astype(np.uint8)
        
        return image


# ============================================================================
# GENERATION AND SAVING FUNCTIONS
# ============================================================================

def generate_and_save_scalograms(data, signals, output_dir, generator):
    """
    Generate RGB composite scalograms and save to disk as .npy files.
    
    Args:
        data: DataFrame with ECG metadata (must have index as ecg_id)
        signals: (N, 12, time_steps) array of preprocessed ECG signals
        output_dir: Directory to save scalograms
        generator: ScalogramGenerator instance
    
    Returns:
        scalogram_paths: List of file paths to saved scalograms
    """
    os.makedirs(output_dir, exist_ok=True)
    
    scalogram_paths = []
    ecg_ids = data.index.values
    
    print(f"Generating and saving {len(signals)} RGB composite scalograms to {output_dir}")
    
    for i, ecg_id in enumerate(tqdm(ecg_ids, desc="Generating RGB scalograms")):
        scalogram = generator.generate_rgb_composite(signals[i])
        scalogram_path = os.path.join(output_dir, f"{ecg_id}_scalogram.npy")
        np.save(scalogram_path, scalogram)
        scalogram_paths.append(scalogram_path)
    
    print(f"Saved {len(scalogram_paths)} RGB composite scalograms successfully!")
    return scalogram_paths


def generate_and_save_composite_phasograms(data, signals, output_dir, generator):
    """
    Generate RGB composite phasograms and save to disk as .npy files.
    
    Args:
        data: DataFrame with ECG metadata (must have index as ecg_id)
        signals: (N, 12, time_steps) array of preprocessed ECG signals
        output_dir: Directory to save phasograms
        generator: ScalogramGenerator instance
    
    Returns:
        phasogram_paths: List of file paths to saved phasograms
    """
    os.makedirs(output_dir, exist_ok=True)
    
    phasogram_paths = []
    ecg_ids = data.index.values
    
    print(f"Generating and saving {len(signals)} RGB composite phasograms to {output_dir}")
    
    for i, ecg_id in enumerate(tqdm(ecg_ids, desc="Generating RGB phasograms")):
        phasogram = generator.generate_rgb_composite_phasogram(signals[i])
        phasogram_path = os.path.join(output_dir, f"{ecg_id}_phasogram.npy")
        np.save(phasogram_path, phasogram)
        phasogram_paths.append(phasogram_path)
    
    print(f"Saved {len(phasogram_paths)} RGB composite phasograms successfully!")
    return phasogram_paths


def generate_and_save_lead2_scalograms(data, signals, output_dir, generator):
    """
    Generate Lead II scalograms and save to disk as .npy files.
    
    Args:
        data: DataFrame with ECG metadata (must have index as ecg_id)
        signals: (N, 12, time_steps) array of preprocessed ECG signals
        output_dir: Directory to save Lead II scalograms
        generator: LeadIIGenerator instance
    
    Returns:
        scalogram_paths: List of file paths to saved scalograms
    """
    os.makedirs(output_dir, exist_ok=True)
    
    scalogram_paths = []
    ecg_ids = data.index.values
    
    print(f"Generating and saving {len(signals)} Lead II scalograms to {output_dir}")
    
    for i, ecg_id in enumerate(tqdm(ecg_ids, desc="Generating Lead II scalograms")):
        scalogram = generator.generate_lead2_scalogram(signals[i])
        scalogram_path = os.path.join(output_dir, f"{ecg_id}_lead2_scalogram.npy")
        np.save(scalogram_path, scalogram)
        scalogram_paths.append(scalogram_path)
    
    print(f"Saved {len(scalogram_paths)} Lead II scalograms successfully!")
    return scalogram_paths


def generate_and_save_lead2_phasograms(data, signals, output_dir, generator):
    """
    Generate Lead II phasograms and save to disk as .npy files.
    
    Args:
        data: DataFrame with ECG metadata (must have index as ecg_id)
        signals: (N, 12, time_steps) array of preprocessed ECG signals
        output_dir: Directory to save Lead II phasograms
        generator: LeadIIGenerator instance
    
    Returns:
        phasogram_paths: List of file paths to saved phasograms
    """
    os.makedirs(output_dir, exist_ok=True)
    
    phasogram_paths = []
    ecg_ids = data.index.values
    
    print(f"Generating and saving {len(signals)} Lead II phasograms to {output_dir}")
    
    for i, ecg_id in enumerate(tqdm(ecg_ids, desc="Generating Lead II phasograms")):
        phasogram = generator.generate_lead2_phasogram(signals[i])
        phasogram_path = os.path.join(output_dir, f"{ecg_id}_lead2_phasogram.npy")
        np.save(phasogram_path, phasogram)
        phasogram_paths.append(phasogram_path)
    
    print(f"Saved {len(phasogram_paths)} Lead II phasograms successfully!")
    return phasogram_paths



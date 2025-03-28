import torchaudio
import torch
import torchaudio.transforms as T
import torch.nn.functional as F

def compute_normalization_params(sample_rate: int = 16000,
                                 target_length: int = 128,
                                 n_mels: int = 128,
                                 n_fft: int = 1024,
                                 hop_length: int = 160) -> (float, float):
    """
    Compute normalization parameters (mean and standard deviation) based on the entire training dataset.
    
    Args:
        sample_rate: Sampling rate.
        target_length: Target number of time frames for the mel spectrogram.
        n_mels: Number of mel filters.
        n_fft: Number of FFT points.
        hop_length: Hop length.
        
    Returns:
        mean: Global mean.
        std: Global standard deviation.
    """
    # Define the mel spectrogram transformation
    transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        power=2.0
    )
    db_transform = T.AmplitudeToDB(stype='power', top_db=80)
    
    # Load the training dataset
    dataset = torchaudio.datasets.SPEECHCOMMANDS(root="./data", download=True, subset="training")
    
    means = []
    stds = []
    for waveform, _, label, *_ in dataset:
        if waveform.shape[-1] < sample_rate:
            continue
        # Use the first second of audio
        waveform = waveform[:, :sample_rate]
        spec = transform(waveform)
        spec_db = db_transform(spec)
        
        # Adjust the time dimension to target_length
        time_dim = spec_db.size(-1)
        if time_dim < target_length:
            pad = target_length - time_dim
            spec_db = F.pad(spec_db, (0, pad), mode='reflect')
        elif time_dim > target_length:
            start = (time_dim - target_length) // 2
            spec_db = spec_db[..., start:start+target_length]
            
        # Rearrange dimensions to [time, freq]
        spec_db = spec_db.squeeze(0).permute(1, 0)
        
        means.append(spec_db.mean().item())
        stds.append(spec_db.std().item())
    
    global_mean = sum(means) / len(means)
    global_std = sum(stds) / len(stds)
    return global_mean, global_std


if __name__ == "__main__":
    mean, std = compute_normalization_params()
    print(f"Computed normalization parameters: Mean = {mean:.2f}, Std = {std:.2f}")

import torchaudio
import os
import random
import torch
import torchaudio.transforms as T
import torch.nn.functional as F

class FewShotSpeechCommands:
    def __init__(self, root="./data", target_length=128, subset="training"):
        """
        Args:
            root: Dataset root directory.
            target_length: Target number of time frames for the mel spectrogram.
            subset: Dataset subset ("training", "validation", "testing").
        """
        # Initialize audio parameters
        self.sample_rate = 16000
        self.target_length = target_length
        self.subset = subset
        
        # Load dataset
        self.dataset = torchaudio.datasets.SPEECHCOMMANDS(
            root=root,
            download=True,
            subset=subset
        )
        
        # Initialize audio transforms and directly set normalization parameters.
        self._init_audio_transforms()
        self.label_to_samples = self._build_index()

    def _init_audio_transforms(self):
        """Initialize audio transform parameters."""
        self.n_mels = 128
        self.n_fft = 1024
        self.hop_length = 160  # 16000/100 = 160
        
        # Mel spectrogram transformation.
        self.transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            power=2.0
        )
        
        # Amplitude to dB conversion.
        self.db_transform = T.AmplitudeToDB(
            stype='power', 
            top_db=80
        )
        
        # Directly set normalization parameters.
        # These parameters were computed by compute_norm_params.py.
        self.do_normalize = True
        self.mean = -25.88
        self.std = 17.71

    def _build_index(self):
        """Build a mapping from labels to samples."""
        label_to_samples = {}
        for waveform, _, label, *_ in self.dataset:
            # Filter out short samples (ensure at least 1 second).
            if waveform.shape[-1] < self.sample_rate:
                continue
            # Trim to the first 1 second of audio.
            trimmed = waveform[:, :self.sample_rate]
            label_to_samples.setdefault(label, []).append(trimmed)
        return label_to_samples

    def sample_episode(self, n_way=5, k_shot=5, q_query=5):
        """
        Sample a few-shot task.
        
        Returns:
            support: [n_way * k_shot, 128, 128]
            query:   [n_way * q_query, 128, 128]
            labels:  [n_way * q_query]
        """
        # Convert dict_keys to list to ensure random.sample receives a sequence.
        classes = random.sample(list(self.label_to_samples.keys()), n_way)
        support, query, labels = [], [], []

        for class_idx, cls in enumerate(classes):
            samples = self.label_to_samples[cls]
            if len(samples) < k_shot + q_query:
                raise ValueError(f"Class {cls} has only {len(samples)} samples, need at least {k_shot+q_query}")
            
            selected = random.sample(samples, k_shot + q_query)
            support += [self._process_waveform(s) for s in selected[:k_shot]]
            query += [self._process_waveform(q) for q in selected[k_shot:k_shot+q_query]]
            labels += [class_idx] * q_query

        return torch.stack(support), torch.stack(query), torch.tensor(labels)

    def _process_waveform(self, waveform):
        """Process a single audio sample and return the normalized mel spectrogram."""
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        waveform = waveform[:, :self.sample_rate]  # Ensure correct length
        
        # Generate mel spectrogram.
        spec = self.transform(waveform)
        # Convert to decibel scale.
        spec_db = self.db_transform(spec)
        
        # Adjust the time axis.
        time_dim = spec_db.size(-1)
        if time_dim < self.target_length:
            pad = self.target_length - time_dim
            spec_db = F.pad(spec_db, (0, pad), mode='reflect')
        elif time_dim > self.target_length:
            if self.subset == 'training':
                start = random.randint(0, time_dim - self.target_length)
            else:
                start = (time_dim - self.target_length) // 2
            spec_db = spec_db[..., start:start+self.target_length]
        
        # Rearrange dimensions to [time, frequency].
        spec_db = spec_db.squeeze(0).permute(1, 0)
        
        # Apply normalization (using global normalization parameters).
        spec_db = (spec_db - self.mean) / self.std
        
        # If you prefer to normalize each sample individually to 0 mean and 1 std,
        # you can use the following line instead:
        # spec_db = (spec_db - spec_db.mean()) / spec_db.std()
        
        return spec_db


if __name__ == "__main__":
    # Test code using the testing subset.
    loader = FewShotSpeechCommands(subset="testing")
    
    try:
        support, query, labels = loader.sample_episode()
        print("\n=== Shape Verification ===")
        print(f"Support set shape: {support.shape} (expected [25, 128, 128])")
        print(f"Query set shape: {query.shape} (expected [25, 128, 128])")
        print(f"Labels shape: {labels.shape} (expected [25])")
        
        # Verify normalization.
        test_wave = torch.randn(loader.sample_rate)  # 1-second random audio.
        spec = loader._process_waveform(test_wave)
        print("\n=== Normalization Check ===")
        print(f"Mean: {spec.mean().item():.2f} (expected ~0.0)")
        print(f"Std: {spec.std().item():.2f} (expected ~1.0)")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print("Please check:")
        print("1. If each class has enough samples.")
        print("2. If the audio preprocessing parameters are correct.")

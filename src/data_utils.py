import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

@dataclass
class MaskingConfig:
    """Configuration for masking parameters"""
    sequence_length: int = 24  # T time steps
    mask_ratio: float = 0.15   # Ratio of data to mask
    batch_size: int = 32       # Batch size for processing
    min_continuous_mask: int = 2  # Minimum continuous time steps to mask
    max_continuous_mask: int = 5  # Maximum continuous time steps to mask
    feature_mask_ratio: float = 0.2  # Ratio of features to mask in random masking

class TimeSeriesMasking:
    """Class for implementing different masking strategies on time series data"""
    
    def __init__(self, config: MaskingConfig):
        """
        Initialize the masking class with configuration
        
        Args:
            config: MaskingConfig object containing masking parameters
        """
        self.config = config
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters"""
        assert 0 < self.config.mask_ratio < 1, "Mask ratio must be between 0 and 1"
        assert 0 < self.config.feature_mask_ratio < 1, "Feature mask ratio must be between 0 and 1"
        assert self.config.min_continuous_mask <= self.config.max_continuous_mask, \
            "Min continuous mask must be <= max continuous mask"
        assert self.config.max_continuous_mask <= self.config.sequence_length, \
            "Max continuous mask must be <= sequence length"

    def continuous_time_masking(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply continuous time masking to batches of data
        
        Args:
            data: Input data of shape (batch_size, sequence_length, n_features)
            
        Returns:
            Tuple of (masked_data, mask) where mask is 1 where data is masked
        """
        batch_size, seq_len, n_features = data.shape
        masked_data = data.copy()
        mask = np.zeros((batch_size, seq_len), dtype=bool)
        
        for b in range(batch_size):
            # Determine number of sequences to mask
            n_masks = int(seq_len * self.config.mask_ratio / self.config.min_continuous_mask)
            
            for _ in range(n_masks):
                # Randomly choose length of continuous mask
                mask_length = np.random.randint(
                    self.config.min_continuous_mask,
                    min(self.config.max_continuous_mask + 1, seq_len - np.sum(mask[b]) + 1)
                )
                
                # Find valid starting points (where we can fit the mask_length)
                valid_starts = [i for i in range(seq_len - mask_length + 1)
                              if not np.any(mask[b, i:i+mask_length])]
                
                if not valid_starts:
                    break
                    
                # Randomly choose start point and apply mask
                start = np.random.choice(valid_starts)
                mask[b, start:start+mask_length] = True
                masked_data[b, start:start+mask_length] = 0
        
        return masked_data, mask

    def random_feature_masking(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply random feature masking across the time series
        
        Args:
            data: Input data of shape (batch_size, sequence_length, n_features)
            
        Returns:
            Tuple of (masked_data, mask) where mask is 1 where data is masked
        """
        batch_size, seq_len, n_features = data.shape
        masked_data = data.copy()
        mask = np.zeros((batch_size, seq_len, n_features), dtype=bool)
        
        n_features_to_mask = int(n_features * self.config.feature_mask_ratio)
        
        for b in range(batch_size):
            for t in range(seq_len):
                # Randomly select features to mask
                features_to_mask = np.random.choice(
                    n_features, 
                    size=n_features_to_mask, 
                    replace=False
                )
                mask[b, t, features_to_mask] = True
                masked_data[b, t, features_to_mask] = 0
                
        return masked_data, mask

    def random_vector_masking(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply random complete vector masking
        
        Args:
            data: Input data of shape (batch_size, sequence_length, n_features)
            
        Returns:
            Tuple of (masked_data, mask) where mask is 1 where data is masked
        """
        batch_size, seq_len, n_features = data.shape
        masked_data = data.copy()
        mask = np.zeros((batch_size, seq_len), dtype=bool)
        
        n_vectors_to_mask = int(seq_len * self.config.mask_ratio)
        
        for b in range(batch_size):
            # Randomly select time points to mask
            times_to_mask = np.random.choice(
                seq_len,
                size=n_vectors_to_mask,
                replace=False
            )
            mask[b, times_to_mask] = True
            masked_data[b, times_to_mask] = 0
            
        return masked_data, mask

    def apply_all_masks(self, data: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Apply all three masking strategies and return results
        
        Args:
            data: Input data of shape (batch_size, sequence_length, n_features)
            
        Returns:
            Tuple of (dict of masked_data, dict of masks) for each strategy
        """
        results_data = {}
        results_masks = {}
        
        # Apply each masking strategy
        results_data['continuous'], results_masks['continuous'] = \
            self.continuous_time_masking(data)
        results_data['feature'], results_masks['feature'] = \
            self.random_feature_masking(data)
        results_data['vector'], results_masks['vector'] = \
            self.random_vector_masking(data)
            
        return results_data, results_masks

def process_in_batches(masker: TimeSeriesMasking,
                      data: np.ndarray,
                      batch_size: Optional[int] = None) -> Tuple[dict, dict]:
    """
    Process large datasets in batches
    
    Args:
        masker: TimeSeriesMasking instance
        data: Input data of shape (n_samples, sequence_length, n_features)
        batch_size: Optional batch size override
        
    Returns:
        Tuple of (masked_data_dict, masks_dict) for all masking strategies
    """
    if batch_size is None:
        batch_size = masker.config.batch_size
        
    n_samples = len(data)
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    all_masked_data = {k: [] for k in ['continuous', 'feature', 'vector']}
    all_masks = {k: [] for k in ['continuous', 'feature', 'vector']}
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_samples)
        batch_data = data[start_idx:end_idx]
        
        # Pad last batch if necessary
        if len(batch_data) < batch_size:
            pad_size = batch_size - len(batch_data)
            pad_shape = (pad_size,) + batch_data.shape[1:]
            batch_data = np.concatenate([batch_data, np.zeros(pad_shape)], axis=0)
        
        # Apply masking
        masked_data, masks = masker.apply_all_masks(batch_data)
        
        # Store results (removing padding if necessary)
        for k in masked_data:
            all_masked_data[k].append(masked_data[k][:end_idx-start_idx])
            all_masks[k].append(masks[k][:end_idx-start_idx])
    
    # Concatenate results
    final_masked_data = {k: np.concatenate(v) for k, v in all_masked_data.items()}
    final_masks = {k: np.concatenate(v) for k, v in all_masks.items()}
    
    return final_masked_data, final_masks

# Example usage:
if __name__ == "__main__":
    # Create sample data
    n_samples = 100
    seq_length = 24
    n_features = 4
    
    data = np.random.randn(n_samples, seq_length, n_features)
    
    # Initialize config and masker
    config = MaskingConfig(
        sequence_length=seq_length,
        mask_ratio=0.15,
        batch_size=32,
        min_continuous_mask=2,
        max_continuous_mask=5,
        feature_mask_ratio=0.2
    )
    
    masker = TimeSeriesMasking(config)
    
    # Process data in batches
    masked_data, masks = process_in_batches(masker, data)
    
    # Print shapes of results
    for k in masked_data:
        print(f"\n{k.capitalize()} masking results:")
        print(f"Masked data shape: {masked_data[k].shape}")
        print(f"Mask shape: {masks[k].shape}")
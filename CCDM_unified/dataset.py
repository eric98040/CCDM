import os
import numpy as np
import h5py
import copy
import torch
from tqdm import tqdm, trange
import joblib
from PIL import Image
from torchvision import transforms as T
from sklearn.preprocessing import QuantileTransformer


class PowerTransformer:
    """Power transformer for normalizing power values"""

    def __init__(self, n_quantiles=1000, output_distribution="normal"):
        self.qt = QuantileTransformer(
            n_quantiles=n_quantiles,
            output_distribution=output_distribution,
            random_state=42,
        )
        self.max_power = 240000
        self.output_distribution = output_distribution

    def fit(self, power_sequences):
        """
        Fit power sequences
        Args:
            power_sequences: shape (N, D) array of power sequences
        """
        # Reshape to handle full sequences
        values = power_sequences.reshape(-1, 1) / self.max_power
        self.qt.fit(values)
        return self

    def transform(self, power_sequences):
        """
        Transform power sequences
        Args:
            power_sequences: shape (N, D) or (D,) array of power sequences
        Returns:
            Transformed sequences in same shape
        """
        original_shape = power_sequences.shape
        normalized = power_sequences.reshape(-1, 1) / self.max_power

        if self.output_distribution == "normal":
            transformed = self.qt.transform(normalized)
            transformed = (transformed - transformed.min()) / (
                transformed.max() - transformed.min()
            )
        else:
            transformed = self.qt.transform(normalized)

        return transformed.reshape(original_shape)

    def inverse_transform(self, transformed_sequences):
        """
        Transform back to original scale
        Args:
            transformed_sequences: shape (N, D) or (D,) array
        Returns:
            Original scale sequences in same shape
        """
        original_shape = transformed_sequences.shape
        values = transformed_sequences.reshape(-1, 1)

        if self.output_distribution == "normal":
            min_val = self.qt.transform(np.array([[0]]))
            max_val = self.qt.transform(np.array([[1]]))
            values = values * (max_val - min_val) + min_val

        original = self.qt.inverse_transform(values)
        return original.reshape(original_shape) * self.max_power

    def save(self, filepath):
        save_dict = {
            "transformer": self.qt,
            "max_power": self.max_power,
            "output_distribution": self.output_distribution,
        }
        joblib.dump(save_dict, filepath)

    @classmethod
    def load(cls, filepath):
        save_dict = joblib.load(filepath)
        transformer = cls(output_distribution=save_dict["output_distribution"])
        transformer.qt = save_dict["transformer"]
        transformer.max_power = save_dict["max_power"]
        return transformer


class PowerSeqDataset(torch.utils.data.Dataset):
    """
    Dataset for loading optical design images and corresponding power sequences.
    This dataset is specifically designed for the Sliced-CCDM model.
    """

    def __init__(
        self,
        design_folder,
        power_path,
        power_transformer=None,
        transform=T.ToTensor(),
        normalize_design=True,
        return_raw_power=False,
    ):
        self.design_folder = design_folder
        # (N, vector_dim) power sequences
        self.power_data = np.loadtxt(power_path, delimiter=",", skiprows=1)
        self.transform = transform
        self.normalize_design = normalize_design
        self.power_transformer = power_transformer
        self.return_raw_power = return_raw_power
        self.max_power = np.max(self.power_data)
        self.data_name = "power_vector"  # For compatibility with other code

        self.samples = []
        self.labels = []  # Store power sequences
        self.original_indices = []  # Store original indices

        # Load all designs
        self.designs = sorted(
            [f for f in os.listdir(design_folder) if f.endswith(".tiff")],
            key=lambda x: int(x.split(".")[0]),
        )

        # integrity check
        if len(self.designs) != len(self.power_data):
            raise ValueError(
                f"Number of design files ({len(self.designs)}) does not match number of power sequences ({len(self.power_data)})"
            )

        # Create dataset
        for idx, design_file in enumerate(self.designs):
            # Load and process design image
            design_path = os.path.join(self.design_folder, design_file)
            design_sample = Image.open(design_path).convert("L")
            design_sample = self.transform(design_sample)

            if self.normalize_design:
                design_sample = (design_sample - 0.5) / 0.5

            # Get power sequence
            power_sequence = self.power_data[idx, :]

            if self.return_raw_power is True:
                pass  # Return raw power sequence (0 ~ 240000)
            elif self.power_transformer is not None:
                power_sequence = self.power_transformer.transform(power_sequence)
            else:
                power_sequence = (
                    power_sequence / self.max_power
                )  # Normalize by max power

            self.samples.append(
                (design_sample, torch.from_numpy(power_sequence.astype(np.float32)))
            )
            self.labels.append(power_sequence)
            self.original_indices.append(idx)

        self.labels = np.array(self.labels)  # (N, D)
        self.original_indices = np.array(self.original_indices)  # (N,)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        design, power_sequence = self.samples[idx]
        # Convert to consistent format for training
        return {"design": design, "labels": power_sequence}

    def sample_sequences(self, num_samples):
        """Sample power sequences for visualization"""
        base_indices = np.random.choice(len(self.labels), size=num_samples)
        base_sequences = self.labels[base_indices]
        noise = np.random.uniform(-0.1, 0.1, size=base_sequences.shape)
        sampled = np.clip(base_sequences + noise, 0, 1)
        return torch.from_numpy(sampled).float()

    def get_sequences(self):
        return torch.from_numpy(self.labels).float()

    def iter_sequences(self, batch_size=1000):
        for start_idx in range(0, len(self.labels), batch_size):
            end_idx = min(start_idx + batch_size, len(self.labels))
            batch_indices = self.original_indices[start_idx:end_idx]
            batch_labels = self.labels[start_idx:end_idx]
            yield torch.from_numpy(batch_labels).float(), batch_indices

    def load_evaluation_data(self):
        """Return all data for evaluation purposes"""
        # Extract images and labels for evaluation
        images = np.array([sample[0].numpy() for sample in self.samples])
        labels = self.labels
        eval_labels = np.unique(labels, axis=0)

        return images, labels, eval_labels

    def load_train_data(self):
        """Return training data"""
        # Extract images and labels for training
        images = np.array([sample[0].numpy() for sample in self.samples])
        labels = self.labels
        # normalized labels are the same as labels in this case
        return images, labels, labels

    def fn_normalize_labels(self, input_labels):
        """
        Normalize labels into [0,1] range.
        For compatibility with the rest of the codebase.

        Args:
            input_labels: Input labels to normalize

        Returns:
            Normalized labels
        """
        # Input is already normalized, so return as is
        return input_labels

    def fn_denormalize_labels(self, input_labels):
        """
        Denormalize labels back to original scale.
        For compatibility with the rest of the codebase.

        Args:
            input_labels: Normalized labels

        Returns:
            Original scale labels
        """
        # For compatibility, we would use the power transformer here
        # But since we're already working with normalized values, we return as is
        return input_labels


# Factory function to create the appropriate dataset
def create_dataset(dataset_type, **kwargs):
    """
    Factory function to create the dataset based on the dataset type

    Args:
        dataset_type: Type of dataset to create
        **kwargs: Additional arguments for the dataset

    Returns:
        Created dataset instance
    """
    if dataset_type == "power_vector":
        return PowerSeqDataset(**kwargs)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

import os
import math
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import timeit
from einops import rearrange, reduce, repeat, pack, unpack

from models import ResNet34_embed_y2h, model_y2h, ResNet34_embed_y2cov, model_y2cov
from utils import IMGs_dataset


class GaussianFourierProjection(nn.Module):
    """
    Gaussian random features for encoding time steps or continuous labels.
    Provides a way to embed continuous values into a higher-dimensional space.
    """

    def __init__(self, embed_dim, scale=30.0):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
        self.embed_dim = embed_dim

    def forward(self, x):
        x_proj = x[:, None] * (self.W[None, :]).to(x.device) * 2 * np.pi
        x_emb = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return x_emb.view(len(x_emb), self.embed_dim)


class CrossAttention(nn.Module):
    """
    Cross-attention mechanism for combining embeddings from different dimensions.
    This allows for rich interactions between different dimensions of the label vector.
    """

    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, embeddings):
        """
        Args:
            embeddings: Stacked embeddings from different dimensions [D, B, embed_dim]
        Returns:
            Combined embedding with cross-attention [B, embed_dim]
        """
        D, B, E = embeddings.shape

        # Reshape for attention computation
        embeddings = embeddings.transpose(0, 1)  # [B, D, E]

        # Project queries, keys, and values
        q = self.q_proj(embeddings)  # [B, D, E]
        k = self.k_proj(embeddings)  # [B, D, E]
        v = self.v_proj(embeddings)  # [B, D, E]

        # Reshape for multi-head attention
        q = q.view(B, D, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # [B, H, D, head_dim]
        k = k.view(B, D, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # [B, H, D, head_dim]
        v = v.view(B, D, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # [B, H, D, head_dim]

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(
            self.head_dim
        )  # [B, H, D, D]
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)  # [B, H, D, head_dim]

        # Reshape back
        out = out.transpose(1, 2).contiguous().view(B, D, E)  # [B, D, E]

        # Combine dimension embeddings using mean
        out = out.mean(dim=1)  # [B, E]

        # Apply output projection and normalization
        out = self.out_proj(out)
        out = self.norm(out)

        return out


class LabelEmbed:
    def __init__(
        self,
        dataset,
        path_y2h,
        path_y2cov=None,
        y2h_type="sinusoidal",
        y2cov_type=None,
        h_dim=128,
        cov_dim=None,
        batch_size=128,
        nc=3,
        device="cuda",
        label_dim=1,  # Dimension of regression labels
        dim_combination="cross",  # Strategy for combining dimension embeddings
    ):
        """
        Enhanced Label Embedding class for handling multi-dimensional labels in Sliced-CCDM.

        Args:
            dataset: Dataset object containing training data
            path_y2h: Path to store/load h embedding model
            path_y2cov: Path to store/load covariance embedding model
            y2h_type: Type of embedding for y2h ("resnet", "sinusoidal", "gaussian")
            y2cov_type: Type of embedding for y2cov
            h_dim: Dimension of h embedding space
            cov_dim: Dimension of covariance embedding space
            batch_size: Batch size for training
            nc: Number of channels in images
            device: Device to use for computation
            label_dim: Dimension of regression labels (scalar=1, vector>1)
            dim_combination: Strategy for combining dimension embeddings
                            ("mean", "weighted", "attention", "cross", "cross_attention")
        """
        self.dataset = dataset
        # Fix path handling for Windows compatibility
        self.path_y2h = os.path.normpath(path_y2h)
        self.path_y2cov = (
            os.path.normpath(path_y2cov) if path_y2cov is not None else None
        )

        # Print normalized paths for debugging
        print(f"\n Normalized paths:")
        print(f" - path_y2h: {self.path_y2h}")
        if self.path_y2cov:
            print(f" - path_y2cov: {self.path_y2cov}")

        self.y2h_type = y2h_type
        self.y2cov_type = y2cov_type
        self.h_dim = h_dim
        self.cov_dim = cov_dim if cov_dim is not None else 64**2 * nc
        self.batch_size = batch_size
        self.nc = nc
        self.label_dim = label_dim  # Number of dimensions in label vector
        self.dim_combination = (
            dim_combination  # Strategy for combining dimension embeddings
        )
        self.device = device

        # Validate embedding types
        assert y2h_type in ["resnet", "sinusoidal", "gaussian"]
        if y2cov_type is not None:
            assert y2cov_type in ["resnet", "sinusoidal", "gaussian"]

        # Initialize dimension combination mechanisms
        if label_dim > 1:
            if dim_combination == "attention":
                # Create separate attention networks for h and cov embeddings
                self.h_attention_net = nn.Sequential(
                    nn.Linear(h_dim, h_dim // 2), nn.ReLU(), nn.Linear(h_dim // 2, 1)
                ).to(device)
                print(f"\n Initialized h attention network for dimension combination")

                # Create separate attention network for covariance embeddings
                if self.y2cov_type is not None:
                    self.cov_attention_net = nn.Sequential(
                        nn.Linear(self.cov_dim, self.cov_dim // 2),
                        nn.ReLU(),
                        nn.Linear(self.cov_dim // 2, 1),
                    ).to(device)
                    print(
                        f"\n Initialized cov attention network for dimension combination"
                    )

            elif dim_combination == "weighted":
                self.dim_weights = nn.Parameter(torch.ones(label_dim) / label_dim)
                self.dim_weights = self.dim_weights.to(device)
                print(f"\n Initialized weighted combination for dimensions")

            elif dim_combination == "cross":
                self.h_cross_net = nn.Sequential(
                    nn.Linear(h_dim * label_dim, h_dim * 2),
                    nn.LayerNorm(h_dim * 2),
                    nn.ReLU(),
                    nn.Linear(h_dim * 2, h_dim),
                    nn.LayerNorm(h_dim),
                ).to(device)
                print(f"\n Initialized h cross-dimension network")

                # Create separate cross network for covariance embeddings
                if self.y2cov_type is not None:
                    self.cov_cross_net = nn.Sequential(
                        nn.Linear(self.cov_dim * label_dim, self.cov_dim * 2),
                        nn.LayerNorm(self.cov_dim * 2),
                        nn.ReLU(),
                        nn.Linear(self.cov_dim * 2, self.cov_dim),
                        nn.LayerNorm(self.cov_dim),
                    ).to(device)
                    print(f"\n Initialized cov cross-dimension network")

            elif dim_combination == "cross_attention":
                self.h_cross_attention = CrossAttention(
                    embed_dim=h_dim, num_heads=4, dropout=0.1
                ).to(device)
                print(f"\n Initialized h cross-attention mechanism for dimensions")

                # Create separate cross-attention network for covariance embeddings
                if self.y2cov_type is not None:
                    self.cov_cross_attention = CrossAttention(
                        embed_dim=self.cov_dim, num_heads=4, dropout=0.1
                    ).to(device)
                    print(
                        f"\n Initialized cov cross-attention mechanism for dimensions"
                    )

        ## Train or load embedding networks based on selected type
        if y2h_type == "resnet":
            os.makedirs(self.path_y2h, exist_ok=True)

            ## training setups
            epochs_resnet = 200
            epochs_mlp = 500
            base_lr_resnet = 1e-4
            base_lr_mlp = 1e-2

            ## Load training data
            # Get images and labels for training
            if hasattr(dataset, "load_train_data"):
                train_images, _, train_labels = self.dataset.load_train_data()
            else:
                # Alternative way to extract training data from dataset
                train_data = []
                for i in range(len(dataset)):
                    batch = dataset[i]
                    if isinstance(batch, dict):
                        train_data.append((batch["design"], batch["labels"]))
                    else:
                        train_data.append(batch)

                train_images = np.array([item[0] for item in train_data])
                train_labels = np.array([item[1] for item in train_data])

            # Create dataset and loader
            trainset = IMGs_dataset(train_images, train_labels, normalize=True)
            trainloader = torch.utils.data.DataLoader(
                trainset, batch_size=self.batch_size, shuffle=True
            )

            # For multi-dimensional labels, we need unique labels for each dimension
            if self.label_dim > 1:
                unique_labels_norm = np.unique(train_labels, axis=0)
            else:
                unique_labels_norm = np.sort(np.array(list(set(train_labels))))

            ## Training embedding network for y2h
            # Fix checkpoint path handling to look in correct subdirectories
            resnet_dir = os.path.join(self.path_y2h, "resnet_y2h_ckpt_in_train")
            mlp_dir = os.path.join(self.path_y2h, "y2h_ckpt_in_train")

            # Check for resnet checkpoint
            resnet_exists = False
            resnet_y2h_filename_ckpt = None
            if os.path.exists(resnet_dir):
                # Try to find the expected checkpoint file
                expected_file = os.path.join(
                    resnet_dir, f"resnet_y2h_checkpoint_epoch_{epochs_resnet}.pth"
                )
                if os.path.isfile(expected_file):
                    resnet_exists = True
                    resnet_y2h_filename_ckpt = expected_file

            # Fallback to the original path if not found in subdirectory
            if not resnet_exists:
                resnet_y2h_filename_ckpt = os.path.join(
                    self.path_y2h, f"ckpt_resnet_y2h_epoch_{epochs_resnet}.pth"
                )
                resnet_exists = os.path.isfile(resnet_y2h_filename_ckpt)

            # Check for MLP checkpoint
            mlp_exists = False
            mlp_y2h_filename_ckpt = None
            if os.path.exists(mlp_dir):
                # Try to find the expected checkpoint file
                expected_file = os.path.join(
                    mlp_dir, f"mlp_y2h_checkpoint_epoch_{epochs_mlp}.pth"
                )
                if os.path.isfile(expected_file):
                    mlp_exists = True
                    mlp_y2h_filename_ckpt = expected_file

            # Fallback to the original path if not found in subdirectory
            if not mlp_exists:
                mlp_y2h_filename_ckpt = os.path.join(
                    self.path_y2h, f"ckpt_mlp_y2h_epoch_{epochs_mlp}.pth"
                )
                mlp_exists = os.path.isfile(mlp_y2h_filename_ckpt)

            # Print detailed checkpoint information for debugging
            print(f"\n Checking for existing embeddings:")
            print(f" - ResNet checkpoint directory: {resnet_dir}")
            print(f"   Exists: {os.path.exists(resnet_dir)}")
            print(f" - ResNet checkpoint file: {resnet_y2h_filename_ckpt}")
            print(f"   Exists: {resnet_exists}")
            print(f" - MLP checkpoint directory: {mlp_dir}")
            print(f"   Exists: {os.path.exists(mlp_dir)}")
            print(f" - MLP checkpoint file: {mlp_y2h_filename_ckpt}")
            print(f"   Exists: {mlp_exists}")

            # Initialize network with correct input channel count
            model_resnet_y2h = ResNet34_embed_y2h(dim_embed=self.h_dim, nc=self.nc)
            model_resnet_y2h = model_resnet_y2h.to(device)
            model_resnet_y2h = nn.DataParallel(model_resnet_y2h)

            model_mlp_y2h = model_y2h(dim_embed=self.h_dim)
            model_mlp_y2h = model_mlp_y2h.to(device)
            model_mlp_y2h = nn.DataParallel(model_mlp_y2h)

            # Training or loading existing ResNet checkpoint
            if not resnet_exists:
                print(f"\n ResNet checkpoint not found at: {resnet_y2h_filename_ckpt}")
                print("\n Start training CNN for y2h label embedding >>>")
                model_resnet_y2h = train_resnet(
                    net=model_resnet_y2h,
                    net_name="resnet_y2h",
                    trainloader=trainloader,
                    epochs=epochs_resnet,
                    resume_epoch=0,
                    lr_base=base_lr_resnet,
                    lr_decay_factor=0.1,
                    lr_decay_epochs=[80, 140],
                    weight_decay=1e-4,
                    path_to_ckpt=self.path_y2h,
                    label_dim=self.label_dim,
                    device=device,
                )
                # Save model
                torch.save(
                    {
                        "net_state_dict": model_resnet_y2h.state_dict(),
                    },
                    resnet_y2h_filename_ckpt,
                )
            else:
                print(f"\n Loading ResNet checkpoint from: {resnet_y2h_filename_ckpt}")
                try:
                    checkpoint = torch.load(
                        resnet_y2h_filename_ckpt, map_location=device
                    )
                    model_resnet_y2h.load_state_dict(checkpoint["net_state_dict"])
                    print(" Successfully loaded ResNet checkpoint!")
                except Exception as e:
                    print(f" Error loading checkpoint: {e}")
                    print(" Falling back to training...")
                    model_resnet_y2h = train_resnet(
                        net=model_resnet_y2h,
                        net_name="resnet_y2h",
                        trainloader=trainloader,
                        epochs=epochs_resnet,
                        resume_epoch=0,
                        lr_base=base_lr_resnet,
                        lr_decay_factor=0.1,
                        lr_decay_epochs=[80, 140],
                        weight_decay=1e-4,
                        path_to_ckpt=self.path_y2h,
                        label_dim=self.label_dim,
                        device=device,
                    )
                    # Save model
                    torch.save(
                        {
                            "net_state_dict": model_resnet_y2h.state_dict(),
                        },
                        resnet_y2h_filename_ckpt,
                    )

            # Train or load MLP
            if not mlp_exists:
                print(f"\n MLP checkpoint not found at: {mlp_y2h_filename_ckpt}")
                print("\n Start training mlp_y2h >>>")
                model_h2y = model_resnet_y2h.module.h2y
                model_mlp_y2h = train_mlp(
                    unique_labels_norm=unique_labels_norm,
                    model_mlp=model_mlp_y2h,
                    model_name="mlp_y2h",
                    model_h2y=model_h2y,
                    epochs=epochs_mlp,
                    lr_base=base_lr_mlp,
                    lr_decay_factor=0.1,
                    lr_decay_epochs=[150, 250, 350],
                    weight_decay=1e-4,
                    batch_size=128,
                    label_dim=self.label_dim,
                    device=device,
                )
                # Save model
                torch.save(
                    {
                        "net_state_dict": model_mlp_y2h.state_dict(),
                    },
                    mlp_y2h_filename_ckpt,
                )
            else:
                print(f"\n Loading MLP checkpoint from: {mlp_y2h_filename_ckpt}")
                try:
                    checkpoint = torch.load(mlp_y2h_filename_ckpt, map_location=device)
                    model_mlp_y2h.load_state_dict(checkpoint["net_state_dict"])
                    print(" Successfully loaded MLP checkpoint!")
                except Exception as e:
                    print(f" Error loading checkpoint: {e}")
                    print(" Falling back to training...")
                    model_h2y = model_resnet_y2h.module.h2y
                    model_mlp_y2h = train_mlp(
                        unique_labels_norm=unique_labels_norm,
                        model_mlp=model_mlp_y2h,
                        model_name="mlp_y2h",
                        model_h2y=model_h2y,
                        epochs=epochs_mlp,
                        lr_base=base_lr_mlp,
                        lr_decay_factor=0.1,
                        lr_decay_epochs=[150, 250, 350],
                        weight_decay=1e-4,
                        batch_size=128,
                        label_dim=self.label_dim,
                        device=device,
                    )
                    # Save model
                    torch.save(
                        {
                            "net_state_dict": model_mlp_y2h.state_dict(),
                        },
                        mlp_y2h_filename_ckpt,
                    )

            self.model_mlp_y2h = model_mlp_y2h

            # Test with some sample labels
            print("\n Testing y2h embedding with sample labels...")
            if self.label_dim > 1:
                # For multi-dimensional labels, test with a few random samples
                indx_tmp = np.random.choice(
                    len(unique_labels_norm), min(5, len(unique_labels_norm))
                )
                labels_tmp = unique_labels_norm[indx_tmp]
                labels_tmp = torch.from_numpy(labels_tmp).float().to(device)
            else:
                # For scalar labels, use the original approach
                indx_tmp = np.arange(len(unique_labels_norm))
                np.random.shuffle(indx_tmp)
                indx_tmp = indx_tmp[: min(5, len(unique_labels_norm))]
                labels_tmp = unique_labels_norm[indx_tmp].reshape(-1, 1)
                labels_tmp = torch.from_numpy(labels_tmp).float().to(device)

            # Add noise for robustness testing
            if self.label_dim > 1:
                epsilons_tmp = np.random.normal(0, 0.05, size=labels_tmp.shape)
                epsilons_tmp = torch.from_numpy(epsilons_tmp).float().to(device)
            else:
                epsilons_tmp = np.random.normal(0, 0.2, len(labels_tmp))
                epsilons_tmp = (
                    torch.from_numpy(epsilons_tmp).view(-1, 1).float().to(device)
                )

            labels_noise_tmp = torch.clamp(labels_tmp + epsilons_tmp, 0.0, 1.0)

            # Forward pass through the network
            model_resnet_y2h.eval()
            model_mlp_y2h.eval()

            with torch.no_grad():
                # Test both clean and noisy labels
                if self.label_dim > 1:
                    # For multi-dimensional labels, use dimension-wise processing
                    labels_hidden_tmp = self.fn_y2h(labels_tmp)
                    labels_noise_hidden_tmp = self.fn_y2h(labels_noise_tmp)
                    print("\n Embedded shape:", labels_hidden_tmp.shape)
                else:
                    # For scalar labels, use direct processing
                    labels_hidden_tmp = model_mlp_y2h(labels_tmp)
                    labels_noise_hidden_tmp = model_mlp_y2h(labels_noise_tmp)
                    print(
                        "\n Original vs embedded shape:",
                        labels_tmp.shape,
                        "->",
                        labels_hidden_tmp.shape,
                    )

        # Similar logic for y2cov if needed
        if y2cov_type == "resnet" and self.path_y2cov is not None:
            # Implementation similar to y2h but for covariance embedding
            os.makedirs(self.path_y2cov, exist_ok=True)

            ## Training setups
            epochs_resnet = 10
            epochs_mlp = 500
            base_lr_resnet = 1e-4
            base_lr_mlp = 1e-3

            ## Load training data
            if hasattr(dataset, "load_train_data"):
                train_images, _, train_labels = self.dataset.load_train_data()
            else:
                # Alternative way to extract training data
                train_data = []
                for i in range(len(dataset)):
                    batch = dataset[i]
                    if isinstance(batch, dict):
                        train_data.append((batch["design"], batch["labels"]))
                    else:
                        train_data.append(batch)

                train_images = np.array([item[0] for item in train_data])
                train_labels = np.array([item[1] for item in train_data])

            trainset = IMGs_dataset(train_images, train_labels, normalize=True)
            trainloader = torch.utils.data.DataLoader(
                trainset, batch_size=self.batch_size, shuffle=True
            )

            # Get unique labels
            if self.label_dim > 1:
                unique_labels_norm = np.unique(train_labels, axis=0)
            else:
                unique_labels_norm = np.sort(np.array(list(set(train_labels))))

            # Fix checkpoint path handling to look in correct subdirectories
            resnet_dir = os.path.join(self.path_y2cov, "resnet_y2cov_ckpt_in_train")
            mlp_dir = os.path.join(self.path_y2cov, "y2cov_ckpt_in_train")

            # Check for resnet checkpoint
            resnet_exists = False
            resnet_y2cov_filename_ckpt = None
            if os.path.exists(resnet_dir):
                # Try to find the expected checkpoint file
                expected_file = os.path.join(
                    resnet_dir, f"resnet_y2cov_checkpoint_epoch_{epochs_resnet}.pth"
                )
                if os.path.isfile(expected_file):
                    resnet_exists = True
                    resnet_y2cov_filename_ckpt = expected_file

            # Fallback to the original path if not found in subdirectory
            if not resnet_exists:
                resnet_y2cov_filename_ckpt = os.path.join(
                    self.path_y2cov, f"ckpt_resnet_y2cov_epoch_{epochs_resnet}.pth"
                )
                resnet_exists = os.path.isfile(resnet_y2cov_filename_ckpt)

            # Check for MLP checkpoint
            mlp_exists = False
            mlp_y2cov_filename_ckpt = None
            if os.path.exists(mlp_dir):
                # Try to find the expected checkpoint file
                expected_file = os.path.join(
                    mlp_dir, f"mlp_y2cov_checkpoint_epoch_{epochs_mlp}.pth"
                )
                if os.path.isfile(expected_file):
                    mlp_exists = True
                    mlp_y2cov_filename_ckpt = expected_file

            # Fallback to the original path if not found in subdirectory
            if not mlp_exists:
                mlp_y2cov_filename_ckpt = os.path.join(
                    self.path_y2cov, f"ckpt_mlp_y2cov_epoch_{epochs_mlp}.pth"
                )
                mlp_exists = os.path.isfile(mlp_y2cov_filename_ckpt)

            # Print detailed checkpoint information for debugging
            print(f"\n Checking for existing covariance embeddings:")
            print(f" - ResNet checkpoint directory: {resnet_dir}")
            print(f"   Exists: {os.path.exists(resnet_dir)}")
            print(f" - ResNet checkpoint file: {resnet_y2cov_filename_ckpt}")
            print(f"   Exists: {resnet_exists}")
            print(f" - MLP checkpoint directory: {mlp_dir}")
            print(f"   Exists: {os.path.exists(mlp_dir)}")
            print(f" - MLP checkpoint file: {mlp_y2cov_filename_ckpt}")
            print(f"   Exists: {mlp_exists}")

            # Initialize networks
            model_resnet_y2cov = ResNet34_embed_y2cov(
                dim_embed=self.cov_dim, nc=self.nc
            )
            model_resnet_y2cov = model_resnet_y2cov.to(device)
            model_resnet_y2cov = nn.DataParallel(model_resnet_y2cov)

            model_mlp_y2cov = model_y2cov(dim_embed=self.cov_dim)
            model_mlp_y2cov = model_mlp_y2cov.to(device)
            model_mlp_y2cov = nn.DataParallel(model_mlp_y2cov)

            # Train or load ResNet
            if not resnet_exists:
                print(
                    f"\n ResNet checkpoint not found at: {resnet_y2cov_filename_ckpt}"
                )
                print("\n Start training CNN for y2cov label embedding >>>")
                model_resnet_y2cov = train_resnet(
                    net=model_resnet_y2cov,
                    net_name="resnet_y2cov",
                    trainloader=trainloader,
                    epochs=epochs_resnet,
                    resume_epoch=0,
                    lr_base=base_lr_resnet,
                    lr_decay_factor=0.1,
                    lr_decay_epochs=[80, 140],
                    weight_decay=1e-4,
                    path_to_ckpt=self.path_y2cov,
                    label_dim=self.label_dim,
                    device=device,
                )
                torch.save(
                    {
                        "net_state_dict": model_resnet_y2cov.state_dict(),
                    },
                    resnet_y2cov_filename_ckpt,
                )
            else:
                print(
                    f"\n Loading ResNet checkpoint from: {resnet_y2cov_filename_ckpt}"
                )
                try:
                    checkpoint = torch.load(
                        resnet_y2cov_filename_ckpt, map_location=device
                    )
                    model_resnet_y2cov.load_state_dict(checkpoint["net_state_dict"])
                    print(" Successfully loaded ResNet checkpoint!")
                except Exception as e:
                    print(f" Error loading checkpoint: {e}")
                    print(" Falling back to training...")
                    model_resnet_y2cov = train_resnet(
                        net=model_resnet_y2cov,
                        net_name="resnet_y2cov",
                        trainloader=trainloader,
                        epochs=epochs_resnet,
                        resume_epoch=0,
                        lr_base=base_lr_resnet,
                        lr_decay_factor=0.1,
                        lr_decay_epochs=[80, 140],
                        weight_decay=1e-4,
                        path_to_ckpt=self.path_y2cov,
                        label_dim=self.label_dim,
                        device=device,
                    )
                    torch.save(
                        {
                            "net_state_dict": model_resnet_y2cov.state_dict(),
                        },
                        resnet_y2cov_filename_ckpt,
                    )

            # Train or load MLP
            if not mlp_exists:
                print(f"\n MLP checkpoint not found at: {mlp_y2cov_filename_ckpt}")
                print("\n Start training mlp_y2cov >>>")
                model_h2y = model_resnet_y2cov.module.h2y
                model_mlp_y2cov = train_mlp(
                    unique_labels_norm=unique_labels_norm,
                    model_mlp=model_mlp_y2cov,
                    model_name="mlp_y2cov",
                    model_h2y=model_h2y,
                    epochs=epochs_mlp,
                    lr_base=base_lr_mlp,
                    lr_decay_factor=0.1,
                    lr_decay_epochs=[150, 250, 350],
                    weight_decay=1e-4,
                    batch_size=128,
                    label_dim=self.label_dim,
                    device=device,
                )
                torch.save(
                    {
                        "net_state_dict": model_mlp_y2cov.state_dict(),
                    },
                    mlp_y2cov_filename_ckpt,
                )
            else:
                print(f"\n Loading MLP checkpoint from: {mlp_y2cov_filename_ckpt}")
                try:
                    checkpoint = torch.load(
                        mlp_y2cov_filename_ckpt, map_location=device
                    )
                    model_mlp_y2cov.load_state_dict(checkpoint["net_state_dict"])
                    print(" Successfully loaded MLP checkpoint!")
                except Exception as e:
                    print(f" Error loading checkpoint: {e}")
                    print(" Falling back to training...")
                    model_h2y = model_resnet_y2cov.module.h2y
                    model_mlp_y2cov = train_mlp(
                        unique_labels_norm=unique_labels_norm,
                        model_mlp=model_mlp_y2cov,
                        model_name="mlp_y2cov",
                        model_h2y=model_h2y,
                        epochs=epochs_mlp,
                        lr_base=base_lr_mlp,
                        lr_decay_factor=0.1,
                        lr_decay_epochs=[150, 250, 350],
                        weight_decay=1e-4,
                        batch_size=128,
                        label_dim=self.label_dim,
                        device=device,
                    )
                    torch.save(
                        {
                            "net_state_dict": model_mlp_y2cov.state_dict(),
                        },
                        mlp_y2cov_filename_ckpt,
                    )

            self.model_mlp_y2cov = model_mlp_y2cov

            # Test covariance embedding
            print("\n Testing y2cov embedding...")
            if self.label_dim > 1:
                # For multi-dimensional labels
                indx_tmp = np.random.choice(
                    len(unique_labels_norm), min(5, len(unique_labels_norm))
                )
                labels_tmp = unique_labels_norm[indx_tmp]
                labels_tmp = torch.from_numpy(labels_tmp).float().to(device)
            else:
                # For scalar labels
                indx_tmp = np.arange(len(unique_labels_norm))
                np.random.shuffle(indx_tmp)
                indx_tmp = indx_tmp[: min(5, len(unique_labels_norm))]
                labels_tmp = unique_labels_norm[indx_tmp].reshape(-1, 1)
                labels_tmp = torch.from_numpy(labels_tmp).float().to(device)

            with torch.no_grad():
                # Get covariance embeddings
                cov_emb = self.fn_y2cov(labels_tmp)
                print(f"\n Covariance embedding shape: {cov_emb.shape}")

    def fn_y2h(self, labels):
        """
        Enhanced function to convert labels to h embedding with multi-dimensional label support.

        Args:
            labels: Labels to embed [B, D] or [B, 1]

        Returns:
            embedding: Embedded labels [B, h_dim]
        """
        embed_dim = self.h_dim
        device = labels.device

        # Handle multi-dimensional labels
        if len(labels.shape) > 1 and labels.shape[1] > 1:
            # Process each dimension separately
            dim_embeddings = []
            for d in range(labels.shape[1]):
                dim_labels = labels[:, d].view(len(labels))

                if self.y2h_type == "sinusoidal":
                    # Create sinusoidal embedding for this dimension
                    max_period = 10000
                    half = embed_dim // 2
                    freqs = torch.exp(
                        -math.log(max_period)
                        * torch.arange(start=0, end=half, dtype=torch.float32)
                        / half
                    ).to(device=device)
                    args = dim_labels[:, None].float() * freqs[None]
                    dim_embed = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

                    # Pad if needed
                    if embed_dim % 2:
                        dim_embed = torch.cat(
                            [dim_embed, torch.zeros_like(dim_embed[:, :1])], dim=-1
                        )

                    dim_embed = (dim_embed + 1) / 2  # Ensure in range [0,1]

                elif self.y2h_type == "gaussian":
                    # Use Gaussian Fourier projection
                    gfp = GaussianFourierProjection(embed_dim=embed_dim).to(device)
                    dim_embed = gfp(dim_labels.unsqueeze(-1))
                    dim_embed = (dim_embed + 1) / 2  # Ensure in range [0,1]

                elif self.y2h_type == "resnet":
                    # Use trained MLP for embedding
                    self.model_mlp_y2h.eval()
                    dim_labels = dim_labels.view(-1, 1)
                    dim_embed = self.model_mlp_y2h(dim_labels)

                dim_embeddings.append(dim_embed)

            # Stack dimension embeddings
            stacked_embeddings = torch.stack(dim_embeddings)  # [D, B, embed_dim]

            # Combine based on selected strategy
            if self.dim_combination == "mean":
                # Simple mean across dimensions
                embedding = torch.mean(stacked_embeddings, dim=0)  # [B, embed_dim]

            elif self.dim_combination == "weighted":
                # Weighted sum across dimensions
                weights = F.softmax(self.dim_weights, dim=0)  # [D]
                weights = weights.view(-1, 1, 1)  # [D, 1, 1]
                embedding = torch.sum(
                    stacked_embeddings * weights, dim=0
                )  # [B, embed_dim]

            elif self.dim_combination == "attention":
                # Attention-based weighting
                stacked_for_attn = stacked_embeddings.permute(
                    1, 0, 2
                )  # [B, D, embed_dim]

                # Generate attention scores using h-specific attention network
                attn_scores = self.h_attention_net(stacked_for_attn).squeeze(
                    -1
                )  # [B, D]
                attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)  # [B, D, 1]

                # Apply attention weights
                embedding = torch.sum(
                    stacked_for_attn * attn_weights, dim=1
                )  # [B, embed_dim]

            elif self.dim_combination == "cross":
                # Consider cross-dimension interactions
                stacked_for_cross = stacked_embeddings.permute(
                    1, 0, 2
                )  # [B, D, embed_dim]
                batch_size = stacked_for_cross.shape[0]

                # Flatten all dimension embeddings
                flattened = stacked_for_cross.reshape(
                    batch_size, -1
                )  # [B, D*embed_dim]

                # Process through cross-dimension network
                embedding = self.h_cross_net(flattened)  # [B, embed_dim]

            elif self.dim_combination == "cross_attention":
                # Use cross-attention mechanism
                embedding = self.h_cross_attention(stacked_embeddings)  # [B, embed_dim]

            else:
                # Default to mean
                embedding = torch.mean(stacked_embeddings, dim=0)

        else:
            # Original code for scalar labels
            if self.y2h_type == "sinusoidal":
                max_period = 10000
                labels = labels.view(len(labels))
                half = embed_dim // 2
                freqs = torch.exp(
                    -math.log(max_period)
                    * torch.arange(start=0, end=half, dtype=torch.float32)
                    / half
                ).to(device=labels.device)
                args = labels[:, None].float() * freqs[None]
                embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
                if embed_dim % 2:
                    embedding = torch.cat(
                        [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
                    )
                embedding = (embedding + 1) / 2  # make sure in [0,1]

            elif self.y2h_type == "gaussian":
                embedding = GaussianFourierProjection(embed_dim=embed_dim).to(device)(
                    labels
                )
                embedding = (embedding + 1) / 2  # make sure in [0,1]

            elif self.y2h_type == "resnet":
                self.model_mlp_y2h.eval()
                self.model_mlp_y2h = self.model_mlp_y2h.to(device)
                embedding = self.model_mlp_y2h(labels)

        return embedding

    def fn_y2cov(self, labels):
        """
        Enhanced function to convert labels to covariance embedding with multi-dimensional label support.

        Args:
            labels: Labels to embed [B, D] or [B, 1]

        Returns:
            embedding: Covariance embedding [B, cov_dim]
        """
        embed_dim = self.cov_dim
        device = labels.device

        # Handle multi-dimensional labels
        if len(labels.shape) > 1 and labels.shape[1] > 1:
            # Process each dimension separately
            dim_embeddings = []
            for d in range(labels.shape[1]):
                dim_labels = labels[:, d].view(len(labels))

                if self.y2cov_type == "sinusoidal":
                    # Create sinusoidal embedding
                    max_period = 10000
                    half = embed_dim // 2
                    freqs = torch.exp(
                        -math.log(max_period)
                        * torch.arange(start=0, end=half, dtype=torch.float32)
                        / half
                    ).to(device=device)
                    args = dim_labels[:, None].float() * freqs[None]
                    dim_embed = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

                    if embed_dim % 2:
                        dim_embed = torch.cat(
                            [dim_embed, torch.zeros_like(dim_embed[:, :1])], dim=-1
                        )

                    # For covariance, ensure positive embeddings
                    dim_embed = dim_embed + 1

                elif self.y2cov_type == "gaussian":
                    # Use Gaussian Fourier projection
                    gfp = GaussianFourierProjection(embed_dim=embed_dim).to(device)
                    dim_embed = gfp(dim_labels.unsqueeze(-1))
                    # Ensure positive for covariance
                    dim_embed = dim_embed + 1

                elif self.y2cov_type == "resnet":
                    # Use trained MLP for embedding
                    self.model_mlp_y2cov.eval()
                    dim_labels = dim_labels.view(-1, 1)
                    dim_embed = self.model_mlp_y2cov(dim_labels)

                dim_embeddings.append(dim_embed)

            # Stack dimension embeddings
            stacked_embeddings = torch.stack(dim_embeddings)  # [D, B, embed_dim]

            # Combine based on selected strategy
            if self.dim_combination == "mean":
                embedding = torch.mean(stacked_embeddings, dim=0)

            elif self.dim_combination == "weighted":
                weights = F.softmax(self.dim_weights, dim=0)
                weights = weights.view(-1, 1, 1)
                embedding = torch.sum(stacked_embeddings * weights, dim=0)

            elif self.dim_combination == "attention":
                stacked_for_attn = stacked_embeddings.permute(1, 0, 2)
                # Use specific attention network for covariance embedding
                attn_scores = self.cov_attention_net(stacked_for_attn).squeeze(-1)
                attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)
                embedding = torch.sum(stacked_for_attn * attn_weights, dim=1)

            elif self.dim_combination == "cross":
                stacked_for_cross = stacked_embeddings.permute(1, 0, 2)
                batch_size = stacked_for_cross.shape[0]
                flattened = stacked_for_cross.reshape(batch_size, -1)
                # Use specific cross network for covariance embedding
                embedding = self.cov_cross_net(flattened)

            elif self.dim_combination == "cross_attention":
                # Use specific cross-attention for covariance embedding
                embedding = self.cov_cross_attention(stacked_embeddings)

            else:
                embedding = torch.mean(stacked_embeddings, dim=0)

        else:
            # Original code for scalar labels
            if self.y2cov_type == "sinusoidal":
                max_period = 10000
                labels = labels.view(len(labels))
                half = embed_dim // 2
                freqs = torch.exp(
                    -math.log(max_period)
                    * torch.arange(start=0, end=half, dtype=torch.float32)
                    / half
                ).to(device=labels.device)
                args = labels[:, None].float() * freqs[None]
                embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
                if embed_dim % 2:
                    embedding = torch.cat(
                        [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
                    )
                embedding = embedding + 1  # make sure embedding is not negative

            elif self.y2cov_type == "gaussian":
                embedding = GaussianFourierProjection(embed_dim=embed_dim).to(device)(
                    labels
                )
                embedding = embedding + 1  # make sure embedding is not negative

            elif self.y2cov_type == "resnet":
                self.model_mlp_y2cov.eval()
                self.model_mlp_y2cov = self.model_mlp_y2cov.to(device)
                embedding = self.model_mlp_y2cov(labels)

        return embedding


class label_dataset(torch.utils.data.Dataset):
    """
    Dataset for training with labels, supporting multi-dimensional labels.
    """

    def __init__(self, labels):
        super(label_dataset, self).__init__()
        self.labels = labels
        self.n_samples = len(self.labels)

    def __getitem__(self, index):
        y = self.labels[index]
        return y

    def __len__(self):
        return self.n_samples


def train_resnet(
    net,
    net_name,
    trainloader,
    epochs=200,
    resume_epoch=0,
    lr_base=0.01,
    lr_decay_factor=0.1,
    lr_decay_epochs=[80, 140],
    weight_decay=1e-4,
    path_to_ckpt=None,
    label_dim=1,
    device="cuda",
):
    """
    Train a ResNet model for embedding generation with multi-dimensional label support.

    Args:
        net: Network to train
        net_name: Name identifier for the network
        trainloader: DataLoader for training data
        epochs: Number of training epochs
        resume_epoch: Epoch to resume from
        lr_base: Base learning rate
        lr_decay_factor: Factor to reduce learning rate at decay epochs
        lr_decay_epochs: List of epochs where learning rate will be reduced
        weight_decay: Weight decay for optimizer
        path_to_ckpt: Path to save checkpoints
        label_dim: Dimension of regression labels
        device: Device to train on

    Returns:
        Trained network
    """

    def adjust_learning_rate_1(optimizer, epoch):
        """Decrease the learning rate"""
        lr = lr_base

        num_decays = len(lr_decay_epochs)
        for decay_i in range(num_decays):
            if epoch >= lr_decay_epochs[decay_i]:
                lr = lr * lr_decay_factor

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    net = net.to(device)
    criterion = nn.MSELoss()
    optimizer_resnet = torch.optim.SGD(
        net.parameters(), lr=lr_base, momentum=0.9, weight_decay=weight_decay
    )

    # Resume training; load checkpoint
    if path_to_ckpt is not None and resume_epoch > 0:
        save_dir = os.path.join(path_to_ckpt, f"{net_name}_ckpt_in_train")
        os.makedirs(save_dir, exist_ok=True)
        save_file = os.path.join(
            save_dir, f"{net_name}_checkpoint_epoch_{resume_epoch}.pth"
        )

        if os.path.isfile(save_file):
            print(f"Loading checkpoint from {save_file}")
            checkpoint = torch.load(save_file, map_location=device)
            net.load_state_dict(checkpoint["net_state_dict"])
            optimizer_resnet.load_state_dict(checkpoint["optimizer_state_dict"])
            torch.set_rng_state(checkpoint["rng_state"])
        else:
            print(f"No checkpoint found at {save_file}, starting from scratch")

    start_tmp = timeit.default_timer()
    for epoch in range(resume_epoch, epochs):
        net.train()
        train_loss = 0
        adjust_learning_rate_1(optimizer_resnet, epoch)

        for _, batch_data in enumerate(trainloader):
            # Process batch data
            if isinstance(batch_data, list) or isinstance(batch_data, tuple):
                batch_train_images, batch_train_labels = batch_data
            else:
                # If data is in dict format
                batch_train_images = (
                    batch_data["design"]
                    if "design" in batch_data
                    else batch_data["image"]
                )
                batch_train_labels = batch_data["labels"]

            batch_train_images = batch_train_images.type(torch.float).to(device)

            # Handle multi-dimensional labels
            if label_dim > 1:
                # For multi-dimensional labels, train separate models for each dimension
                # or use first dimension as representative (simplification)
                batch_train_labels = (
                    batch_train_labels[:, 0:1].type(torch.float).to(device)
                )
            else:
                # For scalar labels
                batch_train_labels = (
                    batch_train_labels.type(torch.float).view(-1, 1).to(device)
                )

            # Forward pass
            outputs, _ = net(batch_train_images)
            loss = criterion(outputs, batch_train_labels)

            # Backward pass
            optimizer_resnet.zero_grad()
            loss.backward()
            optimizer_resnet.step()

            train_loss += loss.cpu().item()

        train_loss = train_loss / len(trainloader)

        print(
            "Train {} for embedding: [epoch {}/{}] train_loss:{:.4f} Time:{:.4f}".format(
                net_name,
                epoch + 1,
                epochs,
                train_loss,
                timeit.default_timer() - start_tmp,
            )
        )

        # Save checkpoint
        if path_to_ckpt is not None and (
            ((epoch + 1) % 50 == 0) or (epoch + 1 == epochs)
        ):
            save_dir = os.path.join(path_to_ckpt, f"{net_name}_ckpt_in_train")
            os.makedirs(save_dir, exist_ok=True)
            save_file = os.path.join(
                save_dir, f"{net_name}_checkpoint_epoch_{epoch + 1}.pth"
            )

            # Print save path for debugging
            print(f"Saving checkpoint to: {save_file}")

            torch.save(
                {
                    "epoch": epoch,
                    "net_state_dict": net.state_dict(),
                    "optimizer_state_dict": optimizer_resnet.state_dict(),
                    "rng_state": torch.get_rng_state(),
                },
                save_file,
            )

    return net


def train_mlp(
    unique_labels_norm,
    model_mlp,
    model_name,
    model_h2y,
    epochs=500,
    lr_base=0.01,
    lr_decay_factor=0.1,
    lr_decay_epochs=[150, 250, 350],
    weight_decay=1e-4,
    batch_size=128,
    label_dim=1,
    device="cuda",
):
    """
    Train an MLP model to map normalized labels to embeddings with multi-dimensional support.

    Args:
        unique_labels_norm: Normalized unique labels array
        model_mlp: MLP model to train
        model_name: Name identifier for the model
        model_h2y: Pre-trained model mapping embeddings to labels
        epochs: Number of training epochs
        lr_base: Base learning rate
        lr_decay_factor: Factor to reduce learning rate at decay epochs
        lr_decay_epochs: List of epochs where learning rate will be reduced
        weight_decay: Weight decay for optimizer
        batch_size: Batch size for training
        label_dim: Dimension of regression labels
        device: Device to train on

    Returns:
        Trained MLP model
    """
    model_mlp = model_mlp.to(device)
    model_h2y = model_h2y.to(device)

    def adjust_learning_rate_2(optimizer, epoch):
        """Decrease the learning rate"""
        lr = lr_base

        num_decays = len(lr_decay_epochs)
        for decay_i in range(num_decays):
            if epoch >= lr_decay_epochs[decay_i]:
                lr = lr * lr_decay_factor

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    # Verify that labels are normalized
    if isinstance(unique_labels_norm, np.ndarray):
        assert np.max(unique_labels_norm) <= 1 and np.min(unique_labels_norm) >= 0

    # Handle multi-dimensional labels differently
    if (
        label_dim > 1
        and len(unique_labels_norm.shape) > 1
        and unique_labels_norm.shape[1] > 1
    ):
        # For multi-dimensional labels, create a dataset with individual dimensions
        flattened_labels = []
        for label in unique_labels_norm:
            for dim_idx in range(len(label)):
                flattened_labels.append([label[dim_idx]])

        trainset = label_dataset(np.array(flattened_labels))
    else:
        # For scalar labels, use original approach
        trainset = label_dataset(unique_labels_norm)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True
    )

    model_h2y.eval()
    optimizer_mlp = torch.optim.SGD(
        model_mlp.parameters(), lr=lr_base, momentum=0.9, weight_decay=weight_decay
    )

    start_tmp = timeit.default_timer()
    for epoch in range(epochs):
        model_mlp.train()
        train_loss = 0
        adjust_learning_rate_2(optimizer_mlp, epoch)

        for _, batch_labels in enumerate(trainloader):
            # If the dataset returns a multi-dim tensor from __getitem__,
            # ensure it's properly shaped as a batch of scalar labels
            if isinstance(batch_labels, np.ndarray) and batch_labels.ndim > 2:
                batch_labels = torch.from_numpy(
                    batch_labels.reshape(batch_labels.shape[0], -1)
                )

            # Convert to float and ensure proper shape
            batch_labels = batch_labels.type(torch.float).to(device)
            if batch_labels.dim() == 1:
                batch_labels = batch_labels.view(-1, 1)

            # Generate noises to add to labels
            batch_size_curr = len(batch_labels)

            # Add noise adapted to dimensions
            if batch_labels.shape[1] > 1:
                # For multi-dimensional labels
                batch_gamma = np.random.normal(0, 0.05, batch_labels.shape)
                batch_gamma = torch.from_numpy(batch_gamma).type(torch.float).to(device)
            else:
                # For scalar labels
                batch_gamma = np.random.normal(0, 0.2, batch_size_curr)
                batch_gamma = (
                    torch.from_numpy(batch_gamma)
                    .view(-1, 1)
                    .type(torch.float)
                    .to(device)
                )

            # Add noise to labels
            batch_labels_noise = torch.clamp(batch_labels + batch_gamma, 0.0, 1.0)

            # If multi-dim, process each dimension individually
            if batch_labels.shape[1] > 1:
                # For simplicity, we'll just process the first dimension
                # A more complete solution would handle all dimensions
                batch_labels_noise_1d = batch_labels_noise[:, 0:1]

                # Forward pass
                batch_hiddens_noise = model_mlp(batch_labels_noise_1d)
                batch_rec_labels_noise = model_h2y(batch_hiddens_noise)

                loss = nn.MSELoss()(batch_rec_labels_noise, batch_labels_noise_1d)
            else:
                # Forward pass for scalar labels
                batch_hiddens_noise = model_mlp(batch_labels_noise)
                batch_rec_labels_noise = model_h2y(batch_hiddens_noise)

                loss = nn.MSELoss()(batch_rec_labels_noise, batch_labels_noise)

            # Backward pass
            optimizer_mlp.zero_grad()
            loss.backward()
            optimizer_mlp.step()

            train_loss += loss.cpu().item()

        train_loss = train_loss / len(trainloader)

        if (epoch + 1) % 50 == 0 or epoch + 1 == epochs:
            print(
                "Train {}: [epoch {}/{}] train_loss:{:.4f} Time:{:.4f}".format(
                    model_name,
                    epoch + 1,
                    epochs,
                    train_loss,
                    timeit.default_timer() - start_tmp,
                )
            )

            # Save checkpoint for MLP training
            save_dir = os.path.join(
                os.path.dirname(os.path.normpath(model_name)),
                f"{model_name}_ckpt_in_train",
            )
            os.makedirs(save_dir, exist_ok=True)
            save_file = os.path.join(
                save_dir, f"{model_name}_checkpoint_epoch_{epoch + 1}.pth"
            )

            print(f"Saving MLP checkpoint to: {save_file}")

            try:
                torch.save(
                    {
                        "epoch": epoch,
                        "net_state_dict": model_mlp.state_dict(),
                        "optimizer_state_dict": optimizer_mlp.state_dict(),
                        "rng_state": torch.get_rng_state(),
                    },
                    save_file,
                )
            except Exception as e:
                print(f"Error saving MLP checkpoint: {e}")

    return model_mlp

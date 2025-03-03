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


def sinusoidal_embedding(labels, embed_dim, device):
    """
    Sinusoidal positional embedding for labels
    """
    max_period = 10000
    labels = labels.view(len(labels))
    half = embed_dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=device)
    args = labels[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if embed_dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""

    """ from https://colab.research.google.com/drive/120kYYBOVa1i0TD85RjlEkFjaWDxSFUx3?usp=sharing#scrollTo=YyQtV7155Nht """

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
        label_dim=1,
        dim_weights_learn=True,
        dim_combination="attention",
    ):
        """
        Enhanced Label Embedding class with advanced multi-dimensional label handling.

        Parameters:
        - dataset: Dataset object or name
        - path_y2h: Path to store/load y2h embedding model
        - path_y2cov: Path to store/load y2cov embedding model
        - y2h_type: Type of embedding for y2h ("resnet", "sinusoidal", "gaussian")
        - y2cov_type: Type of embedding for y2cov
        - h_dim: Dimension of h embedding
        - cov_dim: Dimension of covariance embedding
        - batch_size: Batch size for training
        - nc: Number of channels
        - device: Device to use
        - label_dim: Dimension of labels
        - dim_weights_learn: Whether to learn dimension weights
        - dim_combination: Method to combine dimension embeddings
        """
        self.data_name = dataset.data_name if hasattr(dataset, "data_name") else None
        self.path_y2h = path_y2h
        self.path_y2cov = path_y2cov
        self.y2h_type = y2h_type
        self.y2cov_type = y2cov_type
        self.h_dim = h_dim
        self.cov_dim = cov_dim
        self.batch_size = batch_size
        self.nc = nc
        self.label_dim = label_dim
        self.dim_weights_learn = dim_weights_learn
        self.dim_combination = dim_combination
        self.device = device

        # Initialize dimension weights if needed
        if dim_weights_learn and label_dim > 1:
            self.dim_weights = nn.Parameter(torch.ones(label_dim) / label_dim)
            self.dim_weights = self.dim_weights.to(device)

        # Initialize attention mechanism for dimension combination if needed
        if dim_combination == "attention" and label_dim > 1:
            self.attention_layer = nn.Sequential(
                nn.Linear(h_dim, h_dim // 2), nn.ReLU(), nn.Linear(h_dim // 2, 1)
            ).to(device)

        # Setup cross-dimension interaction module if needed
        if label_dim > 1:
            self.cross_dim_interaction = nn.Sequential(
                nn.Linear(h_dim * label_dim, h_dim * 2),
                nn.LayerNorm(h_dim * 2),
                nn.ReLU(),
                nn.Linear(h_dim * 2, h_dim),
                nn.LayerNorm(h_dim),
            ).to(device)

        # Initialize embedding networks based on types
        if y2h_type == "resnet":
            # Initialize ResNet-based model for y2h
            os.makedirs(path_y2h, exist_ok=True)
            if not os.path.exists(path_y2h + "/net_y2h.pth"):
                print("\n Start training ResNet for label embedding...")
                # Training logic will be handled elsewhere
                pass
            # Load the pre-trained model
            self.model_mlp_y2h = model_y2h(dim_embed=h_dim)
            self.model_mlp_y2h.load_state_dict(torch.load(path_y2h + "/net_y2h.pth"))
            self.model_mlp_y2h.eval()

        # Initialize covariance embedding network if needed
        if y2cov_type == "resnet" and path_y2cov is not None:
            os.makedirs(path_y2cov, exist_ok=True)
            if not os.path.exists(path_y2cov + "/net_y2cov.pth"):
                print("\n Start training ResNet for covariance embedding...")
                # Training logic will be handled elsewhere
                pass
            # Load the pre-trained model
            self.model_mlp_y2cov = model_y2cov(dim_embed=cov_dim)
            self.model_mlp_y2cov.load_state_dict(
                torch.load(path_y2cov + "/net_y2cov.pth")
            )
            self.model_mlp_y2cov.eval()

        # Initialize Gaussian Fourier projection if needed
        if y2h_type == "gaussian":
            self.gfp_h = GaussianFourierProjection(embed_dim=h_dim).to(device)

        if y2cov_type == "gaussian" and cov_dim is not None:
            self.gfp_cov = GaussianFourierProjection(embed_dim=cov_dim).to(device)

    def fn_y2h(self, labels):
        """
        Enhanced function to convert labels to h embedding with sophisticated
        multi-dimensional label handling.

        Parameters:
        - labels: Labels to embed [B, D] or [B]

        Returns:
        - embedding: Embedded labels [B, h_dim]
        """
        embed_dim = self.h_dim

        # Handle multi-dimensional labels
        if len(labels.shape) > 1 and labels.shape[1] > 1:
            device = labels.device

            if self.y2h_type == "sinusoidal":
                # Generate embeddings for each dimension separately
                dim_embeddings = []
                for d in range(labels.shape[1]):
                    dim_labels = labels[:, d].view(len(labels))
                    dim_embed = sinusoidal_embedding(dim_labels, embed_dim, device)
                    dim_embeddings.append(dim_embed)

                # Stack embeddings to [D, B, embed_dim]
                stacked_embeddings = torch.stack(dim_embeddings)

                # Process based on combination method
                if self.dim_combination == "mean":
                    # Simple mean across dimensions
                    embedding = torch.mean(stacked_embeddings, dim=0)

                elif self.dim_combination == "weighted":
                    # Apply learned weights to dimensions
                    if self.dim_weights_learn:
                        weights = F.softmax(self.dim_weights, dim=0)
                        embedding = torch.sum(
                            stacked_embeddings * weights.view(-1, 1, 1), dim=0
                        )
                    else:
                        # Use uniform weights
                        embedding = torch.mean(stacked_embeddings, dim=0)

                elif self.dim_combination == "attention":
                    # Apply attention mechanism over dimensions
                    # [D, B, embed_dim] -> [B, D, embed_dim]
                    stacked_for_attn = stacked_embeddings.permute(1, 0, 2)

                    # Calculate attention scores for each dimension
                    attn_scores = self.attention_layer(stacked_for_attn).squeeze(
                        -1
                    )  # [B, D]
                    attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(
                        -1
                    )  # [B, D, 1]

                    # Apply attention weights
                    embedding = torch.sum(
                        stacked_for_attn * attn_weights, dim=1
                    )  # [B, embed_dim]

                elif self.dim_combination == "cross":
                    # Consider cross-dimension interactions
                    B, D = labels.shape[0], labels.shape[1]

                    # Flatten embeddings: [D, B, embed_dim] -> [B, D*embed_dim]
                    flat_embeddings = stacked_embeddings.permute(1, 0, 2).reshape(
                        B, D * embed_dim
                    )

                    # Apply cross-dimension interaction module
                    embedding = self.cross_dim_interaction(flat_embeddings)

                else:
                    # Default to mean
                    embedding = torch.mean(stacked_embeddings, dim=0)

            elif self.y2h_type == "gaussian":
                # Process each dimension with Gaussian Fourier projection
                dim_embeddings = []
                for d in range(labels.shape[1]):
                    dim_labels = labels[:, d].view(-1, 1)
                    dim_embed = self.gfp_h(dim_labels)
                    dim_embeddings.append(dim_embed)

                # Stack and process based on combination method
                stacked_embeddings = torch.stack(dim_embeddings)

                # Use the same combination logic as above
                if self.dim_combination == "mean":
                    embedding = torch.mean(stacked_embeddings, dim=0)
                elif self.dim_combination == "weighted":
                    if self.dim_weights_learn:
                        weights = F.softmax(self.dim_weights, dim=0)
                        embedding = torch.sum(
                            stacked_embeddings * weights.view(-1, 1, 1), dim=0
                        )
                    else:
                        embedding = torch.mean(stacked_embeddings, dim=0)
                elif self.dim_combination == "attention":
                    stacked_for_attn = stacked_embeddings.permute(1, 0, 2)
                    attn_scores = self.attention_layer(stacked_for_attn).squeeze(-1)
                    attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)
                    embedding = torch.sum(stacked_for_attn * attn_weights, dim=1)
                elif self.dim_combination == "cross":
                    B, D = labels.shape[0], labels.shape[1]
                    flat_embeddings = stacked_embeddings.permute(1, 0, 2).reshape(
                        B, D * embed_dim
                    )
                    embedding = self.cross_dim_interaction(flat_embeddings)
                else:
                    embedding = torch.mean(stacked_embeddings, dim=0)

            elif self.y2h_type == "resnet":
                # For ResNet-based embedding, adapt to handle multi-dimensional labels
                self.model_mlp_y2h.eval()
                self.model_mlp_y2h = self.model_mlp_y2h.to(device)

                # Process each dimension separately
                dim_embeddings = []
                for d in range(labels.shape[1]):
                    dim_labels = labels[:, d].view(-1, 1)
                    dim_embed = self.model_mlp_y2h(dim_labels)
                    dim_embeddings.append(dim_embed)

                # Combine embeddings based on selected method
                stacked_embeddings = torch.stack(dim_embeddings)

                if self.dim_combination == "mean":
                    embedding = torch.mean(stacked_embeddings, dim=0)
                elif self.dim_combination == "weighted":
                    if self.dim_weights_learn:
                        weights = F.softmax(self.dim_weights, dim=0)
                        embedding = torch.sum(
                            stacked_embeddings * weights.view(-1, 1, 1), dim=0
                        )
                    else:
                        embedding = torch.mean(stacked_embeddings, dim=0)
                elif self.dim_combination == "attention":
                    stacked_for_attn = stacked_embeddings.permute(1, 0, 2)
                    attn_scores = self.attention_layer(stacked_for_attn).squeeze(-1)
                    attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)
                    embedding = torch.sum(stacked_for_attn * attn_weights, dim=1)
                elif self.dim_combination == "cross":
                    B, D = labels.shape[0], labels.shape[1]
                    flat_embeddings = stacked_embeddings.permute(1, 0, 2).reshape(
                        B, D * embed_dim
                    )
                    embedding = self.cross_dim_interaction(flat_embeddings)
                else:
                    embedding = torch.mean(stacked_embeddings, dim=0)

        else:
            # Original single-dimension embedding code
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
                embedding = (embedding + 1) / 2

            elif self.y2h_type == "gaussian":
                if hasattr(self, "gfp_h"):
                    embedding = self.gfp_h(labels.view(-1, 1))
                else:
                    embedding = GaussianFourierProjection(embed_dim=embed_dim)(
                        labels.view(-1, 1)
                    )
                embedding = (embedding + 1) / 2

            elif self.y2h_type == "resnet":
                self.model_mlp_y2h.eval()
                self.model_mlp_y2h = self.model_mlp_y2h.to(labels.device)
                embedding = self.model_mlp_y2h(labels)

        return embedding

    def fn_y2cov(self, labels):
        """
        Enhanced function to convert labels to covariance embedding with sophisticated
        multi-dimensional label handling.

        Parameters:
        - labels: Labels to embed [B, D] or [B]

        Returns:
        - embedding: Embedded labels for covariance [B, cov_dim]
        """
        embed_dim = self.cov_dim
        if embed_dim is None:
            # Default to image dimensions if not specified
            embed_dim = 32 * 32 * self.nc

        # Handle multi-dimensional labels
        if len(labels.shape) > 1 and labels.shape[1] > 1:
            device = labels.device

            if self.y2cov_type == "sinusoidal":
                # Generate embeddings for each dimension separately
                dim_embeddings = []
                for d in range(labels.shape[1]):
                    dim_labels = labels[:, d].view(len(labels))

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

                    # Ensure positive embeddings for covariance
                    dim_embed = dim_embed + 1
                    dim_embeddings.append(dim_embed)

                # Stack embeddings [D, B, embed_dim]
                stacked_embeddings = torch.stack(dim_embeddings)

                # Process based on combination method
                if self.dim_combination == "mean":
                    # Simple mean across dimensions
                    embedding = torch.mean(stacked_embeddings, dim=0)
                elif self.dim_combination == "weighted":
                    # Use learned weights
                    if self.dim_weights_learn:
                        weights = F.softmax(self.dim_weights, dim=0)
                        embedding = torch.sum(
                            stacked_embeddings * weights.view(-1, 1, 1), dim=0
                        )
                    else:
                        embedding = torch.mean(stacked_embeddings, dim=0)
                elif self.dim_combination == "attention":
                    # Apply attention
                    stacked_for_attn = stacked_embeddings.permute(1, 0, 2)
                    attn_scores = self.attention_layer(stacked_for_attn).squeeze(-1)
                    attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)
                    embedding = torch.sum(stacked_for_attn * attn_weights, dim=1)
                elif self.dim_combination == "cross":
                    # Cross-dimension interactions
                    B, D = labels.shape[0], labels.shape[1]
                    flat_embeddings = stacked_embeddings.permute(1, 0, 2).reshape(
                        B, D * embed_dim
                    )
                    embedding = self.cross_dim_interaction(flat_embeddings)
                else:
                    embedding = torch.mean(stacked_embeddings, dim=0)

            elif self.y2cov_type == "gaussian":
                # Process with Gaussian Fourier projection
                dim_embeddings = []
                for d in range(labels.shape[1]):
                    dim_labels = labels[:, d].view(-1, 1)
                    if hasattr(self, "gfp_cov"):
                        dim_embed = self.gfp_cov(dim_labels)
                    else:
                        dim_embed = GaussianFourierProjection(embed_dim=embed_dim)(
                            dim_labels
                        )
                    # Ensure positive embeddings for covariance
                    dim_embed = dim_embed + 1
                    dim_embeddings.append(dim_embed)

                # Similar combination methods as above
                stacked_embeddings = torch.stack(dim_embeddings)

                if self.dim_combination == "mean":
                    embedding = torch.mean(stacked_embeddings, dim=0)
                elif self.dim_combination == "weighted":
                    if self.dim_weights_learn:
                        weights = F.softmax(self.dim_weights, dim=0)
                        embedding = torch.sum(
                            stacked_embeddings * weights.view(-1, 1, 1), dim=0
                        )
                    else:
                        embedding = torch.mean(stacked_embeddings, dim=0)
                elif self.dim_combination == "attention":
                    stacked_for_attn = stacked_embeddings.permute(1, 0, 2)
                    attn_scores = self.attention_layer(stacked_for_attn).squeeze(-1)
                    attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)
                    embedding = torch.sum(stacked_for_attn * attn_weights, dim=1)
                elif self.dim_combination == "cross":
                    B, D = labels.shape[0], labels.shape[1]
                    flat_embeddings = stacked_embeddings.permute(1, 0, 2).reshape(
                        B, D * embed_dim
                    )
                    embedding = self.cross_dim_interaction(flat_embeddings)
                else:
                    embedding = torch.mean(stacked_embeddings, dim=0)

            elif self.y2cov_type == "resnet":
                # For ResNet-based embedding
                self.model_mlp_y2cov.eval()
                self.model_mlp_y2cov = self.model_mlp_y2cov.to(device)

                # Process each dimension separately
                dim_embeddings = []
                for d in range(labels.shape[1]):
                    dim_labels = labels[:, d].view(-1, 1)
                    dim_embed = self.model_mlp_y2cov(dim_labels)
                    dim_embeddings.append(dim_embed)

                # Combine embeddings based on selected method
                stacked_embeddings = torch.stack(dim_embeddings)

                if self.dim_combination == "mean":
                    embedding = torch.mean(stacked_embeddings, dim=0)
                elif self.dim_combination == "weighted":
                    if self.dim_weights_learn:
                        weights = F.softmax(self.dim_weights, dim=0)
                        embedding = torch.sum(
                            stacked_embeddings * weights.view(-1, 1, 1), dim=0
                        )
                    else:
                        embedding = torch.mean(stacked_embeddings, dim=0)
                elif self.dim_combination == "attention":
                    stacked_for_attn = stacked_embeddings.permute(1, 0, 2)
                    attn_scores = self.attention_layer(stacked_for_attn).squeeze(-1)
                    attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)
                    embedding = torch.sum(stacked_for_attn * attn_weights, dim=1)
                elif self.dim_combination == "cross":
                    B, D = labels.shape[0], labels.shape[1]
                    flat_embeddings = stacked_embeddings.permute(1, 0, 2).reshape(
                        B, D * embed_dim
                    )
                    embedding = self.cross_dim_interaction(flat_embeddings)
                else:
                    embedding = torch.mean(stacked_embeddings, dim=0)

        else:
            # Original single-dimension embedding
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
                embedding = embedding + 1  # make sure the embedding is not negative

            elif self.y2cov_type == "gaussian":
                if hasattr(self, "gfp_cov"):
                    embedding = self.gfp_cov(labels.view(-1, 1))
                else:
                    embedding = GaussianFourierProjection(embed_dim=embed_dim)(
                        labels.view(-1, 1)
                    )
                embedding = embedding + 1  # make sure the embedding is not negative

            elif self.y2cov_type == "resnet":
                self.model_mlp_y2cov.eval()
                self.model_mlp_y2cov = self.model_mlp_y2cov.to(labels.device)
                embedding = self.model_mlp_y2cov(labels)

        return embedding


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
    device="cuda",
):
    """learning rate decay"""

    def adjust_learning_rate_1(optimizer, epoch):
        """decrease the learning rate"""
        lr = lr_base

        num_decays = len(lr_decay_epochs)
        for decay_i in range(num_decays):
            if epoch >= lr_decay_epochs[decay_i]:
                lr = lr * lr_decay_factor
            # end if epoch
        # end for decay_i
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    net = net.to(device)
    criterion = nn.MSELoss()
    optimizer_resnet = torch.optim.SGD(
        net.parameters(), lr=lr_base, momentum=0.9, weight_decay=weight_decay
    )

    # resume training; load checkpoint
    if path_to_ckpt is not None and resume_epoch > 0:
        save_file = (
            path_to_ckpt
            + "/{}_ckpt_in_train/{}_checkpoint_epoch_{}.pth".format(
                net_name, net_name, resume_epoch
            )
        )
        checkpoint = torch.load(save_file)
        net.load_state_dict(checkpoint["net_state_dict"])
        optimizer_resnet.load_state_dict(checkpoint["optimizer_state_dict"])
        torch.set_rng_state(checkpoint["rng_state"])
    # end if

    start_tmp = timeit.default_timer()
    for epoch in range(resume_epoch, epochs):
        net.train()
        train_loss = 0
        adjust_learning_rate_1(optimizer_resnet, epoch)
        for _, (batch_train_images, batch_train_labels) in enumerate(trainloader):

            batch_train_images = batch_train_images.type(torch.float).to(device)
            batch_train_labels = (
                batch_train_labels.type(torch.float).view(-1, 1).to(device)
            )

            # Forward pass
            outputs, _ = net(batch_train_images)
            loss = criterion(outputs, batch_train_labels)

            # backward pass
            optimizer_resnet.zero_grad()
            loss.backward()
            optimizer_resnet.step()

            train_loss += loss.cpu().item()
        # end for batch_idx
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

        # save checkpoint
        if path_to_ckpt is not None and (
            ((epoch + 1) % 50 == 0) or (epoch + 1 == epochs)
        ):
            save_file = (
                path_to_ckpt
                + "/{}_ckpt_in_train/{}_checkpoint_epoch_{}.pth".format(
                    net_name, net_name, epoch + 1
                )
            )
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "net_state_dict": net.state_dict(),
                    "optimizer_state_dict": optimizer_resnet.state_dict(),
                    "rng_state": torch.get_rng_state(),
                },
                save_file,
            )
    # end for epoch

    return net


class label_dataset(torch.utils.data.Dataset):
    def __init__(self, labels):
        super(label_dataset, self).__init__()

        self.labels = labels
        self.n_samples = len(self.labels)

    def __getitem__(self, index):

        y = self.labels[index]
        return y

    def __len__(self):
        return self.n_samples


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
    device="cuda",
):
    """
    unique_labels_norm: an array of normalized unique labels
    """

    model_mlp = model_mlp.to(device)
    model_h2y = model_h2y.to(device)

    """ learning rate decay """

    def adjust_learning_rate_2(optimizer, epoch):
        """decrease the learning rate"""
        lr = lr_base

        num_decays = len(lr_decay_epochs)
        for decay_i in range(num_decays):
            if epoch >= lr_decay_epochs[decay_i]:
                lr = lr * lr_decay_factor
            # end if epoch
        # end for decay_i
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    assert np.max(unique_labels_norm) <= 1 and np.min(unique_labels_norm) >= 0

    # Handle multi-dimensional labels if necessary
    if len(unique_labels_norm.shape) > 1 and unique_labels_norm.shape[1] > 1:
        # Transform multi-dimensional labels into dataset
        trainset = []
        for label in unique_labels_norm:
            for dim_idx in range(len(label)):
                trainset.append(label[dim_idx])
        trainset = label_dataset(np.array(trainset))
    else:
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

            batch_labels = batch_labels.type(torch.float).view(-1, 1).to(device)

            # generate noises which will be added to labels
            batch_size_curr = len(batch_labels)
            batch_gamma = np.random.normal(0, 0.2, batch_size_curr)
            batch_gamma = (
                torch.from_numpy(batch_gamma).view(-1, 1).type(torch.float).to(device)
            )

            # add noise to labels
            batch_labels_noise = torch.clamp(batch_labels + batch_gamma, 0.0, 1.0)

            # Forward pass
            batch_hiddens_noise = model_mlp(batch_labels_noise)
            batch_rec_labels_noise = model_h2y(batch_hiddens_noise)

            loss = nn.MSELoss()(batch_rec_labels_noise, batch_labels_noise)

            # backward pass
            optimizer_mlp.zero_grad()
            loss.backward()
            optimizer_mlp.step()

            train_loss += loss.cpu().item()
        # end for batch_idx
        train_loss = train_loss / len(trainloader)

        print(
            "\n Train {}: [epoch {}/{}] train_loss:{:.4f} Time:{:.4f}".format(
                model_name,
                epoch + 1,
                epochs,
                train_loss,
                timeit.default_timer() - start_tmp,
            )
        )
    # end for epoch

    return model_mlp

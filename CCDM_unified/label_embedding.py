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
    def __init__(self, embed_dim, scale=30.):
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
        label_dim=1,  # Added parameter for label dimension
        dim_combination="attention"  # Strategy for combining dimension embeddings
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
        """
        self.dataset = dataset
        self.path_y2h = path_y2h
        self.path_y2cov = path_y2cov
        self.y2h_type = y2h_type
        self.y2cov_type = y2cov_type
        self.h_dim = h_dim
        self.cov_dim = cov_dim if cov_dim is not None else 64**2*nc
        self.batch_size = batch_size
        self.nc = nc
        self.label_dim = label_dim  # Number of dimensions in label vector
        self.dim_combination = dim_combination  # Strategy for combining dimension embeddings

        assert y2h_type in ['resnet', 'sinusoidal', 'gaussian']
        if y2cov_type is not None:
            assert y2cov_type in ['resnet', 'sinusoidal', 'gaussian']
            
        # Initialize attention network for dimension combination if needed
        if dim_combination == "attention" and label_dim > 1:
            self.attention_net = nn.Sequential(
                nn.Linear(h_dim, h_dim // 2),
                nn.ReLU(),
                nn.Linear(h_dim // 2, 1)
            ).to(device)
            
        # Initialize weighting parameters for dimension combination
        if dim_combination == "weighted" and label_dim > 1:
            self.dim_weights = nn.Parameter(torch.ones(label_dim) / label_dim)
            self.dim_weights = self.dim_weights.to(device)
        
        # For cross-dimension interactions
        if dim_combination == "cross" and label_dim > 1:
            self.cross_net = nn.Sequential(
                nn.Linear(h_dim * label_dim, h_dim * 2),
                nn.LayerNorm(h_dim * 2),
                nn.ReLU(),
                nn.Linear(h_dim * 2, h_dim),
                nn.LayerNorm(h_dim)
            ).to(device)
        
        ## Train or load embedding networks based on selected type
        if y2h_type == "resnet":
            os.makedirs(path_y2h, exist_ok=True)
            
            ## training setups
            epochs_resnet = 200  # Increased from 10 for better convergence
            epochs_mlp = 500
            base_lr_resnet = 1e-4
            base_lr_mlp = 1e-2
            
            ## Load training data
            # Get images and labels for training
            if hasattr(dataset, 'load_train_data'):
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
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, shuffle=True)
            
            # For multi-dimensional labels, we need unique labels for each dimension
            if self.label_dim > 1:
                unique_labels_norm = np.unique(train_labels, axis=0)
            else:
                unique_labels_norm = np.sort(np.array(list(set(train_labels))))
            
            ## Training embedding network for y2h
            resnet_y2h_filename_ckpt = os.path.join(self.path_y2h, 'ckpt_resnet_y2h_epoch_{}.pth'.format(epochs_resnet))
            mlp_y2h_filename_ckpt = os.path.join(self.path_y2h, 'ckpt_mlp_y2h_epoch_{}.pth'.format(epochs_mlp))
            
            # Initialize network with correct input channel count
            model_resnet_y2h = ResNet34_embed_y2h(dim_embed=self.h_dim, nc=self.nc)
            model_resnet_y2h = model_resnet_y2h.to(device)
            model_resnet_y2h = nn.DataParallel(model_resnet_y2h)
            
            model_mlp_y2h = model_y2h(dim_embed=self.h_dim)
            model_mlp_y2h = model_mlp_y2h.to(device)
            model_mlp_y2h = nn.DataParallel(model_mlp_y2h)
            
            # Training or loading existing checkpoint
            if not os.path.isfile(resnet_y2h_filename_ckpt):
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
                    label_dim=self.label_dim
                )
                # Save model
                torch.save({
                'net_state_dict': model_resnet_y2h.state_dict(),
                }, resnet_y2h_filename_ckpt)
            else:
                print("\n resnet_y2h ckpt already exists")
                print("\n Loading...")
                checkpoint = torch.load(resnet_y2h_filename_ckpt, map_location=device)
                model_resnet_y2h.load_state_dict(checkpoint['net_state_dict'])
            
            # Train or load MLP
            if not os.path.isfile(mlp_y2h_filename_ckpt):
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
                    device=device
                )
                # Save model
                torch.save({
                'net_state_dict': model_mlp_y2h.state_dict(),
                }, mlp_y2h_filename_ckpt)
            else:
                print("\n model mlp_y2h ckpt already exists")
                print("\n Loading...")
                checkpoint = torch.load(mlp_y2h_filename_ckpt, map_location=device)
                model_mlp_y2h.load_state_dict(checkpoint['net_state_dict'])
            
            self.model_mlp_y2h = model_mlp_y2h
            
            # Test with some sample labels
            print("\n Testing y2h embedding with sample labels...")
            if self.label_dim > 1:
                # For multi-dimensional labels, test with a few random samples
                indx_tmp = np.random.choice(len(unique_labels_norm), 5)
                labels_tmp = unique_labels_norm[indx_tmp]
                labels_tmp = torch.from_numpy(labels_tmp).float().to(device)
            else:
                # For scalar labels, use the original approach
                indx_tmp = np.arange(len(unique_labels_norm))
                np.random.shuffle(indx_tmp)
                indx_tmp = indx_tmp[:5]
                labels_tmp = unique_labels_norm[indx_tmp].reshape(-1, 1)
                labels_tmp = torch.from_numpy(labels_tmp).float().to(device)
            
            # Add noise for robustness testing
            if self.label_dim > 1:
                epsilons_tmp = np.random.normal(0, 0.05, size=labels_tmp.shape)
                epsilons_tmp = torch.from_numpy(epsilons_tmp).float().to(device)
            else:
                epsilons_tmp = np.random.normal(0, 0.2, len(labels_tmp))
                epsilons_tmp = torch.from_numpy(epsilons_tmp).view(-1, 1).float().to(device)
            
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
                    print("\n Original vs embedded shape:", labels_tmp.shape, "->", labels_hidden_tmp.shape)
            
        # Similar logic for y2cov if needed
        if y2cov_type == "resnet" and path_y2cov is not None:
            # Implementation similar to y2h but for covariance embedding
            os.makedirs(path_y2cov, exist_ok=True)
            
            ## Training setups
            epochs_resnet = 10
            epochs_mlp = 500
            base_lr_resnet = 1e-4
            base_lr_mlp = 1e-3
            
            ## Load training data
            if hasattr(dataset, 'load_train_data'):
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
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, shuffle=True)
            
            # Get unique labels
            if self.label_dim > 1:
                unique_labels_norm = np.unique(train_labels, axis=0)
            else:
                unique_labels_norm = np.sort(np.array(list(set(train_labels))))
            
            # File paths for checkpoints
            resnet_y2cov_filename_ckpt = os.path.join(self.path_y2cov, 'ckpt_resnet_y2cov_epoch_{}.pth'.format(epochs_resnet))
            mlp_y2cov_filename_ckpt = os.path.join(self.path_y2cov, 'ckpt_mlp_y2cov_epoch_{}.pth'.format(epochs_mlp))
            
            # Initialize networks
            model_resnet_y2cov = ResNet34_embed_y2cov(dim_embed=self.cov_dim, nc=self.nc)
            model_resnet_y2cov = model_resnet_y2cov.to(device)
            model_resnet_y2cov = nn.DataParallel(model_resnet_y2cov)
            
            model_mlp_y2cov = model_y2cov(dim_embed=self.cov_dim)
            model_mlp_y2cov = model_mlp_y2cov.to(device)
            model_mlp_y2cov = nn.DataParallel(model_mlp_y2cov)
            
            # Train or load ResNet
            if not os.path.isfile(resnet_y2cov_filename_ckpt):
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
                    label_dim=self.label_dim
                )
                torch.save({
                'net_state_dict': model_resnet_y2cov.state_dict(),
                }, resnet_y2cov_filename_ckpt)
            else:
                print("\n resnet_y2cov ckpt already exists")
                print("\n Loading...")
                checkpoint = torch.load(resnet_y2cov_filename_ckpt, map_location=device)
                model_resnet_y2cov.load_state_dict(checkpoint['net_state_dict'])
            
            # Train or load MLP
            if not os.path.isfile(mlp_y2cov_filename_ckpt):
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
                    device=device
                )
                torch.save({
                'net_state_dict': model_mlp_y2cov.state_dict(),
                }, mlp_y2cov_filename_ckpt)
            else:
                print("\n model mlp_y2cov ckpt already exists")
                print("\n Loading...")
                checkpoint = torch.load(mlp_y2cov_filename_ckpt, map_location=device)
                model_mlp_y2cov.load_state_dict(checkpoint['net_state_dict'])
            
            self.model_mlp_y2cov = model_mlp_y2cov
            
            # Test covariance embedding
            print("\n Testing y2cov embedding...")
            if self.label_dim > 1:
                # For multi-dimensional labels
                indx_tmp = np.random.choice(len(unique_labels_norm), 5)
                labels_tmp = unique_labels_norm[indx_tmp]
                labels_tmp = torch.from_numpy(labels_tmp).float().to(device)
            else:
                # For scalar labels
                indx_tmp = np.arange(len(unique_labels_norm))
                np.random.shuffle(indx_tmp)
                indx_tmp = indx_tmp[:5]
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
        
        # Handle multi-dimensional labels
        if len(labels.shape) > 1 and labels.shape[1] > 1:
            device = labels.device
            
            if self.y2h_type == "sinusoidal":
                # Process each dimension separately
                dim_embeddings = []
                for d in range(labels.shape[1]):
                    dim_labels = labels[:, d].view(len(labels))
                    
                    # Create sinusoidal embedding for this dimension
                    max_period = 10000
                    half = embed_dim // 2
                    freqs = torch.exp(
                        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
                    ).to(device=device)
                    args = dim_labels[:, None].float() * freqs[None]
                    dim_embed = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
                    
                    # Pad if needed
                    if embed_dim % 2:
                        dim_embed = torch.cat([dim_embed, torch.zeros_like(dim_embed[:, :1])], dim=-1)
                    
                    dim_embed = (dim_embed + 1) / 2  # Ensure in range [0,1]
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
                    embedding = torch.sum(stacked_embeddings * weights, dim=0)  # [B, embed_dim]
                    
                elif self.dim_combination == "attention":
                    # Attention-based weighting
                    stacked_for_attn = stacked_embeddings.permute(1, 0, 2)  # [B, D, embed_dim]
                    
                    # Generate attention scores
                    attn_scores = self.attention_net(stacked_for_attn).squeeze(-1)  # [B, D]
                    attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)  # [B, D, 1]
                    
                    # Apply attention weights
                    embedding = torch.sum(stacked_for_attn * attn_weights, dim=1)  # [B, embed_dim]
                    
                elif self.dim_combination == "cross":
                    # Consider cross-dimension interactions
                    stacked_for_cross = stacked_embeddings.permute(1, 0, 2)  # [B, D, embed_dim]
                    batch_size = stacked_for_cross.shape[0]
                    
                    # Flatten all dimension embeddings
                    flattened = stacked_for_cross.reshape(batch_size, -1)  # [B, D*embed_dim]
                    
                    # Process through cross-dimension network
                    embedding = self.cross_net(flattened)  # [B, embed_dim]
                    
                else:
                    # Default to mean
                    embedding = torch.mean(stacked_embeddings, dim=0)
                
            elif self.y2h_type == "gaussian":
                # Process each dimension separately
                dim_embeddings = []
                for d in range(labels.shape[1]):
                    dim_labels = labels[:, d].view(-1, 1)
                    
                    # Use Gaussian Fourier projection
                    gfp = GaussianFourierProjection(embed_dim=embed_dim).to(device)
                    dim_embed = gfp(dim_labels)
                    dim_embed = (dim_embed + 1) / 2  # Ensure in range [0,1]
                    dim_embeddings.append(dim_embed)
                
                # Combine using the same strategy as for sinusoidal
                stacked_embeddings = torch.stack(dim_embeddings)
                
                if self.dim_combination == "mean":
                    embedding = torch.mean(stacked_embeddings, dim=0)
                elif self.dim_combination == "weighted":
                    weights = F.softmax(self.dim_weights, dim=0)
                    weights = weights.view(-1, 1, 1)
                    embedding = torch.sum(stacked_embeddings * weights, dim=0)
                elif self.dim_combination == "attention":
                    stacked_for_attn = stacked_embeddings.permute(1, 0, 2)
                    attn_scores = self.attention_net(stacked_for_attn).squeeze(-1)
                    attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)
                    embedding = torch.sum(stacked_for_attn * attn_weights, dim=1)
                elif self.dim_combination == "cross":
                    stacked_for_cross = stacked_embeddings.permute(1, 0, 2)
                    batch_size = stacked_for_cross.shape[0]
                    flattened = stacked_for_cross.reshape(batch_size, -1)
                    embedding = self.cross_net(flattened)
                else:
                    embedding = torch.mean(stacked_embeddings, dim=0)
            
            elif self.y2h_type == "resnet":
                # ResNet-based embedding for multi-dimensional labels
                self.model_mlp_y2h.eval()
                self.model_mlp_y2h = self.model_mlp_y2h.to(device)
                
                # Process each dimension separately
                dim_embeddings = []
                for d in range(labels.shape[1]):
                    dim_labels = labels[:, d].view(-1, 1)
                    dim_embed = self.model_mlp_y2h(dim_labels)
                    dim_embeddings.append(dim_embed)
                
                # Combine dimension embeddings
                stacked_embeddings = torch.stack(dim_embeddings)
                
                if self.dim_combination == "mean":
                    embedding = torch.mean(stacked_embeddings, dim=0)
                elif self.dim_combination == "weighted":
                    weights = F.softmax(self.dim_weights, dim=0)
                    weights = weights.view(-1, 1, 1)
                    embedding = torch.sum(stacked_embeddings * weights, dim=0)
                elif self.dim_combination == "attention":
                    stacked_for_attn = stacked_embeddings.permute(1, 0, 2)
                    attn_scores = self.attention_net(stacked_for_attn).squeeze(-1)
                    attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)
                    embedding = torch.sum(stacked_for_attn * attn_weights, dim=1)
                elif self.dim_combination == "cross":
                    stacked_for_cross = stacked_embeddings.permute(1, 0, 2)
                    batch_size = stacked_for_cross.shape[0]
                    flattened = stacked_for_cross.reshape(batch_size, -1)
                    embedding = self.cross_net(flattened)
                else:
                    embedding = torch.mean(stacked_embeddings, dim=0)
        
        else:
            # Original code for scalar labels
            if self.y2h_type == "sinusoidal":
                max_period = 10000
                labels = labels.view(len(labels))
                half = embed_dim // 2
                freqs = torch.exp(
                    -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
                ).to(device=labels.device)
                args = labels[:, None].float() * freqs[None]
                embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
                if embed_dim % 2:
                    embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
                embedding = (embedding + 1) / 2  # make sure in [0,1]
                
            elif self.y2h_type == "gaussian":
                embedding = GaussianFourierProjection(embed_dim=embed_dim)(labels)
                embedding = (embedding + 1) / 2  # make sure in [0,1]
            
            elif self.y2h_type == "resnet":
                self.model_mlp_y2h.eval()
                self.model_mlp_y2h = self.model_mlp_y2h.to(labels.device)
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
        
        # Handle multi-dimensional labels
        if len(labels.shape) > 1 and labels.shape[1] > 1:
            device = labels.device
            
            if self.y2cov_type == "sinusoidal":
                # Process each dimension separately
                dim_embeddings = []
                for d in range(labels.shape[1]):
                    dim_labels = labels[:, d].view(len(labels))
                    
                    # Create sinusoidal embedding
                    max_period = 10000
                    half = embed_dim // 2
                    freqs = torch.exp(
                        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
                    ).to(device=device)
                    args = dim_labels[:, None].float() * freqs[None]
                    dim_embed = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
                    
                    if embed_dim % 2:
                        dim_embed = torch.cat([dim_embed, torch.zeros_like(dim_embed[:, :1])], dim=-1)
                    
                    # For covariance, ensure positive embeddings
                    dim_embed = dim_embed + 1
                    dim_embeddings.append(dim_embed)
                
                # Combine dimension embeddings
                stacked_embeddings = torch.stack(dim_embeddings)
                
                if self.dim_combination == "mean":
                    embedding = torch.mean(stacked_embeddings, dim=0)
                elif self.dim_combination == "weighted":
                    weights = F.softmax(self.dim_weights, dim=0)
                    weights = weights.view(-1, 1, 1)
                    embedding = torch.sum(stacked_embeddings * weights, dim=0)
                elif self.dim_combination == "attention":
                    stacked_for_attn = stacked_embeddings.permute(1, 0, 2)
                    attn_scores = self.attention_net(stacked_for_attn).squeeze(-1)
                    attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)
                    embedding = torch.sum(stacked_for_attn * attn_weights, dim=1)
                elif self.dim_combination == "cross":
                    stacked_for_cross = stacked_embeddings.permute(1, 0, 2)
                    batch_size = stacked_for_cross.shape[0]
                    flattened = stacked_for_cross.reshape(batch_size, -1)
                    embedding = self.cross_net(flattened)
                else:
                    embedding = torch.mean(stacked_embeddings, dim=0)
                    
            elif self.y2cov_type == "gaussian":
                # Process each dimension with Gaussian Fourier projection
                dim_embeddings = []
                for d in range(labels.shape[1]):
                    dim_labels = labels[:, d].view(-1, 1)
                    gfp = GaussianFourierProjection(embed_dim=embed_dim).to(device)
                    dim_embed = gfp(dim_labels)
                    # Ensure positive for covariance
                    dim_embed = dim_embed + 1
                    dim_embeddings.append(dim_embed)
                
                # Combine dimension embeddings
                stacked_embeddings = torch.stack(dim_embeddings)
                
                if self.dim_combination == "mean":
                    embedding = torch.mean(stacked_embeddings, dim=0)
                elif self.dim_combination == "weighted":
                    weights = F.softmax(self.dim_weights, dim=0)
                    weights = weights.view(-1, 1, 1)
                    embedding = torch.sum(stacked_embeddings * weights, dim=0)
                elif self.dim_combination == "attention":
                    stacked_for_attn = stacked_embeddings.permute(1, 0, 2)
                    attn_scores = self.attention_net(stacked_for_attn).squeeze(-1)
                    attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)
                    embedding = torch.sum(stacked_for_attn * attn_weights, dim=1)
                elif self.dim_combination == "cross":
                    stacked_for_cross = stacked_embeddings.permute(1, 0, 2)
                    batch_size = stacked_for_cross.shape[0]
                    flattened = stacked_for_cross.reshape(batch_size, -1)
                    embedding = self.cross_net(flattened)
                else:
                    embedding = torch.mean(stacked_embeddings, dim=0)
                    
            elif self.y2cov_type == "resnet":
                # ResNet-based covariance embedding
                self.model_mlp_y2cov.eval()
                self.model_mlp_y2cov = self.model_mlp_y2cov.to(device)
                
                # Process each dimension separately
                dim_embeddings = []
                for d in range(labels.shape[1]):
                    dim_labels = labels[:, d].view(-1, 1)
                    dim_embed = self.model_mlp_y2cov(dim_labels)
                    dim_embeddings.append(dim_embed)
                
                # Combine dimension embeddings
                stacked_embeddings = torch.stack(dim_embeddings)
                
                if self.dim_combination == "mean":
                    embedding = torch.mean(stacked_embeddings, dim=0)
                elif self.dim_combination == "weighted":
                    weights = F.softmax(self.dim_weights, dim=0)
                    weights = weights.view(-1, 1, 1)
                    embedding = torch.sum(stacked_embeddings * weights, dim=0)
                elif self.dim_combination == "attention":
                    stacked_for_attn = stacked_embeddings.permute(1, 0, 2)
                    attn_scores = self.attention_net(stacked_for_attn).squeeze(-1)
                    attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)
                    embedding = torch.sum(stacked_for_attn * attn_weights, dim=1)
                elif self.dim_combination == "cross":
                    stacked_for_cross = stacked_embeddings.permute(1, 0, 2)
                    batch_size = stacked_for_cross.shape[0]
                    flattened = stacked_for_cross.reshape(batch_size, -1)
                    embedding = self.cross_net(flattened)
                else:
                    embedding = torch.mean(stacked_embeddings, dim=0)
                
        else:
            # Original code for scalar labels
            if self.y2cov_type == "sinusoidal":
                max_period = 10000
                labels = labels.view(len(labels))
                half = embed_dim // 2
                freqs = torch.exp(
                    -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
                ).to(device=labels.device)
                args = labels[:, None].float() * freqs[None]
                embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
                if embed_dim % 2:
                    embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
                embedding = embedding + 1  # make sure embedding is not negative
                
            elif self.y2cov_type == "gaussian":
                embedding = GaussianFourierProjection(embed_dim=embed_dim)(labels)
                embedding = embedding + 1  # make sure embedding is not negative
            
            elif self.y2cov_type == "resnet":
                self.model_mlp_y2cov.eval()
                self.model_mlp_y2cov = self.model_mlp_y2cov.to(labels.device)
                embedding = self.model_mlp_y2cov(labels)
        
        return embedding


def train_resnet(net, net_name, trainloader, epochs=200, resume_epoch=0, lr_base=0.01, 
                lr_decay_factor=0.1, lr_decay_epochs=[80, 140], weight_decay=1e-4, 
                path_to_ckpt=None, device="cuda", label_dim=1):
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
        device: Device to train on
        label_dim: Dimension of regression labels
        
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
            param_group['lr'] = lr

    net = net.to(device)
    criterion = nn.MSELoss()
    optimizer_resnet = torch.optim.SGD(net.parameters(), lr=lr_base, momentum=0.9, weight_decay=weight_decay)

    # Resume training; load checkpoint
    if path_to_ckpt is not None and resume_epoch > 0:
        save_file = path_to_ckpt + "/{}_ckpt_in_train/{}_checkpoint_epoch_{}.pth".format(net_name, net_name, resume_epoch)
        checkpoint = torch.load(save_file, map_location=device)
        net.load_state_dict(checkpoint['net_state_dict'])
        optimizer_resnet.load_state_dict(checkpoint['optimizer_state_dict'])
        torch.set_rng_state(checkpoint['rng_state'])

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
                batch_train_images = batch_data['design'] if 'design' in batch_data else batch_data['image']
                batch_train_labels = batch_data['labels']
            
            batch_train_images = batch_train_images.type(torch.float).to(device)
            
            # Handle multi-dimensional labels
            if label_dim > 1:
                # For multi-dimensional labels, train separate models for each dimension
                # or use first dimension as representative (simplification)
                batch_train_labels = batch_train_labels[:, 0:1].type(torch.float).to(device)
            else:
                # For scalar labels
                batch_train_labels = batch_train_labels.type(torch.float).view(-1, 1).to(device)
            
            # Forward pass
            outputs, _ = net(batch_train_images)
            loss = criterion(outputs, batch_train_labels)
            
            # Backward pass
            optimizer_resnet.zero_grad()
            loss.backward()
            optimizer_resnet.step()
            
            train_loss += loss.cpu().item()
        
        train_loss = train_loss / len(trainloader)
        
        print('Train {} for embedding: [epoch {}/{}] train_loss:{:.4f} Time:{:.4f}'.format(
            net_name, epoch+1, epochs, train_loss, timeit.default_timer()-start_tmp))
        
        # Save checkpoint
        if path_to_ckpt is not None and (((epoch+1) % 50 == 0) or (epoch+1 == epochs)):
            save_file = path_to_ckpt + "/{}_ckpt_in_train/{}_checkpoint_epoch_{}.pth".format(net_name, net_name, epoch+1)
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            torch.save({
                'epoch': epoch,
                'net_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer_resnet.state_dict(),
                'rng_state': torch.get_rng_state()
            }, save_file)
    
    return net


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


def train_mlp(unique_labels_norm, model_mlp, model_name, model_h2y, epochs=500, 
             lr_base=0.01, lr_decay_factor=0.1, lr_decay_epochs=[150, 250, 350], 
             weight_decay=1e-4, batch_size=128, device="cuda", label_dim=1):
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
        device: Device to train on
        label_dim: Dimension of regression labels
        
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
            param_group['lr'] = lr
    
    # Verify that labels are normalized
    if isinstance(unique_labels_norm, np.ndarray):
        assert np.max(unique_labels_norm) <= 1 and np.min(unique_labels_norm) >= 0
    
    # Handle multi-dimensional labels differently
    if label_dim > 1 and len(unique_labels_norm.shape) > 1 and unique_labels_norm.shape[1] > 1:
        # For multi-dimensional labels, create a dataset with individual dimensions
        flattened_labels = []
        for label in unique_labels_norm:
            for dim_idx in range(len(label)):
                flattened_labels.append([label[dim_idx]])
        
        trainset = label_dataset(np.array(flattened_labels))
    else:
        # For scalar labels, use original approach
        trainset = label_dataset(unique_labels_norm)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    
    model_h2y.eval()
    optimizer_mlp = torch.optim.SGD(model_mlp.parameters(), lr=lr_base, momentum=0.9, weight_decay=weight_decay)
    
    start_tmp = timeit.default_timer()
    for epoch in range(epochs):
        model_mlp.train()
        train_loss = 0
        adjust_learning_rate_2(optimizer_mlp, epoch)
        
        for _, batch_labels in enumerate(trainloader):
            # If the dataset returns a multi-dim tensor from __getitem__,
            # ensure it's properly shaped as a batch of scalar labels
            if isinstance(batch_labels, np.ndarray) and batch_labels.ndim > 2:
                batch_labels = torch.from_numpy(batch_labels.reshape(batch_labels.shape[0], -1))
            
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
                batch_gamma = torch.from_numpy(batch_gamma).view(-1, 1).type(torch.float).to(device)
            
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
            print('Train {}: [epoch {}/{}] train_loss:{:.4f} Time:{:.4f}'.format(
                model_name, epoch+1, epochs, train_loss, timeit.default_timer()-start_tmp))
    
    return model_mlp

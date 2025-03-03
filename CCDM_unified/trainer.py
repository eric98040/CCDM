import numpy as np
import math
import copy
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count
import os

import torch
from torch import nn, einsum
from torch.cuda.amp import autocast
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam

from torchvision import transforms as T, utils

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from PIL import Image
from tqdm.auto import tqdm
from accelerate import Accelerator

from ema_pytorch import EMA
from diffusion import generate_random_vectors, compute_distance, compute_projection
from utils import (
    cycle,
    divisible_by,
    exists,
    normalize_images,
    random_hflip,
    random_rotate,
    random_vflip,
)

# from moviepy.editor import ImageSequenceClip


class Trainer(object):
    def __init__(
        self,
        data_name,
        diffusion_model,
        train_images,
        train_labels,
        vicinal_params,
        *,
        train_batch_size=16,
        gradient_accumulate_every=1,
        train_lr=1e-4,
        train_num_steps=100000,
        ema_update_after_step=1e30,
        ema_update_every=10,
        ema_decay=0.995,
        adam_betas=(0.9, 0.99),
        sample_every=1000,
        save_every=1000,
        results_folder="./results",
        amp=False,
        mixed_precision_type="fp16",
        split_batches=True,
        max_grad_norm=1.0,
        y_visual=None,
        nrow_visual=6,
        cond_scale_visual=1.5,
        vicinity_type="shv",
        kappa=None,
        sigma_delta=None,
        vector_type="gaussian",
        num_projections=1,
        distance="l2",
        label_dim=1,
        adaptive_slicing=False,
        hyperparameter="rule_of_thumb",
        percentile=5.0,
    ):
        super().__init__()

        # dataset
        ## training images are not normalized here !!!
        self.data_name = data_name
        self.train_images = train_images
        self.train_labels = train_labels
        self.unique_train_labels = np.sort(np.array(list(set(train_labels))))
        assert train_images.max() > 1.0
        assert train_labels.min() >= 0 and train_labels.max() <= 1.0
        print(
            "\n Training labels' range is [{},{}].".format(
                train_labels.min(), train_labels.max()
            )
        )

        # vicinal params
        self.kernel_sigma = vicinal_params["kernel_sigma"]
        self.kappa = vicinal_params["kappa"]
        self.nonzero_soft_weight_threshold = vicinal_params[
            "nonzero_soft_weight_threshold"
        ]

        # visualize
        self.y_visual = y_visual
        self.cond_scale_visual = cond_scale_visual
        self.nrow_visual = nrow_visual

        # accelerator
        self.accelerator = Accelerator(
            # split_batches = split_batches,
            mixed_precision=mixed_precision_type if amp else "no"
        )

        # model
        self.model = diffusion_model  # diffusion model instead of unet
        self.channels = diffusion_model.channels

        # sampling and training hyperparameters
        # assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        # self.num_samples = num_samples
        self.sample_every = sample_every
        self.save_every = save_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        assert (
            train_batch_size * gradient_accumulate_every
        ) >= 16, f"your effective batch size (train_batch_size x gradient_accumulate_every) should be at least 16 or above"

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        self.max_grad_norm = max_grad_norm

        # optimizer
        self.opt = Adam(diffusion_model.parameters(), lr=train_lr, betas=adam_betas)

        # for logging results in a folder periodically
        if self.accelerator.is_main_process:
            self.ema = EMA(
                diffusion_model,
                update_after_step=ema_update_after_step,
                beta=ema_decay,
                update_every=ema_update_every,
            )
            self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)

        # step counter state
        self.step = 0

        # prepare model, dataloader, optimizer with accelerator
        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

        # additional parameters for Sliced CCDM
        self.vicinity_type = vicinity_type
        self.kappa = kappa
        self.sigma_delta = sigma_delta
        self.vector_type = vector_type
        self.num_projections = num_projections
        self.distance = distance
        self.label_dim = label_dim
        self.adaptive_slicing = adaptive_slicing
        self.hyperparameter = hyperparameter
        self.percentile = percentile

        # Compute kappa and sigma_delta if needed
        if not adaptive_slicing and (kappa is None or sigma_delta is None):
            self.sigma_delta, self.kappa = self.compute_hyperparameters()

    def compute_hyperparameters(self):
        """
        Compute sigma_delta and kappa based on the training labels
        """
        if self.hyperparameter == "rule_of_thumb":
            # Compute sigma_delta using rule of thumb formula
            if len(self.train_labels.shape) > 1 and self.train_labels.shape[1] > 1:
                std_label = np.std(self.train_labels, axis=0)
                sigma_delta = 1.06 * std_label * (len(self.train_labels)) ** (-1 / 5)
            else:
                std_label = np.std(self.train_labels)
                sigma_delta = 1.06 * std_label * (len(self.train_labels)) ** (-1 / 5)

            # Compute kappa based on unique labels
            unique_labels = np.unique(self.train_labels, axis=0)
            n_unique = len(unique_labels)

            if n_unique > 1:
                # Sort unique labels
                idx = np.lexsort(
                    [
                        unique_labels[:, i]
                        for i in range(unique_labels.shape[1] - 1, -1, -1)
                    ]
                )
                sorted_labels = unique_labels[idx]

                # Compute differences between consecutive sorted labels
                diff_list = []
                for i in range(1, n_unique):
                    diff = np.linalg.norm(sorted_labels[i] - sorted_labels[i - 1])
                    diff_list.append(diff)

                kappa_base = max(diff_list)

                is_hard_vicinity = self.vicinity_type in ["hv", "shv"]
                if is_hard_vicinity:
                    kappa = kappa_base
                else:  # sv, ssv
                    kappa = 1 / kappa_base**2
            else:
                # Fallback for single unique label case
                kappa = 0.01 if is_hard_vicinity else 10000

        else:  # percentile method
            # Calculate pairwise distances between all labels
            distances = []
            for i in range(len(self.train_labels)):
                for j in range(i + 1, len(self.train_labels)):
                    if self.distance == "l2":
                        dist = np.linalg.norm(
                            self.train_labels[i] - self.train_labels[j]
                        )
                    elif self.distance == "l1":
                        dist = np.sum(
                            np.abs(self.train_labels[i] - self.train_labels[j])
                        )
                    else:  # cosine
                        dist = 1 - np.dot(
                            self.train_labels[i], self.train_labels[j]
                        ) / (
                            np.linalg.norm(self.train_labels[i])
                            * np.linalg.norm(self.train_labels[j])
                        )
                    distances.append(dist)

            # Set kappa to the percentile of distances
            kappa = np.percentile(distances, self.percentile)

            # Set sigma_delta to be proportional to kappa
            sigma_delta = kappa / 3  # Empirical choice

            if self.vicinity_type in ["sv", "ssv"]:
                kappa = 1 / kappa**2

        print(f"\n Using {self.hyperparameter} method to compute hyperparameters >>>")
        print(f"\r Sigma_delta: {sigma_delta}, Kappa: {kappa}")

        return sigma_delta, kappa

    def compute_adaptive_params(self, batch_labels):
        """
        Dynamically compute kappa and sigma_delta for the current batch
        """
        # Similar to compute_hyperparameters but for a batch
        if self.hyperparameter == "rule_of_thumb":
            std_label = np.std(batch_labels, axis=0)
            sigma_delta = 1.06 * std_label * (len(batch_labels)) ** (-1 / 5)

            # For kappa, use the minimum distance between any pair in the batch
            distances = []
            for i in range(len(batch_labels)):
                for j in range(i + 1, len(batch_labels)):
                    dist = np.linalg.norm(batch_labels[i] - batch_labels[j])
                    distances.append(dist)

            if len(distances) > 0:
                kappa_base = min(distances)
                kappa = (
                    kappa_base
                    if self.vicinity_type in ["hv", "shv"]
                    else 1 / kappa_base**2
                )
            else:
                kappa = 0.01 if self.vicinity_type in ["hv", "shv"] else 10000

        else:  # percentile method
            # Calculate pairwise distances in batch
            distances = []
            for i in range(len(batch_labels)):
                for j in range(i + 1, len(batch_labels)):
                    if self.distance == "l2":
                        dist = np.linalg.norm(batch_labels[i] - batch_labels[j])
                    elif self.distance == "l1":
                        dist = np.sum(np.abs(batch_labels[i] - batch_labels[j]))
                    else:  # cosine
                        dist = 1 - np.dot(batch_labels[i], batch_labels[j]) / (
                            np.linalg.norm(batch_labels[i])
                            * np.linalg.norm(batch_labels[j])
                        )
                    distances.append(dist)

            if len(distances) > 0:
                kappa = np.percentile(distances, self.percentile)
                sigma_delta = kappa / 3

                if self.vicinity_type in ["sv", "ssv"]:
                    kappa = 1 / kappa**2
            else:
                sigma_delta = 0.01
                kappa = 0.01 if self.vicinity_type in ["hv", "shv"] else 10000

        return sigma_delta, kappa

    def sample_labels_batch(self, batch_size):
        """
        Sample a batch of labels from the unique training labels
        """
        # For multi-dimensional labels, we need to handle it differently
        unique_labels = np.unique(self.train_labels, axis=0)
        indices = np.random.choice(len(unique_labels), size=batch_size, replace=True)
        return unique_labels[indices]

    def sample_real_indices_sliced(self, target_labels):
        """
        Sample indices of real images with labels in the sliced vicinity of target_labels
        using multiple projection vectors.

        Parameters:
        - target_labels: Target labels to find vicinity samples for [B, D]

        Returns:
        - batch_real_indx: Indices of selected real samples [B]
        """
        batch_size = len(target_labels)
        batch_real_indx = np.zeros(batch_size, dtype=int)

        # Get label dimension
        dim = self.label_dim
        device = target_labels.device

        # Generate random vectors for projection (use multiple vectors)
        v_all = generate_random_vectors(
            self.vector_type, dim, self.num_projections, device
        )

        # Convert training labels to tensor once
        train_labels_tensor = torch.from_numpy(self.train_labels).float().to(device)

        # Process each target label
        for j in range(batch_size):
            target_label = target_labels[j : j + 1]  # Keep dimension [1, D]

            # Track matched indices across all projections
            all_matched_indices = []

            # Try each projection vector
            for v_idx in range(self.num_projections):
                v = v_all[v_idx : v_idx + 1]  # [1, D]

                # Normalize projection vector
                v_norm = F.normalize(v, dim=1)

                # Project all training labels and target label
                proj_train_labels = torch.matmul(
                    train_labels_tensor, v_norm.t()
                ).squeeze(
                    -1
                )  # [N]
                proj_target_label = torch.matmul(target_label, v_norm.t()).squeeze(
                    -1
                )  # [1]

                # Calculate projection differences
                proj_diff = torch.abs(proj_train_labels - proj_target_label)

                # Adjust kappa based on vector norm for numerical stability
                effective_kappa = self.kappa * torch.norm(v)

                # Find indices within vicinity
                indx_real_in_vicinity = torch.where(proj_diff <= effective_kappa)[0]

                # Store found indices
                if len(indx_real_in_vicinity) > 0:
                    all_matched_indices.append(indx_real_in_vicinity)

            # Combine all matched indices
            if len(all_matched_indices) > 0:
                # Concatenate all indices
                combined_indices = torch.cat(all_matched_indices)

                # Count frequency of each index (samples that match multiple projections are preferred)
                unique_indices, counts = torch.unique(
                    combined_indices, return_counts=True
                )

                # Sort by frequency (descending)
                sorted_indices = torch.argsort(counts, descending=True)
                best_indices = unique_indices[sorted_indices]

                # Select top-k indices based on frequency
                top_k = min(10, len(best_indices))
                candidate_indices = best_indices[:top_k]

                # Randomly select one from the top candidates
                chosen_idx = torch.randint(
                    0, len(candidate_indices), (1,), device=device
                )[0]
                batch_real_indx[j] = candidate_indices[chosen_idx].cpu().numpy()
            else:
                # Fallback: If no matches found across all projections, use nearest neighbor
                if train_labels_tensor.dim() > 1 and train_labels_tensor.shape[1] > 1:
                    # Multi-dimensional case
                    diff = train_labels_tensor - target_label
                    dist = torch.sqrt((diff**2).sum(dim=1))
                else:
                    # 1D case
                    dist = torch.abs(train_labels_tensor - target_label)

                # Get index of closest training sample
                closest_idx = torch.argmin(dist)
                batch_real_indx[j] = closest_idx.cpu().numpy()

        return batch_real_indx

    def sample_real_indices_vicinity(self, target_labels):
        """
        Sample indices of real images with labels in the vicinity of target_labels
        """
        batch_size = len(target_labels)
        batch_real_indx = np.zeros(batch_size, dtype=int)

        # Convert target_labels to tensor for distance computation
        target_labels_tensor = torch.from_numpy(target_labels).float().to(self.device)
        train_labels_tensor = (
            torch.from_numpy(self.train_labels).float().to(self.device)
        )

        for j in range(batch_size):
            # Compute distances from all training labels to this target label
            distances = compute_distance(
                train_labels_tensor, target_labels_tensor[j].unsqueeze(0), self.distance
            )

            # Find indices with distances less than kappa
            if self.vicinity_type == "hv":
                indx_real_in_vicinity = torch.where(distances <= self.kappa)[0]
            else:  # 'sv'
                # For soft vicinity, we use all indices but weight them later
                indx_real_in_vicinity = torch.arange(
                    len(self.train_labels), device=self.device
                )

            # If none found, fall back to random selection
            if len(indx_real_in_vicinity) < 1:
                indx_real_in_vicinity = torch.randint(
                    0, len(self.train_labels), (1,), device=self.device
                )

            # Randomly select one of the matching indices
            chosen_idx = torch.randint(
                0, len(indx_real_in_vicinity), (1,), device=self.device
            )
            batch_real_indx[j] = indx_real_in_vicinity[chosen_idx].cpu().numpy()

        return batch_real_indx

    def process_images(self, batch_real_indx):
        """
        Process image batch from indices (apply data augmentation, etc.)
        """
        batch_images = self.train_images[batch_real_indx]

        # Apply data augmentation based on dataset
        if self.data_name == "UTKFace":
            batch_images = random_hflip(batch_images)
        elif self.data_name == "Cell200":
            batch_images = random_rotate(batch_images)
            batch_images = random_hflip(batch_images)
            batch_images = random_vflip(batch_images)

        # Normalize images
        batch_images = (
            torch.from_numpy(normalize_images(batch_images, to_neg_one_to_one=False))
            .type(torch.float)
            .to(self.device)
        )

        return batch_images

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            "step": self.step,
            "model": self.accelerator.get_state_dict(self.model),
            "opt": self.opt.state_dict(),
            "ema": self.ema.state_dict(),
            "scaler": (
                self.accelerator.scaler.state_dict()
                if exists(self.accelerator.scaler)
                else None
            ),
            # 'version': __version__
        }

        torch.save(data, str(self.results_folder / f"model-{milestone}.pt"))
        # torch.save(data, str(self.results_folder / f'model-{self.step}.pt'))

    def load(self, milestone, return_ema=False, return_unet=False):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(
            str(self.results_folder / f"model-{milestone}.pt"),
            map_location=device,
            weights_only=True,
        )

        self.model = self.accelerator.unwrap_model(self.model)
        self.model.load_state_dict(data["model"])

        self.step = data["step"]
        self.opt.load_state_dict(data["opt"])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])
            if return_ema:
                return self.ema

        if "version" in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data["scaler"]):
            self.accelerator.scaler.load_state_dict(data["scaler"])

        if return_unet:
            return self.model.model  # take unet

    def train(self, fn_y2h):
        accelerator = self.accelerator
        device = accelerator.device

        log_filename = os.path.join(
            self.results_folder, "log_loss_niters{}.txt".format(self.train_num_steps)
        )
        if not os.path.isfile(log_filename):
            logging_file = open(log_filename, "w")
            logging_file.close()
        with open(log_filename, "a") as file:
            file.write(
                "\n==================================================================================================="
            )

        with tqdm(
            initial=self.step,
            total=self.train_num_steps,
            disable=not accelerator.is_main_process,
        ) as pbar:
            while self.step < self.train_num_steps:
                total_loss = 0.0

                for _ in range(self.gradient_accumulate_every):
                    # Handle different vicinity types
                    if self.vicinity_type in ["shv", "ssv"]:
                        # Handle Sliced Vicinal Loss for multi-dimensional labels
                        # Draw batch_size target labels from unique_train_labels
                        batch_target_labels_in_dataset = self.sample_labels_batch(
                            self.batch_size
                        )

                        # Generate random vectors for projection
                        if self.adaptive_slicing:
                            # Dynamically compute kappa and sigma_delta for this batch
                            self.sigma_delta, self.kappa = self.compute_adaptive_params(
                                batch_target_labels_in_dataset
                            )

                        # Add Gaussian noise to target labels
                        batch_epsilons = np.random.normal(
                            0, self.sigma_delta, (self.batch_size, self.label_dim)
                        )
                        batch_target_labels = (
                            batch_target_labels_in_dataset + batch_epsilons
                        )

                        # Set up training batch
                        batch_target_labels = (
                            torch.from_numpy(batch_target_labels)
                            .type(torch.float)
                            .cuda()
                        )

                        # Sample real images with similar projected labels
                        batch_real_indx = self.sample_real_indices_sliced(
                            batch_target_labels
                        )

                        # Prepare data
                        batch_images = self.process_images(batch_real_indx)
                        batch_labels = self.train_labels[batch_real_indx]
                        batch_labels = (
                            torch.from_numpy(batch_labels).type(torch.float).cuda()
                        )

                        # Define weight vector based on vicinity type
                        if self.vicinity_type == "shv":
                            # Will be applied inside diffusion.p_losses
                            vicinal_weights = torch.ones(
                                self.batch_size, dtype=torch.float
                            ).cuda()
                        else:  # 'ssv'
                            # Will be applied inside diffusion.p_losses
                            vicinal_weights = torch.ones(
                                self.batch_size, dtype=torch.float
                            ).cuda()

                        # Forward through model with Sliced Vicinal Loss
                        with self.accelerator.autocast():
                            loss = self.model(
                                batch_images,
                                labels_emb=fn_y2h(batch_labels),
                                labels=batch_labels,
                                vicinal_weights=vicinal_weights,
                                vicinity_type=self.vicinity_type,
                                kappa=self.kappa,
                                vector_type=self.vector_type,
                                num_projections=self.num_projections,
                            )
                            loss = loss / self.gradient_accumulate_every
                            total_loss += loss.item()

                    elif self.vicinity_type in ["hv", "sv"]:
                        # Original hard or soft vicinal loss but supporting multi-dimensional labels
                        # Draw batch_size target labels from unique_train_labels
                        batch_target_labels_in_dataset = self.sample_labels_batch(
                            self.batch_size
                        )

                        # Add Gaussian noise to target labels
                        batch_epsilons = np.random.normal(
                            0, self.sigma_delta, (self.batch_size, self.label_dim)
                        )
                        batch_target_labels = (
                            batch_target_labels_in_dataset + batch_epsilons
                        )

                        # Sample real images with similar labels
                        batch_real_indx = self.sample_real_indices_vicinity(
                            batch_target_labels
                        )

                        # Prepare data
                        batch_images = self.process_images(batch_real_indx)
                        batch_labels = self.train_labels[batch_real_indx]
                        batch_labels = (
                            torch.from_numpy(batch_labels).type(torch.float).cuda()
                        )
                        batch_target_labels = (
                            torch.from_numpy(batch_target_labels)
                            .type(torch.float)
                            .cuda()
                        )

                        # Define weight vector based on vicinity type
                        if self.vicinity_type == "hv":
                            vicinal_weights = torch.ones(
                                self.batch_size, dtype=torch.float
                            ).cuda()
                            # Apply hard vicinity weights based on distance
                            for i in range(self.batch_size):
                                dist = compute_distance(
                                    batch_labels[i],
                                    batch_target_labels[i],
                                    self.distance,
                                )
                                if dist > self.kappa:
                                    vicinal_weights[i] = 0.0
                        else:  # 'sv'
                            vicinal_weights = torch.zeros(
                                self.batch_size, dtype=torch.float
                            ).cuda()
                            nu = 1.0 / (self.kappa**2)
                            for i in range(self.batch_size):
                                dist = compute_distance(
                                    batch_labels[i],
                                    batch_target_labels[i],
                                    self.distance,
                                )
                                vicinal_weights[i] = torch.exp(-nu * (dist**2))

                        # Forward through model with Vicinal Loss
                        with self.accelerator.autocast():
                            loss = self.model(
                                batch_images,
                                labels_emb=fn_y2h(batch_labels),
                                labels=batch_labels,
                                vicinal_weights=vicinal_weights,
                            )
                            loss = loss / self.gradient_accumulate_every
                            total_loss += loss.item()

                    else:  # No vicinity (original training approach)
                        batch_real_indx = np.random.choice(
                            np.arange(len(self.train_images)),
                            size=self.batch_size,
                            replace=True,
                        )
                        batch_images = self.process_images(batch_real_indx)
                        batch_labels = self.train_labels[batch_real_indx]
                        batch_labels = (
                            torch.from_numpy(batch_labels).type(torch.float).cuda()
                        )

                        with self.accelerator.autocast():
                            loss = self.model(
                                batch_images,
                                labels_emb=fn_y2h(batch_labels),
                                labels=batch_labels,
                                vicinal_weights=None,
                            )
                            loss = loss / self.gradient_accumulate_every
                            total_loss += loss.item()

                    self.accelerator.backward(loss)

                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                pbar.set_description(f"loss: {total_loss:.4f}")

                if self.step % 500 == 0:
                    with open(log_filename, "a") as file:
                        file.write(f"\r Step: {self.step}, Loss: {total_loss:.4f}.")

                accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step != 0 and divisible_by(self.step, self.sample_every):
                        self.ema.ema_model.eval()
                        with torch.inference_mode():
                            gen_imgs = self.ema.ema_model.ddim_sample(
                                labels_emb=fn_y2h(self.y_visual),
                                labels=self.y_visual,
                                shape=(
                                    self.y_visual.shape[0],
                                    self.channels,
                                    self.image_size,
                                    self.image_size,
                                ),
                                cond_scale=self.cond_scale_visual,
                            )

                            gen_imgs = gen_imgs.detach().cpu()
                            if gen_imgs.min() < 0 or gen_imgs.max() > 1:
                                print(
                                    f"\r Generated images are out of range. (min={gen_imgs.min()}, max={gen_imgs.max()})"
                                )
                            gen_imgs = torch.clip(gen_imgs, 0, 1)
                            assert gen_imgs.size(1) == self.channels
                            utils.save_image(
                                gen_imgs.data,
                                str(self.results_folder) + f"/sample_{self.step}.png",
                                nrow=self.nrow_visual,
                                normalize=False,
                                padding=1,
                            )

                    if self.step != 0 and divisible_by(self.step, self.save_every):
                        milestone = self.step
                        self.ema.ema_model.eval()
                        self.save(milestone)

                pbar.update(1)

            accelerator.print("training complete")
        ## end def

    def sample_given_labels(
        self,
        given_labels,
        fn_y2h,
        batch_size,
        denorm=True,
        to_numpy=True,
        verbose=False,
        sampler="ddpm",
        cond_scale=6.0,
        sample_timesteps=1000,
        ddim_eta=0,
    ):
        """
        Generate samples based on given labels
        :given_labels: normalized labels
        :fn_y2h: label embedding function
        """
        accelerator = self.accelerator
        device = accelerator.device

        assert given_labels.min() >= 0 and given_labels.max() <= 1.0
        nfake = len(given_labels)

        if batch_size > nfake:
            batch_size = nfake
        fake_images = []
        assert nfake % batch_size == 0

        tmp = 0
        while tmp < nfake:
            batch_fake_labels = (
                torch.from_numpy(given_labels[tmp : (tmp + batch_size)])
                .type(torch.float)
                .view(-1)
                .cuda()
            )
            self.ema.ema_model.eval()
            with torch.inference_mode():
                if sampler == "ddpm":
                    batch_fake_images = self.ema.ema_model.sample(
                        labels_emb=fn_y2h(batch_fake_labels),
                        labels=batch_fake_labels,
                        cond_scale=cond_scale,
                        # preset_sampling_timesteps=sample_timesteps,
                    )
                elif sampler == "ddim":
                    batch_fake_images = self.ema.ema_model.ddim_sample(
                        labels_emb=fn_y2h(batch_fake_labels),
                        labels=batch_fake_labels,
                        shape=(
                            batch_fake_labels.shape[0],
                            self.channels,
                            self.image_size,
                            self.image_size,
                        ),
                        cond_scale=cond_scale,
                        # preset_sampling_timesteps = sample_timesteps,
                        # preset_ddim_sampling_eta = ddim_eta, # 1 for ddpm, 0 for ddim
                    )

                batch_fake_images = batch_fake_images.detach().cpu()

            if denorm:  # denorm imgs to save memory
                # assert batch_fake_images.max().item()<=1.0 and batch_fake_images.min().item()>=0
                if batch_fake_images.min() < 0 or batch_fake_images.max() > 1:
                    print(
                        "\r Generated images are out of range. (min={}, max={})".format(
                            batch_fake_images.min(), batch_fake_images.max()
                        )
                    )
                batch_fake_images = torch.clip(batch_fake_images, 0, 1)
                batch_fake_images = (batch_fake_images * 255.0).type(torch.uint8)

            fake_images.append(batch_fake_images.detach().cpu())
            tmp += batch_size
            if verbose:
                # pb.update(min(float(tmp)/nfake, 1)*100)
                print("\r {}/{} complete...".format(tmp, nfake))

        fake_images = torch.cat(fake_images, dim=0)
        # remove extra entries
        fake_images = fake_images[0:nfake]

        if to_numpy:
            fake_images = fake_images.numpy()

        return fake_images, given_labels

    ## end def

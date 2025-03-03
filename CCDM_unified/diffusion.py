import numpy as np
import math
from functools import partial
from collections import namedtuple
from tqdm.auto import tqdm

import torch
from torch import nn, einsum
from torch.cuda.amp import autocast
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from utils import (
    default,
    identity,
    unnormalize_to_zero_to_one,
    normalize_to_neg_one_to_one,
    prob_mask_like,
)

# gaussian diffusion trainer class

ModelPrediction = namedtuple("ModelPrediction", ["pred_noise", "pred_x_start"])


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def generate_random_vectors(vector_type, dim, n_vectors, device):
    """
    Generate random vectors for projection
    """
    if vector_type == "gaussian":
        # Generate random Gaussian vectors
        return torch.randn(n_vectors, dim, device=device)
    elif vector_type == "rademacher":
        # Generate random Rademacher vectors (±1)
        return torch.randint(0, 2, (n_vectors, dim), device=device) * 2 - 1
    elif vector_type == "sphere":
        # Generate random vectors on the unit sphere
        vectors = torch.randn(n_vectors, dim, device=device)
        vectors = F.normalize(vectors, dim=1)
        return vectors
    else:
        raise ValueError(f"Unknown vector type: {vector_type}")


def compute_distance(y1, y2, distance_type="l2"):
    """
    Compute distance between two label vectors based on specified distance type
    """
    if distance_type == "l1":
        return torch.abs(y1 - y2).sum(dim=-1)
    elif distance_type == "l2":
        return torch.sqrt(((y1 - y2) ** 2).sum(dim=-1))
    elif distance_type == "cosine":
        return 1 - F.cosine_similarity(y1, y2, dim=-1)
    else:
        raise ValueError(f"Unknown distance type: {distance_type}")


def compute_projection(y, v):
    """
    Project label vector y onto random direction v
    """
    # Normalize the projection vector v
    v_norm = torch.sqrt(torch.sum(v**2, dim=-1, keepdim=True))
    v_normalized = v / (v_norm + 1e-8)

    # Compute the projection
    projection = torch.matmul(y, v_normalized.t())
    return projection


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        use_Hy=False,  # use y dependent covariance matrix
        fn_y2cov=None,
        cond_drop_prob=0.5,
        timesteps=1000,
        sampling_timesteps=None,  # if sampling_timesteps<timesteps, do ddim sampling
        objective="pred_noise",
        beta_schedule="cosine",
        ddim_sampling_eta=0,  # 1 for ddpm, 0 for ddim
        offset_noise_strength=0.0,
        min_snr_loss_weight=False,
        min_snr_gamma=5,
        use_cfg_plus_plus=False,  # https://arxiv.org/pdf/2406.08070
    ):
        super().__init__()
        assert not (
            type(self) == GaussianDiffusion
            and model.module.in_channels != model.module.out_dim
        )
        assert not model.module.random_or_learned_sinusoidal_cond

        self.model = model
        self.channels = self.model.module.in_channels

        # use y dependent covariance matrix
        self.use_Hy = use_Hy
        self.fn_y2cov = fn_y2cov
        if self.use_Hy:
            assert self.fn_y2cov is not None

        self.cond_drop_prob = cond_drop_prob  # originally not in diffusion but in unet

        self.image_size = image_size

        self.objective = objective

        assert objective in {
            "pred_noise",
            "pred_x0",
            "pred_v",
        }, "objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])"

        if beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"unknown beta schedule {beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)

        # use cfg++ when ddim sampling

        self.use_cfg_plus_plus = use_cfg_plus_plus

        # sampling related parameters

        self.sampling_timesteps = default(
            sampling_timesteps, timesteps
        )  # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(
            name, val.to(torch.float32)
        )

        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer("posterior_variance", posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        # offset noise strength - 0.1 was claimed ideal

        self.offset_noise_strength = offset_noise_strength

        # loss weight

        snr = alphas_cumprod / (1 - alphas_cumprod)

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max=min_snr_gamma)

        if objective == "pred_noise":
            loss_weight = maybe_clipped_snr / snr
        elif objective == "pred_x0":
            loss_weight = maybe_clipped_snr
        elif objective == "pred_v":
            loss_weight = maybe_clipped_snr / (snr + 1)

        register_buffer("loss_weight", loss_weight)

    @property
    def device(self):
        return self.betas.device

    # compute x_0 from x_t and pred noise: the reverse of `q_sample`; inverse of Eq.(9) in improved DDPM
    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise
            - extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    # Compute the mean and variance of the diffusion posterior: q(x_{t-1} | x_t, x_0)  # Eq.(6) of DDPM; Eq.(12) of improved DDPM
    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(
        self, x, t, labels, cond_scale=6.0, rescaled_phi=0.7, clip_x_start=False
    ):
        model_output, model_output_null = self.model.module.forward_with_cond_scale(
            x, t, labels, cond_scale=cond_scale, rescaled_phi=rescaled_phi
        )
        maybe_clip = (
            partial(torch.clamp, min=-1.0, max=1.0) if clip_x_start else identity
        )

        if self.objective == "pred_noise":
            pred_noise = (
                model_output if not self.use_cfg_plus_plus else model_output_null
            )

            x_start = self.predict_start_from_noise(x, t, model_output)
            x_start = maybe_clip(x_start)

        elif self.objective == "pred_x0":
            x_start = model_output
            x_start = maybe_clip(x_start)
            x_start_for_pred_noise = (
                x_start if not self.use_cfg_plus_plus else maybe_clip(model_output_null)
            )

            pred_noise = self.predict_noise_from_start(x, t, x_start_for_pred_noise)

        elif self.objective == "pred_v":
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)

            x_start_for_pred_noise = x_start
            if self.use_cfg_plus_plus:
                x_start_for_pred_noise = self.predict_start_from_v(
                    x, t, model_output_null
                )
                x_start_for_pred_noise = maybe_clip(x_start_for_pred_noise)

            pred_noise = self.predict_noise_from_start(x, t, x_start_for_pred_noise)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(
        self, x, t, labels_emb, cond_scale, rescaled_phi, clip_denoised=True
    ):
        preds = self.model_predictions(x, t, labels_emb, cond_scale, rescaled_phi)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_start, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(
        self,
        x,
        t: int,
        labels_emb,
        cond_scale=6.0,
        rescaled_phi=0.7,
        clip_denoised=True,
    ):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x=x,
            t=batched_times,
            labels_emb=labels_emb,
            cond_scale=cond_scale,
            rescaled_phi=rescaled_phi,
            clip_denoised=clip_denoised,
        )
        noise = torch.randn_like(x) if t > 0 else 0.0  # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(
        self, labels_emb, labels, shape, cond_scale=6.0, rescaled_phi=0.7
    ):
        batch, device = shape[0], self.betas.device

        # img = torch.randn(shape, device=device)
        if self.use_Hy:
            img = torch.randn(shape, device=device) * torch.sqrt(
                self.convert_y_to_cov(labels)
            )
        else:
            img = torch.randn(shape, device=device)

        # x_start = None

        for t in tqdm(
            reversed(range(0, self.sampling_timesteps)),
            desc="sampling loop time step",
            total=self.sampling_timesteps,
        ):
            img, _ = self.p_sample(img, t, labels_emb, cond_scale, rescaled_phi)

        img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def ddim_sample(
        self,
        labels_emb,
        labels,
        shape,
        cond_scale=6.0,
        rescaled_phi=0.7,
        clip_denoised=True,
    ):

        batch, device, total_timesteps, sampling_timesteps, eta, objective = (
            shape[0],
            self.betas.device,
            self.num_timesteps,
            self.sampling_timesteps,
            self.ddim_sampling_eta,
            self.objective,
        )

        times = torch.linspace(
            -1, total_timesteps - 1, steps=sampling_timesteps + 1
        )  # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(
            zip(times[:-1], times[1:])
        )  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        if self.use_Hy:
            img = torch.randn(shape, device=device) * torch.sqrt(
                self.convert_y_to_cov(labels)
            )
        else:
            img = torch.randn(shape, device=device)

        x_start = None

        for time, time_next in tqdm(time_pairs, desc="sampling loop time step"):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(
                img,
                time_cond,
                labels_emb,
                cond_scale=cond_scale,
                rescaled_phi=rescaled_phi,
                clip_x_start=clip_denoised,
            )

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = (
                eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            )
            c = (1 - alpha_next - sigma**2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

        img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def sample(self, labels_emb, labels, cond_scale=6.0, rescaled_phi=0.7):
        batch_size, image_size, channels = (
            labels_emb.shape[0],
            self.image_size,
            self.channels,
        )
        # sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        sample_fn = self.p_sample_loop
        return sample_fn(
            labels_emb,
            labels,
            (batch_size, channels, image_size, image_size),
            cond_scale,
            rescaled_phi,
        )

    # @autocast('cuda', enabled = False)
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        if self.offset_noise_strength > 0.0:
            offset_noise = torch.randn(x_start.shape[:2], device=self.device)
            noise += self.offset_noise_strength * rearrange(
                offset_noise, "b c -> b c 1 1"
            )

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    ## conduct y to convariance
    @torch.no_grad()
    def convert_y_to_cov(self, labels):
        b, c, h, w = len(labels), self.channels, self.image_size, self.image_size
        return torch.exp(-self.fn_y2cov(labels).view(b, c, h, w))

    def p_losses(
        self,
        x_start,
        t,
        *,
        labels,
        labels_emb,
        noise=None,
        vicinal_weights=None,
        **kwargs,
    ):
        """
        Modified p_losses to support various vicinal loss types including Sliced variants
        for multi-dimensional labels.

        Parameters:
        - x_start: Starting images [B, C, H, W]
        - t: Timesteps [B]
        - labels: Continuous labels [B, D] where D is the dimensionality of labels
        - labels_emb: Embedded labels from label_embedding network
        - noise: Optional pre-defined noise
        - vicinal_weights: Optional pre-defined vicinal weights
        - kwargs: Additional parameters including vicinity_type, kappa, vector_type, etc.

        Returns:
        - loss: Computed loss value
        """
        vicinity_type = kwargs.get("vicinity_type", "shv")  # shv, ssv, hv, sv
        kappa = kwargs.get("kappa", 0.01)
        vector_type = kwargs.get("vector_type", "gaussian")
        num_projections = kwargs.get("num_projections", 1)
        distance = kwargs.get("distance", "l2")  # l2, l1, cosine

        b, c, h, w = x_start.shape

        # Generate keep_mask for random label dropping
        keep_mask = prob_mask_like((b,), 1 - self.cond_drop_prob, device=x_start.device)
        null_indx = torch.where(keep_mask == False)[0]

        # Use y dependent covariance matrix or not
        if self.use_Hy:
            # Generate noise based on y-dependent covariance
            if noise is None:
                Hy_sqrt = torch.sqrt(self.convert_y_to_cov(labels))
                noise = torch.randn_like(x_start) * Hy_sqrt
                # For null indices, use standard normal
                if len(null_indx) > 0:
                    noise[null_indx] = torch.randn_like(x_start[null_indx])
        else:
            noise = default(noise, lambda: torch.randn_like(x_start))

        # Noise sample
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        # Predict and take gradient step
        model_out = self.model(
            x=x, timesteps=t, labels_emb=labels_emb, keep_mask=keep_mask
        )

        # Determine target based on objective
        if self.objective == "pred_noise":
            target = noise
        elif self.objective == "pred_x0":
            target = x_start
        elif self.objective == "pred_v":
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f"unknown objective {self.objective}")

        # Calculate base MSE loss
        loss = F.mse_loss(model_out, target, reduction="none")

        # Apply y-dependent weighting if using Hy
        if self.use_Hy:
            loss_divisor = self.convert_y_to_cov(labels)
            if len(null_indx) > 0:
                loss_divisor[null_indx] = torch.ones_like(loss_divisor[null_indx])
            loss = loss / loss_divisor

        # Reduce across spatial dimensions
        loss = reduce(loss, "b ... -> b (...)", "mean")

        # Apply timestep weighting
        loss = loss * extract(self.loss_weight, t, loss.shape)

        # Apply vicinal weights based on vicinity type
        if vicinal_weights is not None:
            # Reshape loss to [B]
            loss = torch.sum(loss, dim=1)

            # For multi-dimensional labels
            if (
                vicinity_type in ["shv", "ssv"]
                and labels.dim() > 1
                and labels.shape[1] > 1
            ):
                device = labels.device
                dim = labels.shape[1]

                # Initialize batch weights
                batch_weights = torch.zeros_like(vicinal_weights)

                # Generate or retrieve random projection vectors
                if "cached_vectors" in kwargs and kwargs["cached_vectors"] is not None:
                    v = kwargs["cached_vectors"]
                else:
                    v = generate_random_vectors(
                        vector_type, dim, num_projections, device
                    )

                # Apply weights based on all projection vectors
                for proj_idx in range(num_projections):
                    # Project labels onto current random direction
                    proj_vec = v[proj_idx : proj_idx + 1]

                    # Normalize projection vector for numerical stability
                    proj_vec_norm = F.normalize(proj_vec, dim=1, eps=1e-8)

                    # Calculate projections for all labels at once
                    all_projections = torch.matmul(labels, proj_vec_norm.t()).squeeze(
                        -1
                    )  # [B]

                    # Calculate pairwise projection differences efficiently
                    proj_diff_matrix = all_projections.unsqueeze(
                        1
                    ) - all_projections.unsqueeze(
                        0
                    )  # [B, B]

                    if vicinity_type == "shv":  # Sliced Hard Vicinal
                        # Calculate effective kappa based on projection vector norm
                        effective_kappa = kappa * torch.norm(proj_vec) + 1e-8

                        # Create mask for each sample
                        in_vicinity_mask = (
                            torch.abs(proj_diff_matrix) <= effective_kappa
                        ).float()  # [B, B]

                        # Update batch weights (sum over all labels that are in vicinity)
                        batch_weights += in_vicinity_mask.sum(dim=1) / num_projections

                    else:  # Sliced Soft Vicinal
                        # Calculate nu parameter
                        nu = 1.0 / (kappa**2)

                        # Calculate soft weights
                        soft_weights = torch.exp(-nu * proj_diff_matrix**2)  # [B, B]

                        # Update batch weights (sum over weighted contributions)
                        batch_weights += soft_weights.sum(dim=1) / num_projections

                # Normalize batch weights
                batch_weights = batch_weights / b  # Normalize by batch size

                # Don't apply weighting on null inputs
                if len(null_indx) > 0:
                    batch_weights[null_indx] = 1.0

                # Apply weights to loss
                loss = torch.sum(batch_weights * loss) / (b * c * h * w)

            elif vicinity_type in ["hv", "sv"]:  # Traditional Hard/Soft Vicinal
                # Initialize weights
                batch_weights = torch.zeros_like(vicinal_weights)

                # Calculate pairwise distances efficiently
                if distance == "l2":
                    # L2 distance
                    if labels.dim() > 1 and labels.shape[1] > 1:
                        # Multi-dimensional case
                        diff = labels.unsqueeze(1) - labels.unsqueeze(0)  # [B, B, D]
                        pairwise_dist = torch.sqrt((diff**2).sum(dim=2))  # [B, B]
                    else:
                        # 1D case
                        diff = labels.unsqueeze(1) - labels.unsqueeze(0)  # [B, B]
                        pairwise_dist = torch.abs(diff)  # [B, B]

                elif distance == "l1":
                    # L1 distance
                    diff = labels.unsqueeze(1) - labels.unsqueeze(
                        0
                    )  # [B, B, D] or [B, B]
                    if diff.dim() > 2:
                        pairwise_dist = torch.abs(diff).sum(dim=2)  # [B, B]
                    else:
                        pairwise_dist = torch.abs(diff)  # [B, B]

                elif distance == "cosine":
                    # Cosine distance
                    if labels.dim() > 1 and labels.shape[1] > 1:
                        # Normalize labels
                        norm_labels = F.normalize(labels, dim=1)
                        # Calculate cosine similarity
                        cos_sim = torch.matmul(norm_labels, norm_labels.t())  # [B, B]
                        # Convert to distance (1 - similarity)
                        pairwise_dist = 1 - cos_sim  # [B, B]
                    else:
                        # For 1D labels, cosine distance doesn't make sense
                        # Fallback to L2
                        diff = labels.unsqueeze(1) - labels.unsqueeze(0)
                        pairwise_dist = torch.abs(diff)

                # Apply vicinity weights
                if vicinity_type == "hv":  # Hard Vicinal
                    # Create binary mask for vicinity
                    vicinity_mask = (pairwise_dist <= kappa).float()  # [B, B]
                    # Sum over all labels in vicinity for each sample
                    batch_weights = vicinity_mask.sum(dim=1)

                else:  # Soft Vicinal
                    # Calculate soft weights
                    nu = 1.0 / (kappa**2)
                    batch_weights = torch.exp(-nu * pairwise_dist**2).sum(dim=1)

                # Normalize
                batch_weights = batch_weights / b

                # Don't apply weighting on null inputs
                if len(null_indx) > 0:
                    batch_weights[null_indx] = 1.0

                # Apply weights to loss
                loss = torch.sum(batch_weights * loss) / (b * c * h * w)

            else:  # Use provided vicinal weights (original implementation)
                if len(null_indx) > 0:
                    vicinal_weights[null_indx] = (
                        1.0  # Don't apply weighting on null inputs
                    )
                loss = torch.sum(vicinal_weights * loss) / (b * c * h * w)
        else:
            loss = loss.mean()

        return loss

    def forward(self, img, *args, **kwargs):
        (
            b,
            c,
            h,
            w,
            device,
            img_size,
        ) = (
            *img.shape,
            img.device,
            self.image_size,
        )
        assert (
            h == img_size and w == img_size
        ), f"height and width of image must be {img_size}"
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = normalize_to_neg_one_to_one(img)

        return self.p_losses(img, t, *args, **kwargs)

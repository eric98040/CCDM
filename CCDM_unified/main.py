print(
    "\n==================================================================================================="
)

import os
import math
from abc import abstractmethod
import random
import sys

from PIL import Image
import requests
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm
import matplotlib.pyplot as plt
import h5py
import gc
import copy
import timeit

from models import Unet
from models.vit import ViT
from dataset import PowerSeqDataset  # , PowerTransformer

# from dataset import LoadDataSet
from label_embedding import LabelEmbed
from diffusion import GaussianDiffusion
from trainer import Trainer
from opts import parse_opts
from utils import (
    get_parameter_number,
    SimpleProgressBar,
    IMGs_dataset,
    compute_entropy,
    predict_class_labels,
)

##############################################
""" Settings """
args = parse_opts()

# seeds
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
cudnn.benchmark = False
np.random.seed(args.seed)

if args.torch_model_path != "None":
    os.environ["TORCH_HOME"] = args.torch_model_path

# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True


#######################################################################################
"""                                Output folders                                  """
#######################################################################################
path_to_output = os.path.join(
    args.root_path, "output/{}_{}".format(args.data_name, args.image_size)
)
os.makedirs(path_to_output, exist_ok=True)

save_setting_folder = os.path.join(path_to_output, "{}".format(args.setting_name))
os.makedirs(save_setting_folder, exist_ok=True)

setting_log_file = os.path.join(save_setting_folder, "setting_info.txt")
if not os.path.isfile(setting_log_file):
    logging_file = open(setting_log_file, "w")
    logging_file.close()
with open(setting_log_file, "a") as logging_file:
    logging_file.write(
        "\n==================================================================================================="
    )
    print(args, file=logging_file)

save_results_folder = os.path.join(save_setting_folder, "results")
os.makedirs(save_results_folder, exist_ok=True)


#######################################################################################
"""                                Make dataset                                     """
#######################################################################################

if args.dataset == "power_vector":
    # initialize Power transformer and train
    # power_transformer = PowerTransformer()

    # load Power dataset
    dataset = PowerSeqDataset(
        design_folder=args.design_folder,
        power_path=args.power_data_path,
        power_transformer=None,
        normalize_design=True,
    )

    # prepare training data
    train_data = []
    for i in range(len(dataset)):
        batch = dataset[i]
        train_data.append((batch["design"], batch["labels"]))

    # separate images and labels
    train_images = torch.stack([item[0] for item in train_data]).numpy()
    train_labels = torch.stack([item[1] for item in train_data]).numpy()
    train_labels_norm = train_labels.copy()  # normalized labels

    # unique labels
    unique_labels_norm = np.unique(train_labels_norm, axis=0)

    # multi-dim label
    args.label_dim = train_labels.shape[1]

else:
    # Keep this as fallback for any legacy dataset handlers
    # (though we should eventually phase this out)
    dataset = LoadDataSet(
        data_name=args.data_name,
        data_path=args.data_path,
        min_label=args.min_label,
        max_label=args.max_label,
        img_size=args.image_size,
        max_num_img_per_label=args.max_num_img_per_label,
        num_img_per_label_after_replica=args.num_img_per_label_after_replica,
    )

    train_images, train_labels, train_labels_norm = dataset.load_train_data()
    unique_labels_norm = np.sort(np.array(list(set(train_labels_norm))))

    # single-dim label
    args.label_dim = 1

# Modify the hyperparameter calculation section to handle multi-dimensional labels
if args.kernel_sigma < 0:
    if args.label_dim > 1:
        # For multi-dimensional labels, compute std per dimension and average
        std_label = np.mean(np.std(train_labels_norm, axis=0))
    else:
        std_label = np.std(train_labels_norm)

    args.kernel_sigma = 1.06 * std_label * (len(train_labels_norm)) ** (-1 / 5)

    print("\n Use rule-of-thumb formula to compute kernel_sigma >>>")
    print(
        "\r The std of {} labels is {:.4f} so the kernel sigma is {:.4f}".format(
            len(train_labels_norm), std_label, args.kernel_sigma
        )
    )
# 변수 설정
is_hard_vicinity = args.vicinity_type in ["hv", "shv"]

# kappa 계산 시 hard/soft 여부 결정
if args.kappa < 0:
    if args.hyperparameter == "rule_of_thumb":
        # Rule of thumb method
        if args.label_dim > 1:
            # For multi-dimensional labels
            n_unique = len(unique_labels_norm)

            if n_unique > 1:
                diff_list = []
                for i in range(1, n_unique):
                    # Compute distance between consecutive labels
                    if args.distance == "l1":
                        diff = np.sum(
                            np.abs(unique_labels_norm[i] - unique_labels_norm[i - 1])
                        )
                    elif args.distance == "cosine":
                        norm1 = np.linalg.norm(unique_labels_norm[i])
                        norm2 = np.linalg.norm(unique_labels_norm[i - 1])
                        dot_product = np.dot(
                            unique_labels_norm[i], unique_labels_norm[i - 1]
                        )
                        diff = (
                            1 - dot_product / (norm1 * norm2)
                            if norm1 > 0 and norm2 > 0
                            else 1
                        )
                    else:  # default to l2
                        diff = np.linalg.norm(
                            unique_labels_norm[i] - unique_labels_norm[i - 1]
                        )
                    diff_list.append(diff)

                kappa_base = np.max(np.array(diff_list))
            else:
                # Fallback for single unique label
                kappa_base = 0.01
        else:
            # Original logic for scalar labels
            n_unique = len(unique_labels_norm)
            diff_list = []
            for i in range(1, n_unique):
                diff_list.append(unique_labels_norm[i] - unique_labels_norm[i - 1])
            kappa_base = np.max(np.array(diff_list))

        if is_hard_vicinity:
            args.kappa = np.abs(args.kappa) * kappa_base
        else:
            args.kappa = 1 / (np.abs(args.kappa) * kappa_base) ** 2

    elif args.hyperparameter == "percentile":
        # Percentile method
        # Calculate pairwise distances between all labels
        distances = []
        for i in range(len(train_labels_norm)):
            for j in range(i + 1, len(train_labels_norm)):
                if args.distance == "l1":
                    dist = np.sum(np.abs(train_labels_norm[i] - train_labels_norm[j]))
                elif args.distance == "cosine":
                    norm1 = np.linalg.norm(train_labels_norm[i])
                    norm2 = np.linalg.norm(train_labels_norm[j])
                    dot_product = np.dot(train_labels_norm[i], train_labels_norm[j])
                    dist = (
                        1 - dot_product / (norm1 * norm2)
                        if norm1 > 0 and norm2 > 0
                        else 1
                    )
                else:  # default to l2
                    dist = np.linalg.norm(train_labels_norm[i] - train_labels_norm[j])
                distances.append(dist)

        # Set kappa to the percentile of distances
        args.kappa = np.percentile(distances, args.percentile)

        # Adjust based on vicinity type
        if not is_hard_vicinity:  # For soft vicinity
            args.kappa = 1 / (args.kappa) ** 2

# Update the vicinal parameters dict with all needed parameters
vicinal_params = {
    "kernel_sigma": args.kernel_sigma,
    "kappa": args.kappa,
    "nonzero_soft_weight_threshold": args.nonzero_soft_weight_threshold,
    "vicinity_type": args.vicinity_type,
    "vector_type": args.vector_type,
    "num_projections": args.num_projections,
    "distance": args.distance,
    "label_dim": args.label_dim,
    "adaptive_slicing": args.adaptive_slicing,
    "hyperparameter": args.hyperparameter,
    "percentile": args.percentile,
}


#######################################################################################
"""                             label embedding method                              """
#######################################################################################

# Initialize label embedding based on the selected type
if args.label_embed == "ccdm1":
    label_embedding = LabelEmbed(
        dataset=dataset,
        path_y2h=path_to_output + "/model_y2h",
        y2h_type=args.y2h_embed_type,
        h_dim=args.dim_embed,
        nc=args.num_channels,
        label_dim=args.label_dim,
        dim_combination=args.dim_combination,
    )
elif args.label_embed == "ccdm2":
    label_embedding = LabelEmbed(
        dataset=dataset,
        path_y2h=path_to_output + "/model_y2h",
        path_y2cov=path_to_output + "/model_y2cov",
        y2h_type=args.y2h_embed_type,
        y2cov_type=args.y2cov_embed_type,
        h_dim=args.dim_embed,
        cov_dim=args.image_size**2 * args.num_channels,
        nc=args.num_channels,
        label_dim=args.label_dim,
        dim_combination=args.dim_combination,
    )
else:  # random
    label_embedding = LabelEmbed(
        dataset=dataset,
        path_y2h=path_to_output + "/model_y2h",
        y2h_type="gaussian",
        h_dim=args.dim_embed,
        nc=args.num_channels,
        label_dim=args.label_dim,
        dim_combination=args.dim_combination,
    )

fn_y2h = label_embedding.fn_y2h
fn_y2cov = getattr(
    label_embedding, "fn_y2cov", None
)  # Use getattr to avoid attribute error


#######################################################################################
"""                             Diffusion  training                                 """
#######################################################################################

if args.architecture == "unet":
    channel_mult = (args.channel_mult).split("_")
    channel_mult = [int(dim) for dim in channel_mult]

    model = Unet(
        dim=args.model_channels,
        embed_input_dim=args.dim_embed,
        cond_drop_prob=args.cond_drop_prob,
        dim_mults=channel_mult,
        in_channels=args.num_channels,
        learned_variance=False,
        learned_sinusoidal_cond=False,
        random_fourier_features=False,
        learned_sinusoidal_dim=16,
        attn_dim_head=args.attn_dim_head,
        attn_heads=args.num_heads,
    )
else:  # vit
    model = ViT(
        dim=args.model_channels,
        embed_input_dim=args.dim_embed,
        cond_drop_prob=args.cond_drop_prob,
        in_channels=args.num_channels,
        learned_variance=False,
        learned_sinusoidal_cond=False,
        random_fourier_features=False,
        learned_sinusoidal_dim=16,
        attn_dim_head=args.attn_dim_head,
        attn_heads=args.num_heads,
        patch_size=2,  # 2x2 patches
        num_blocks=8,  # 8 transformer blocks
    )

model = nn.DataParallel(model)
print("\r model size:", get_parameter_number(model))


## build diffusion process
diffusion = GaussianDiffusion(
    model,
    use_Hy=args.use_Hy,
    fn_y2cov=fn_y2cov,
    cond_drop_prob=args.cond_drop_prob,
    image_size=args.image_size,
    timesteps=args.train_timesteps,
    sampling_timesteps=args.sample_timesteps,
    objective=args.pred_objective,
    beta_schedule=args.beta_schedule,
    ddim_sampling_eta=args.ddim_eta,
    vicinity_type=args.vicinity_type,
).cuda()


## for visualization
if args.image_size > 128:
    n_row = 6
else:
    n_row = 10
n_col = n_row
start_label = np.quantile(train_labels_norm, 0.05)
end_label = np.quantile(train_labels_norm, 0.95)
selected_labels = np.linspace(start_label, end_label, num=n_row)
y_visual = np.zeros(n_row * n_col)
for i in range(n_row):
    curr_label = selected_labels[i]
    for j in range(n_col):
        y_visual[i * n_col + j] = curr_label
y_visual = torch.from_numpy(y_visual).type(torch.float).view(-1).cuda()
print(y_visual)


trainer = Trainer(
    data_name=args.data_name,
    diffusion_model=diffusion,
    train_images=train_images,
    train_labels=train_labels_norm,
    vicinal_params=vicinal_params,
    train_batch_size=args.train_batch_size,
    gradient_accumulate_every=args.gradient_accumulate_every,
    train_lr=args.train_lr,
    train_num_steps=args.niters,
    ema_update_after_step=100,
    ema_update_every=10,
    ema_decay=0.995,
    adam_betas=(0.9, 0.99),
    sample_every=args.sample_every,
    save_every=args.save_every,
    results_folder=save_results_folder,
    amp=args.train_amp,
    mixed_precision_type="fp16",
    split_batches=True,
    max_grad_norm=1.0,
    y_visual=y_visual,
    nrow_visual=n_col,
    cond_scale_visual=args.sample_cond_scale,
    vicinity_type=args.vicinity_type,
    kappa=args.kappa,
    sigma_delta=args.kernel_sigma,
    vector_type=args.vector_type,
    num_projections=args.num_projections,
    distance=args.distance,
    label_dim=args.label_dim,
    adaptive_slicing=args.adaptive_slicing,
    hyperparameter=args.hyperparameter,
    percentile=args.percentile,
)

if args.resume_niter > 0:
    trainer.load(args.resume_niter)
trainer.train(fn_y2h=fn_y2h)


#######################################################################################
"""                                Sampling                                        """
#######################################################################################

print(
    "\n Start sampling {} fake images per label from the model >>>".format(
        args.nfake_per_label
    )
)

## get evaluation labels
_, _, eval_labels = dataset.load_evaluation_data()

num_eval_labels = len(eval_labels)
print(eval_labels)


###########################################
""" multiple h5 files """

dump_fake_images_folder = os.path.join(
    save_results_folder,
    "fake_data_niters{}_nfake{}_{}_sampstep{}".format(
        args.niters,
        int(args.nfake_per_label * num_eval_labels),
        args.sampler,
        args.sample_timesteps,
    ),
)
os.makedirs(dump_fake_images_folder, exist_ok=True)

fake_images = []
fake_labels = []
total_sample_time = 0
for i in range(num_eval_labels):
    print(
        "\n [{}/{}]: Generating fake data for label {}...".format(
            i + 1, num_eval_labels, eval_labels[i]
        )
    )
    curr_label = eval_labels[i]
    dump_fake_images_filename = os.path.join(
        dump_fake_images_folder, "{}.h5".format(curr_label)
    )
    if not os.path.isfile(dump_fake_images_filename):
        fake_labels_i = curr_label * np.ones(args.nfake_per_label)
        start = timeit.default_timer()
        fake_images_i, _ = trainer.sample_given_labels(
            given_labels=dataset.fn_normalize_labels(fake_labels_i),
            fn_y2h=fn_y2h,
            batch_size=args.samp_batch_size,
            denorm=True,
            to_numpy=True,
            verbose=False,
            sampler=args.sampler,
            cond_scale=args.sample_cond_scale,
            sample_timesteps=args.sample_timesteps,
            ddim_eta=args.ddim_eta,
        )
        stop = timeit.default_timer()
        assert len(fake_images_i) == len(fake_labels_i)
        sample_time_i = stop - start
        if args.dump_fake_data:
            with h5py.File(dump_fake_images_filename, "w") as f:
                f.create_dataset(
                    "fake_images_i",
                    data=fake_images_i,
                    dtype="uint8",
                    compression="gzip",
                    compression_opts=6,
                )
                f.create_dataset("fake_labels_i", data=fake_labels_i, dtype="float")
                f.create_dataset(
                    "sample_time_i", data=np.array([sample_time_i]), dtype="float"
                )
    else:
        with h5py.File(dump_fake_images_filename, "r") as f:
            fake_images_i = f["fake_images_i"][:]
            fake_labels_i = f["fake_labels_i"][:]
            sample_time_i = f["sample_time_i"][0]
        assert len(fake_images_i) == len(fake_labels_i)
    ##end if
    total_sample_time += sample_time_i
    fake_images.append(fake_images_i)
    fake_labels.append(fake_labels_i)
    print(
        "\r {}/{}: Got {} fake images for label {}. Time spent {:.2f}, Total time {:.2f}.".format(
            i + 1,
            num_eval_labels,
            len(fake_images_i),
            curr_label,
            sample_time_i,
            total_sample_time,
        )
    )

    ## dump some imgs for visualization
    img_vis_i = fake_images_i[0:36] / 255.0
    img_vis_i = torch.from_numpy(img_vis_i)
    img_filename = os.path.join(
        dump_fake_images_folder, "sample_{}.png".format(curr_label)
    )
    torchvision.utils.save_image(img_vis_i.data, img_filename, nrow=6, normalize=False)
    del fake_images_i, fake_labels_i
    gc.collect()

##end for i

fake_images = np.concatenate(fake_images, axis=0)
fake_labels = np.concatenate(fake_labels)
print("Sampling finished; Time elapses: {}s".format(total_sample_time))


print(
    "\n==================================================================================================="
)

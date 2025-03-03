import argparse


def parse_opts():
    parser = argparse.ArgumentParser()

    """ Overall Settings """
    parser.add_argument("--root_path", type=str, default="./")
    parser.add_argument(
        "--data_name",
        type=str,
        default="UTKFace",
        choices=["UTKFace", "RC-49", "Cell200", "SteeringAngle"],
    )
    parser.add_argument("--data_path", type=str, default="./")
    parser.add_argument("--torch_model_path", type=str, default="None")
    parser.add_argument("--eval_ckpt_path", type=str, default="./")
    parser.add_argument(
        "--seed", type=int, default=111, metavar="S", help="random seed"
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--setting_name", type=str, default="Setup1")

    """Additional Parameters for Sliced CCDM"""
    parser.add_argument(
        "--architecture",
        type=str,
        default="unet",
        choices=["unet", "vit"],
        help="Network architecture for the denoising model",
    )
    parser.add_argument(
        "--label_embed",
        type=str,
        default="ccdm2",
        choices=["ccdm1", "ccdm2", "random"],
        help="Label embedding network type",
    )
    parser.add_argument(
        "--vicinity_type",
        type=str,
        default="shv",
        choices=["shv", "ssv", "hv", "sv"],
        help="Type of vicinity loss: Sliced Hard Vicinal (shv), Sliced Soft Vicinal (ssv), Hard Vicinal (hv), Soft Vicinal (sv)",
    )
    parser.add_argument(
        "--distance",
        type=str,
        default="l2",
        choices=["l1", "l2", "cosine"],
        help="Distance metric for vicinity computation when using HV-NLL or SV-NLL",
    )
    parser.add_argument(
        "--vector_type",
        type=str,
        default="gaussian",
        choices=["gaussian", "rademacher", "sphere"],
        help="Type of random vector for projection",
    )
    parser.add_argument(
        "--hyperparameter",
        type=str,
        default="rule_of_thumb",
        choices=["rule_of_thumb", "percentile"],
        help="Method to set kappa and sigma_delta",
    )
    parser.add_argument(
        "--adaptive_slicing",
        action="store_true",
        default=False,
        help="Compute kappa and sigma_delta dynamically for each batch",
    )
    parser.add_argument(
        "--label_dim", type=int, default=1, help="Dimension of regression labels"
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=5.0,
        help="Percentile for kappa when using percentile method",
    )
    parser.add_argument(
        "--num_projections",
        type=int,
        default=1,
        help="Number of random projections (M) for Sliced Vicinal Loss",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="default",
        choices=["default", "power_vector"],
        help="Dataset type to use",
    )
    parser.add_argument(
        "--power_data_path",
        type=str,
        default="./data/0130_40_generated_power_array1_drop3.csv",
        help="Path to power vector CSV file",
    )
    parser.add_argument(
        "--design_folder",
        type=str,
        default="./data/image/",
        help="Path to design image folder",
    )

    """ Dataset """
    parser.add_argument("--min_label", type=float, default=0.0)
    parser.add_argument("--max_label", type=float, default=90.0)
    parser.add_argument("--num_channels", type=int, default=3, metavar="N")
    parser.add_argument("--image_size", type=int, default=64, metavar="N")
    parser.add_argument("--max_num_img_per_label", type=int, default=1e30, metavar="N")
    parser.add_argument(
        "--num_img_per_label_after_replica", type=int, default=0, metavar="N"
    )

    """ Model Config """
    parser.add_argument("--model_channels", type=int, default=64, metavar="N")
    parser.add_argument("--num_res_blocks", type=int, default=2, metavar="N")
    parser.add_argument("--num_heads", type=int, default=4, metavar="N")
    parser.add_argument("--num_groups", type=int, default=8, metavar="N")
    parser.add_argument("--attention_resolutions", type=str, default="16_32")
    parser.add_argument("--channel_mult", type=str, default="1_2_4_8")
    parser.add_argument("--attn_dim_head", type=int, default=32, metavar="N")
    parser.add_argument("--cond_drop_prob", type=float, default=0.1)

    """ Training """
    ## Diffusion model
    parser.add_argument("--pred_objective", type=str, default="pred_noise")
    parser.add_argument("--niters", type=int, default=10, metavar="N")
    parser.add_argument("--resume_niter", type=int, default=0, metavar="N")
    parser.add_argument("--train_timesteps", type=int, default=1000, metavar="N")
    parser.add_argument("--train_batch_size", type=int, default=16, metavar="N")
    parser.add_argument("--train_lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--train_amp", action="store_true", default=False)
    parser.add_argument("--gradient_accumulate_every", type=int, default=1, metavar="N")
    parser.add_argument("--beta_schedule", type=str, default="cosine")
    parser.add_argument("--sample_every", type=int, default=1000, metavar="N")
    parser.add_argument("--save_every", type=int, default=10000, metavar="N")

    ## label embedding setting:
    # type of label embedding
    parser.add_argument(
        "--y2h_embed_type",
        type=str,
        default="sinusoidal",
        choices=["resnet", "sinusoidal", "gaussian"],
    )  # for y to h
    parser.add_argument(
        "--y2cov_embed_type",
        type=str,
        default="sinusoidal",
        choices=["resnet", "sinusoidal", "gaussian"],
    )  # for y to cov
    parser.add_argument("--use_Hy", action="store_true", default=False)
    # if resnet: y2h
    parser.add_argument(
        "--net_embed", type=str, default="ResNet34_embed"
    )  # ResNetXX_emebed
    parser.add_argument(
        "--epoch_cnn_embed", type=int, default=200
    )  # epoch of cnn training for label embedding
    parser.add_argument(
        "--resumeepoch_cnn_embed", type=int, default=0
    )  # epoch of cnn training for label embedding
    parser.add_argument("--epoch_net_y2h", type=int, default=500)
    parser.add_argument(
        "--dim_embed", type=int, default=128
    )  # dimension of the embedding space
    parser.add_argument("--batch_size_embed", type=int, default=256, metavar="N")
    # if resnet: y2cov
    parser.add_argument("--net_embed_y2cov", type=str, default="ResNet34_embed_y2cov")
    parser.add_argument("--epoch_cnn_embed_y2cov", type=int, default=10)
    parser.add_argument("--resumeepoch_cnn_embed_y2cov", type=int, default=0)
    parser.add_argument("--epoch_net_y2cov", type=int, default=500)
    parser.add_argument("--batch_size_embed_y2cov", type=int, default=256, metavar="N")

    ## vicinal loss
    parser.add_argument(
        "--kernel_sigma",
        type=float,
        default=-1.0,
        help="If kernel_sigma<0, then use rule-of-thumb formula to compute the sigma.",
    )
    parser.add_argument(
        "--threshold_type", type=str, default="hard", choices=["soft", "hard"]
    )
    parser.add_argument("--kappa", type=float, default=-1)
    parser.add_argument(
        "--nonzero_soft_weight_threshold",
        type=float,
        default=1e-3,
        help="threshold for determining nonzero weights for SVDL; we neglect images with too small weights",
    )

    """ Sampling """
    parser.add_argument("--sampler", type=str, default="ddim")
    parser.add_argument("--sample_timesteps", type=int, default=250, metavar="N")
    parser.add_argument("--sample_cond_scale", type=float, default=1.5)
    parser.add_argument("--ddim_eta", type=float, default=0)

    parser.add_argument("--nfake_per_label", type=int, default=200)
    parser.add_argument("--samp_batch_size", type=int, default=100)
    parser.add_argument("--dump_fake_data", action="store_true", default=False)
    parser.add_argument("--dump_fake_for_NIQE", action="store_true", default=False)
    parser.add_argument("--niqe_dump_path", type=str, default="None")

    args = parser.parse_args()

    return args

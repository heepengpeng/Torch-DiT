# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
import os.path
import random
import string
import zipfile

import torch
import torch.distributed as dist
from diffusers.models import AutoencoderKL
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import save_image

from diffusion import create_diffusion
from download import find_model
from models import DiT_XL_2


# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
def zipDir(dirpath, outFullName):
    """
    压缩指定文件夹
    :param dirpath: 目标文件夹路径
    :param outFullName: 压缩文件保存路径+xxxx.zip
    :return: 无
    """
    zip = zipfile.ZipFile(outFullName, "w", zipfile.ZIP_DEFLATED)
    for path, dirnames, filenames in os.walk(dirpath):
        # 去掉目标跟路径，只对目标文件夹下边的文件及文件夹进行压缩
        fpath = path.replace(dirpath, '')

        for filename in filenames:
            zip.write(os.path.join(path, filename), os.path.join(fpath, filename))


# Setup PyTorch:
torch.manual_seed(0)
torch.set_grad_enabled(False)
num_sampling_steps = 250
cfg_scale = 4.0

# Multi GPU
dist.init_process_group("nccl")
rank = dist.get_rank()
device = rank % torch.cuda.device_count()

# Load model:
image_size = 256
assert image_size in [256, 512], "We only provide pre-trained models for 256x256 and 512x512 resolutions."
latent_size = image_size // 8
model = DiT_XL_2(input_size=latent_size)
model = DDP(model.to(device), device_ids=[rank])
state_dict = find_model(f"DiT-XL-2-{image_size}x{image_size}.pt")
model.load_state_dict(state_dict, False)

model.eval()

vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)

# Labels to condition the model with:
class_count = 1000
l = list(range(class_count))
n = 10
class_labels_batch = [l[i:i + n] for i in range(0, len(l), n)]

repeat_num = 1  # 50 * 1000 = 50K

for i in range(repeat_num):
    print("batch %d start" % i)
    for class_labels in class_labels_batch:
        diffusion = create_diffusion(str(num_sampling_steps))
        # Create sampling noise:
        n = len(class_labels)
        z = torch.randn(n, 4, latent_size, latent_size)
        y = torch.tensor(class_labels, device=device)

        # Setup classifier-free guidance:
        z = torch.cat([z, z], 0)
        y_null = torch.tensor([1000] * n, device=device)
        y = torch.cat([y, y_null], 0)
        model_kwargs = dict(y=y, cfg_scale=cfg_scale)

        # Sample images:
        # # Save and display images:
        samples = diffusion.p_sample_loop(
            model.forward, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True,
            device=device
        )
        samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        samples = vae.decode(samples / 0.18215).sample
        samples = samples.split(1)
        for idx, sample in enumerate(samples):
            save_dir = "sample/%d" % i
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_image(sample, save_dir + "/sample%s.png" % (''.join(random.sample(string.ascii_letters, 8))), nrow=1,
                       normalize=True,
                       value_range=(-1, 1))
    print("batch %d end" % i)

    input_path = "./sample"
    output_path = "./sample.zip"

    zipDir(input_path, output_path)

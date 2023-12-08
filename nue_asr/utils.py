# Copyright 2023 rinna Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import random
from typing import Optional, Union

import numpy as np
import torch
from transformers import AutoTokenizer

from .model import NueASRModel

DEFAULT_MODEL_NAME = "rinna/nue-asr"

logger = logging.getLogger(__name__)


def str2bool(v: str):
    if v.lower() in ("true", "t", "yes", "y", "1"):
        return True
    if v.lower() in ("false", "f", "no", "n", "0"):
        return False
    raise ValueError(f"Invalid boolean value: {v}")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_tokenizer(model_name_or_path: Optional[str] = None):
    if model_name_or_path is None:
        model_name_or_path = DEFAULT_MODEL_NAME

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, use_fast=False, legacy=True
    )
    return tokenizer


def load_model(
    model_name_or_path: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = "cuda",
    fp16: bool = True,
    use_deepspeed: bool = False,
) -> NueASRModel:
    if model_name_or_path is None:
        model_name_or_path = DEFAULT_MODEL_NAME

    device = torch.device(device)
    if device.type == "cpu":
        if torch.cuda.is_available():
            logging.warning(
                "CUDA is available but using CPU. "
                "If you want to use CUDA, set `device` to `cuda`."
            )
        if fp16:
            logging.warning("FP16 is not supported on CPU. Using FP32 instead.")
            fp16 = False
        if use_deepspeed:
            logging.warning("DeepSpeed is not supported on CPU. Disabling it.")
            use_deepspeed = False

    dtype = torch.float16 if fp16 else torch.float32

    model = NueASRModel.from_pretrained(model_name_or_path)
    model.to(dtype)

    if use_deepspeed:
        try:
            import deepspeed
        except ImportError:
            raise ImportError(
                "DeepSpeed is not installed. Please install it with `pip install deepspeed`."
            )

        ds_engine = deepspeed.init_inference(
            model.llm,
            replace_with_kernel_inject=True,
            dtype=dtype,
        )
        for m in ds_engine.modules():
            if (
                getattr(m, "config", None)
                and getattr(m.config, "mlp_after_attn", None) is not None
            ):
                m.config.mlp_after_attn = not model.llm.config.use_parallel_residual
        model.llm = ds_engine.module

    if device is not None:
        model.to(device)

    logger.info(f"Finished loading model from {model_name_or_path}")

    return model

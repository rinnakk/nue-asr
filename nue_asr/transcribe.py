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
from dataclasses import dataclass
from typing import Union

import numpy as np
import torch
from transformers import PreTrainedTokenizer

from .model import NueASRModel

WARN_TOO_LONG_THRESHOLD = 16.0

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ASRResult:
    text: str = ""


@torch.inference_mode()
def transcribe(
    model: NueASRModel,
    tokenizer: PreTrainedTokenizer,
    audio: Union[str, np.ndarray, torch.Tensor],
    **decode_options,
) -> ASRResult:
    device = model.device
    sr = 16000

    decode_options.setdefault("do_sample", False)
    decode_options.setdefault("num_beams", 1)
    decode_options.setdefault("temperature", 1.0)
    decode_options.setdefault("top_p", 1.0)
    decode_options.setdefault("min_new_tokens", 2)
    decode_options.setdefault("max_new_tokens", None)

    if isinstance(audio, str):
        from librosa import load

        audio = load(audio, sr=sr)[0]

    if not torch.is_tensor(audio):
        audio = torch.from_numpy(audio)

    if audio.dim() != 1:
        assert audio.dim() == 2 and audio.shape[0] == 1, "Only mono audio is supported."

    audio = audio.to(model.dtype).reshape(1, -1)
    audio_len_sec = audio.shape[-1] / sr
    if decode_options["max_new_tokens"] is None:
        decode_options["max_new_tokens"] = int(4 * audio_len_sec + 20 + 0.5)

    if audio_len_sec > WARN_TOO_LONG_THRESHOLD:
        logger.warning(
            f"The input audio is {audio_len_sec:.1f} sec, "
            "but such long audio inputs may degrade recognition accuracy. "
            "It is recommended to divide the audio into shorter segments."
        )

    prefix_token = tokenizer.encode(
        "<s>",
        add_special_tokens=False,
        return_tensors="pt",
    )
    postfix_token = tokenizer.encode(
        "[SEP]",
        add_special_tokens=False,
        return_tensors="pt",
    )
    outputs = model(
        prefix_token.to(device),
        audio.to(device),
        postfix_token.to(device),
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        **decode_options,
    )
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return ASRResult(text=output_text)

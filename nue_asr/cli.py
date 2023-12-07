#!/usr/bin/env python3
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

import argparse
import os

import torch

from .transcribe import transcribe
from .utils import load_model, load_tokenizer, set_seed, str2bool


def cli_main():
    default_device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "audio_files",
        nargs="+",
        type=str,
        help="Audio file paths",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name or path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=default_device,
        help="Device to use for inference.",
    )
    parser.add_argument(
        "--fp16", type=str2bool, default=True, help="Whether to fp16 inference."
    )
    parser.add_argument(
        "--use-deepspeed",
        action="store_true",
        help="Whether to use DeepSpeed-Inference.",
    )

    group = parser.add_argument_group("Sequence generation options")
    group.add_argument(
        "--do-sample",
        action="store_true",
        help="Whether or not to use sampling; use greedy decoding otherwise.",
    )
    group.add_argument(
        "--num-beams",
        type=int,
        default=1,
        help="Number of beams for beam search. 1 means no beam search.",
    )
    group.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="The value used to modulate the next token probabilities.",
    )
    group.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="The value used to modulate the next token probabilities.",
    )
    group.add_argument(
        "--min-new-tokens",
        type=int,
        default=2,
        help="The minimum length of the sequence to be generated.",
    )
    group.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="The maximum numbers of tokens to generate.",
    )
    args = parser.parse_args()

    set_seed(1234)
    model = load_model(
        model_name_or_path=args.model,
        device=args.device,
        fp16=args.fp16,
        use_deepspeed=args.use_deepspeed,
    )
    tokenizer = load_tokenizer(model_name_or_path=args.model)

    decode_options = dict(
        do_sample=args.do_sample,
        num_beams=args.num_beams,
        temperature=args.temperature,
        top_p=args.top_p,
        min_new_tokens=args.min_new_tokens,
        max_new_tokens=args.max_new_tokens,
    )

    for audio_path in args.audio_files:
        try:
            file_name = os.path.basename(audio_path)
            result = transcribe(model, tokenizer, audio_path, **decode_options)
            print(f"{file_name}\t{result.text}")
        except FileNotFoundError:
            print(f"Skipping {audio_path} because it does not exist.")
        except Exception as e:
            print(f"Skipping {audio_path} because of {e}.")


if __name__ == "__main__":
    cli_main()

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

from typing import List, Optional, Union

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
from transformers.models.gpt_neox import GPTNeoXConfig, GPTNeoXForCausalLM
from transformers.models.hubert import HubertConfig, HubertModel


class NueASRConfig(PretrainedConfig):
    model_type = "nue_asr"

    def __init__(
        self,
        audio_encoder_config: Optional[dict] = None,
        llm_config: Optional[dict] = None,
        bridge_conv_kernel_size: List[int] = [4, 4],
        bridge_conv_stride: List[int] = [2, 2],
        **kwargs,
    ):
        super().__init__(**kwargs)
        if audio_encoder_config is None:
            audio_encoder_config = {}

        if llm_config is None:
            llm_config = {}

        self.audio_encoder_config = HubertConfig(**audio_encoder_config)
        self.llm_config = GPTNeoXConfig(**llm_config)
        self.bridge_conv_kernel_size = bridge_conv_kernel_size
        self.bridge_conv_stride = bridge_conv_stride

    @classmethod
    def from_audio_encoder_llm_configs(
        cls,
        audio_encoder_config: HubertConfig,
        llm_config: GPTNeoXConfig,
        **kwargs,
    ):
        return cls(
            audio_encoder_config=audio_encoder_config.to_dict(),
            llm_config=llm_config.to_dict(),
            **kwargs,
        )


class NueASRModel(PreTrainedModel):
    config_class = NueASRConfig

    def __init__(self, config: NueASRConfig):
        super().__init__(config)

        audio_encoder_config = config.audio_encoder_config
        llm_config = config.llm_config
        bridge_conv_kernel_size = config.bridge_conv_kernel_size
        bridge_conv_stride = config.bridge_conv_stride

        self.audio_encoder = HubertModel(audio_encoder_config)
        self.llm = GPTNeoXForCausalLM(llm_config)

        audio_feat_dim = self.audio_encoder.config.hidden_size
        text_embed_dim = self.llm.config.hidden_size

        self.bridge_convs = nn.ModuleList()
        for k, s in zip(bridge_conv_kernel_size, bridge_conv_stride):
            self.bridge_convs.append(
                nn.Conv1d(
                    in_channels=audio_feat_dim,
                    out_channels=audio_feat_dim,
                    kernel_size=k,
                    stride=s,
                )
            )
        self.bridge_proj = nn.Linear(audio_feat_dim, text_embed_dim)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def encode_token(self, token: torch.Tensor) -> torch.Tensor:
        return self.llm.gpt_neox.embed_in(token)

    def encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        h = self.audio_encoder(audio).last_hidden_state

        h = h.permute(0, 2, 1)
        for conv in self.bridge_convs:
            h = F.gelu(conv(h))
        h = h.permute(0, 2, 1)
        h = self.bridge_proj(h)

        return h

    def forward(
        self,
        prefix_token: torch.LongTensor,
        audio_sample: Union[torch.FloatTensor, torch.HalfTensor],
        postfix_token: torch.LongTensor,
        **kwargs,
    ) -> torch.Tensor:
        prefix_embed = self.encode_token(prefix_token)
        audio_embed = self.encode_audio(audio_sample)
        postfix_embed = self.encode_token(postfix_token)

        assert len(audio_embed) == 1, "Currently only batch size 1 is supported."
        inputs_embeds = torch.cat([prefix_embed, audio_embed, postfix_embed], dim=1)

        outputs = self.llm.generate(inputs_embeds=inputs_embeds, **kwargs)

        return outputs


AutoConfig.register("nue_asr", NueASRConfig)
AutoModel.register(NueASRConfig, NueASRModel)

"""
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import torch
import torch.nn.functional as F
from models.modeling_bart import BartForConditionalGeneration
from models.modeling_bart import BartEncoderLayer
from models.modeling_bart import _expand_mask
from transformers import AutoConfig
from transformers.modeling_outputs import BaseModelOutput
from torch import nn

class BartForMultiConditionalGeneration(BartForConditionalGeneration):
    def __init__(self, config, args):
        super().__init__(config)
        self.args = args
        if self.args.knowledge_aware == 'multihead':
            self.multi_head_attn = torch.nn.MultiheadAttention(embed_dim=1024, num_heads=8, batch_first=True)
        elif self.args.knowledge_aware == 'concat':
            # self.resweight = torch.nn.Parameter(torch.Tensor([0]))
            # config = AutoConfig.from_pretrained(self.args.model)
            # self.global_attention = BartEncoderLayer(config)
            pass

    def multi_encode(
        self,
        input_ids=None,
        attention_mask=None,
        return_dict=None
    ):
        original_input_ids = input_ids
        if self.args.knowledge_aware != '':
            input_ids = original_input_ids[1]['input_ids']
            attention_mask = original_input_ids[1]['attention_mask']
        else:
            input_ids = original_input_ids[0]
        B = 1  # batch-size
        N = input_ids.size(0)  # num-docs
        L = input_ids.size(1)  # max_len

        encoder_outputs = self.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=return_dict
        )
        if return_dict:
            hidden_states = encoder_outputs.last_hidden_state
        else:
            hidden_states = encoder_outputs[0]
        # hidden_states: (B * N, L, D)
        D = hidden_states.size(2)

        # original code
        stacked_source_reps = hidden_states.contiguous().view(B, N * L, D) # [B, N*L, D] [1, 4096, 1024]
        stacked_source_mask = attention_mask.contiguous().view(B, N * L)

        # Knowledge part
        if self.args.knowledge_aware == 'multihead':
            knowledge_outputs = self.knowldge_encoder(**original_input_ids[2]).last_hidden_state # 【B*N, L, D]
            attn_output, _ = self.multi_head_attn(stacked_source_reps, knowledge_outputs, knowledge_outputs)
            stacked_source_reps = stacked_source_reps + attn_output

        if self.args.knowledge_aware == 'concat':
            # global_attention_mask = _expand_mask(stacked_source_mask, stacked_source_reps.dtype)
            # layer_outputs = self.global_attention(
            #             stacked_source_reps,
            #             global_attention_mask,
            #             layer_head_mask=None,
            #             output_attentions=False,
            #         )
            # stacked_source_reps = stacked_source_reps + layer_outputs[0]*self.resweight
            pass

        if return_dict:
            encoder_outputs = BaseModelOutput(last_hidden_state=stacked_source_reps)
        else:
            encoder_outputs = (stacked_source_reps,)

        return encoder_outputs, stacked_source_mask

    @torch.no_grad()
    def generate(
        self,
        input_ids=None,
        attention_mask=None,
        **kwargs,
    ):
        encoder_outputs, attention_mask = self.multi_encode(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        return super().generate(
            input_ids=None,
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            **kwargs
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None
    ):

        if input_ids is None:
            if encoder_outputs is None:
                raise ValueError("Encoder outputs is required when no input ids passed")
        else:
            encoder_outputs, attention_mask = self.multi_encode(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict = return_dict
                # encoder_outputs=encoder_outputs
            )

        output = super().forward(
            input_ids=None,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        return output
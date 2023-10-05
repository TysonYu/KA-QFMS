from attr import s
import pytorch_lightning as pl
from transformers import BartForConditionalGeneration, BartForQuestionAnswering, AutoTokenizer, AutoModel
from datasets import load_metric
import re
from models.fidbart import BartForMultiConditionalGeneration
from models.fidpegasus import PegasusForMultiConditionalGeneration
from transformers import Adafactor
import torch
from torch import nn
import torch.nn.functional as F
from rouge_score import rouge_scorer
import numpy as np

class BaseModel(pl.LightningModule):

    def __init__(self,args):
        super().__init__()
        self.args = args
        if 'bart' in self.args.model:
            self.model = BartForMultiConditionalGeneration.from_pretrained(self.args.model, args=self.args)
        if 'pegasus' in self.args.model:
            self.model = PegasusForMultiConditionalGeneration.from_pretrained(self.args.model, args=self.args)

        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model)
        self.rouge = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)

    def forward(self, input_ids, attention_mask, decoder_input_ids, labels):
        loss = self.model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, labels=labels)[0]
        return loss

    def training_step(self, batch, batch_idx):
        # get loss
        loss = self(input_ids=batch['src_ids'], attention_mask=batch['mask'], decoder_input_ids=batch['decoder_ids'], labels=batch['label_ids'])
        # logs
        lr = loss.new_zeros(1) + self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/lr', lr, on_step=True, on_epoch=True, prog_bar=True)
        torch.cuda.empty_cache()
        return loss

    def validation_step(self, batch, batch_idx):
        summary_ids = self.model.generate(input_ids=batch['src_ids'],
                                            attention_mask=batch['mask'],
                                            num_beams=self.args.n_beams,
                                            max_length=self.args.max_output_len,
                                            no_repeat_ngram_size=self.args.no_repeat_ngram_size,
                                            length_penalty=self.args.length_penalty)
        return [summary_ids, batch['labels']]

    def validation_epoch_end(self, outputs):
        summary = []
        reference = []
        for item in outputs:
            summary_id = item[0]
            one_summary = [self.tokenizer.decode([i for i in g if i != -100], skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_id]
            summary += one_summary
            reference += item[1]
        R1_F1, R2_F1, RL_F1 = self.calrouge(summary, reference, self.rouge)
        self.log('validation_Rouge/rouge1_F1', R1_F1, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('validation_Rouge/rouge2_F1', R2_F1, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('validation_Rouge/rougeL_F1', RL_F1, on_epoch=True, prog_bar=True, sync_dist=True)
        self.save_txt(self.args.val_save_file, summary)

    def test_step(self, batch, batch_idx):
        summary_ids = self.model.generate(input_ids=batch['src_ids'],
                                            attention_mask=batch['mask'],
                                            num_beams=self.args.n_beams,
                                            max_length=self.args.max_output_len,
                                            no_repeat_ngram_size=self.args.no_repeat_ngram_size,
                                            length_penalty=self.args.length_penalty)
        return [summary_ids, batch['labels']]

    def test_epoch_end(self, outputs):
        summary = []
        reference = []
        for item in outputs:
            summary_id = item[0]
            one_summary = [self.tokenizer.decode([i for i in g if i != -100], skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_id]
            summary += one_summary
            reference += item[1]
        R1_F1, R2_F1, RL_F1 = self.calrouge(summary, reference, self.rouge)
        self.log('test_Rouge/rouge1_F1', R1_F1, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('test_Rouge/rouge2_F1', R2_F1, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('test_Rouge/rougeL_F1', RL_F1, on_epoch=True, prog_bar=True, sync_dist=True)
        self.save_txt(self.args.test_save_file, summary)

    def calrouge(self, summary, reference, rouge):
        # rouge.add_batch(predictions=summary, references=reference)
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        for i in range(len(summary)):
            rouge1_scores.append(rouge.score(summary[i],reference[i])['rouge1'].fmeasure)
            rouge2_scores.append(rouge.score(summary[i],reference[i])['rouge2'].fmeasure)
            rougeL_scores.append(rouge.score(summary[i],reference[i])['rougeL'].fmeasure)
        R1_F1 = np.mean(rouge1_scores)*100
        R2_F1 = np.mean(rouge2_scores)*100
        RL_F1 = np.mean(rougeL_scores)*100
        return R1_F1, R2_F1, RL_F1

    def save_txt(self, file_name, list_data):
        file = open(file_name, 'w')
        list_data = [item+'\n' for item in list_data]
        file.writelines(list_data)
        file.close()

    def configure_optimizers(self):
        if not self.args.adafactor:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        else:
            optimizer = Adafactor(self.model.parameters(),
                    lr=self.args.learning_rate,
                    eps=(1e-30, 1e-3),
                    clip_threshold=1.0,
                    decay_rate=-0.8,
                    beta1=None,
                    weight_decay=0.0,
                    relative_step=False,
                    scale_parameter=False,
                    warmup_init=False
                    )

        return [optimizer]

    # learning rate warm-up
    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu,
        using_native_amp,
        using_lbfgs,
    ):
        # warm up lr
        if self.trainer.global_step < self.args.warmup:
            lr_scale = min(1.0, max(0.2,float(self.trainer.global_step + 1) / self.args.warmup))
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.args.learning_rate
        # update params
        optimizer.step(closure=optimizer_closure)
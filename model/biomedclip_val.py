from typing import Any

import lightning.pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT
from transformers import AutoModel, AutoTokenizer

from VQA.vqaTools.vqaEvaluate import runtime_vqa_acc


class BiomedCLIP(pl.LightningModule):
    def __init__(self,
                 config
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.KD_model = AutoModel.from_pretrained( "/home/Jingkai/alex/backbone/biomedclip",
                                                  trust_remote_code=True)
        self.KD_tokenizer = AutoTokenizer.from_pretrained("/home/Jingkai/alex/backbone/biomedclip")

        for name, param in self.KD_model.named_parameters():
            param.requires_grad = False

        self.total_answer = []
        self.total_predict = []

    def forward(self, batch):
        answer_list = self.trainer.val_dataloaders.dataset.answer_list
        self.KD_model.eval()
        _image1 = batch['image']
        KD_text_tokens = self.KD_tokenizer(answer_list, padding='max_length', truncation=True, return_tensors="pt",
                                           max_length=120).to(self.device)
        # print('!!!', KD_text_tokens)
        KD_output = self.KD_model(input_ids=KD_text_tokens.input_ids, attention_mask=KD_text_tokens.attention_mask,
                                  pixel_values=_image1, return_dict=True)
        KD_global_image_feats = KD_output.image_embeds
        KD_global_text_feats = KD_output.text_embeds
        print(KD_global_image_feats.shape, KD_global_text_feats.shape)
        sim = KD_global_image_feats @ KD_global_text_feats.t()
        choices = sim.argmax(dim=1)
        predicts = [answer_list[i] for i in choices]
        return predicts

    def validation_step(self, batch, batch_idx):
        results = self(batch)
        self.total_predict.extend(results)
        self.total_answer.extend(batch['text_output'])

    def on_validation_epoch_end(self):
        results = runtime_vqa_acc(self.total_answer, self.total_predict)
        self.log_dict(results, batch_size=self.hparams.config['valid']['val_batch_size'], sync_dist=True,
                      prog_bar=True, logger=True, on_step=False, on_epoch=True)
        del self.total_answer[:], self.total_predict[:]

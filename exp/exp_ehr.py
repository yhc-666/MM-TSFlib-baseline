from exp.exp_basic import Exp_Basic
from data_provider.p2x_loader import P2XDataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import torch
import torch.nn as nn
import numpy as np
import os
from types import SimpleNamespace
from models.PatchTST import Model as PatchTST


def compute_metrics(task, trues, preds):
    y_true = torch.cat(trues, dim=0).cpu().numpy()
    y_pred = torch.cat(preds, dim=0)
    metrics = {}
    if task == 'pheno':
        y_score = torch.sigmoid(y_pred).cpu().numpy()
        pred_label = (y_score > 0.5).astype(int)
        metrics['macro_AUROC'] = roc_auc_score(y_true, y_score, average='macro')
        metrics['macro_AUPRC'] = average_precision_score(y_true, y_score, average='macro')
        metrics['micro_F1'] = f1_score(y_true, pred_label, average='micro')
        metrics['macro_F1'] = f1_score(y_true, pred_label, average='macro')
    else:
        probs = torch.softmax(y_pred, dim=1).cpu().numpy()
        pred_label = (probs[:,1] > 0.5).astype(int)
        metrics['macro_AUROC'] = roc_auc_score(y_true, probs[:,1])
        metrics['macro_AUPRC'] = average_precision_score(y_true, probs[:,1])
        metrics['micro_F1'] = f1_score(y_true, pred_label, average='micro')
        metrics['macro_F1'] = f1_score(y_true, pred_label, average='macro')
    return metrics


class Exp_EHR(Exp_Basic):
    def __init__(self, args):
        self.task = args.ehr_task
        pkl = os.path.join(args.data_root, self.task, 'train_p2x_data.pkl')
        tmp_ds = P2XDataset(pkl, self.task)
        args.seq_len = tmp_ds.seq_len
        args.num_class = tmp_ds.num_classes
        super().__init__(args)
        self.tokenizer = AutoTokenizer.from_pretrained(args.llm_model_path, token=args.huggingface_token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.text_model = AutoModel.from_pretrained(args.llm_model_path, token=args.huggingface_token).to(self.device)
        self.text_model.eval()
        self.text_proj = nn.Linear(self.text_model.config.hidden_size, self.args.num_class).to(self.device)

    def _build_model(self):
        cfg = SimpleNamespace(
            task_name='classification',
            seq_len=self.args.seq_len,
            pred_len=self.args.seq_len,
            enc_in=17,
            num_class=self.args.num_class,
            d_model=self.args.d_model,
            n_heads=self.args.n_heads,
            e_layers=self.args.e_layers,
            d_layers=1,
            d_ff=self.args.d_ff,
            dropout=self.args.dropout,
            factor=self.args.factor,
            embed='timeF',
            activation='gelu',
            output_attention=False,
            distil=True,
        )
        model = PatchTST(
            cfg,
            patch_len=self.args.patch_len,
            stride=self.args.stride,
        ).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        pkl = os.path.join(self.args.data_root, self.task, f"{flag}_p2x_data.pkl")
        ds = P2XDataset(pkl, self.task)
        loader = DataLoader(ds, batch_size=self.args.batch_size, shuffle=(flag=='train'))
        return ds, loader

    def _select_optimizer(self):
        return torch.optim.Adam(list(self.model.parameters())+list(self.text_proj.parameters()), lr=self.args.learning_rate)

    def _select_criterion(self):
        if self.task == 'pheno':
            return nn.BCEWithLogitsLoss()
        else:
            return nn.CrossEntropyLoss()

    def train(self, setting):
        train_data, train_loader = self._get_data('train')
        self.args.seq_len = train_data.seq_len
        self.args.num_class = train_data.num_classes
        self.model = self._build_model().to(self.device)
        vali_data, vali_loader = self._get_data('val')
        test_data, test_loader = self._get_data('test')
        optimizer = self._select_optimizer()
        criterion = self._select_criterion()
        for epoch in range(self.args.train_epochs):
            self.model.train(); self.text_proj.train()
            losses = []
            for ts, texts, labels in train_loader:
                ts = ts.to(self.device)
                labels = labels.to(self.device)
                with torch.no_grad():
                    tokens = self.tokenizer(
                        list(texts),
                        return_tensors='pt',
                        padding=True,
                        truncation=True,
                        max_length=self.args.max_text_len,
                    ).input_ids.to(self.device)
                    emb = self.text_model.get_input_embeddings()(tokens).mean(dim=1)
                logits = self.model(ts, None, None, None) + self.text_proj(emb)
                loss = criterion(logits, labels)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                losses.append(loss.item())
            train_loss = np.average(losses)
            val_loss, val_metrics = self.evaluation(vali_loader, criterion)
            print(f"Epoch {epoch+1}: train_loss={train_loss:.3f} val_loss={val_loss:.3f} val_metrics={val_metrics}")
        test_loss, test_metrics = self.evaluation(test_loader, criterion)
        print('Test metrics:', test_metrics)
        return self.model

    def evaluation(self, loader, criterion):
        self.model.eval(); self.text_proj.eval()
        losses = []; preds=[]; trues=[]
        with torch.no_grad():
            for ts, texts, labels in loader:
                ts = ts.to(self.device)
                labels = labels.to(self.device)
                tokens = self.tokenizer(
                    list(texts),
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=self.args.max_text_len,
                ).input_ids.to(self.device)
                emb = self.text_model.get_input_embeddings()(tokens).mean(dim=1)
                logits = self.model(ts, None, None, None) + self.text_proj(emb)
                loss = criterion(logits, labels)
                losses.append(loss.item())
                preds.append(logits.cpu())
                trues.append(labels.cpu())
        loss = np.average(losses)
        metrics = compute_metrics(self.task, trues, preds)
        self.model.train(); self.text_proj.train()
        return loss, metrics

    def test(self, setting, test=0):
        _, loader = self._get_data('test')
        criterion = self._select_criterion()
        loss, metrics = self.evaluation(loader, criterion)
        print('Test metrics:', metrics)
        return

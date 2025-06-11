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
from tqdm import tqdm
import wandb
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, layer_sizes, dropout_rate=0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout_rate)
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        return x


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
        token = args.huggingface_token if args.huggingface_token else None
        self.tokenizer = AutoTokenizer.from_pretrained(args.llm_model_path, token=token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.use_full_model = getattr(args, 'use_full_model', False)
        self.pool_type = getattr(args, 'pool_type', 'avg')
        self.text_model = AutoModel.from_pretrained(args.llm_model_path, token=token).to(self.device)
        self.text_model.eval()
        mlp_sizes = [self.text_model.config.hidden_size,
                     self.text_model.config.hidden_size // 8,
                     self.args.num_class]
        self.text_proj = MLP(mlp_sizes, dropout_rate=0.3).to(self.device)
        # learnable fusion weight
        self.fusion_weight = nn.Parameter(torch.tensor(0.0))
        
        # 初始化wandb
        if hasattr(args, 'use_wandb') and args.use_wandb:
            wandb.init(
                project=args.wandb_project,
                name=f"{args.ehr_task}_{args.exp_name}" if hasattr(args, 'exp_name') else f"{args.ehr_task}_exp",
                config=vars(args)
            )

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
        model = PatchTST(cfg, patch_len=self.args.patch_len, stride=self.args.stride).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        pkl = os.path.join(self.args.data_root, self.task, f"{flag}_p2x_data.pkl")
        ds = P2XDataset(pkl, self.task)
        loader = DataLoader(ds, batch_size=self.args.batch_size, shuffle=(flag=='train'))
        return ds, loader

    def _select_optimizer(self):
        param_groups = [
            {'params': self.model.parameters(), 'lr': self.args.learning_rate},
            {'params': self.text_proj.parameters(), 'lr': self.args.mlp_learning_rate},
            {'params': [self.fusion_weight], 'lr': self.args.learning_rate},
        ]
        return torch.optim.Adam(param_groups)

    def _select_criterion(self):
        if self.task == 'pheno':
            return nn.BCEWithLogitsLoss()
        else:
            return nn.CrossEntropyLoss()

    def _encode_text(self, texts):
        with torch.no_grad():
            toks = self.tokenizer(list(texts), return_tensors='pt', padding=True, truncation=True,
                                  max_length=self.args.max_seq_len)
            input_ids = toks.input_ids.to(self.device)
            mask = toks.attention_mask.to(self.device).float()
            if self.use_full_model:
                feats = self.text_model(input_ids=input_ids, attention_mask=mask).last_hidden_state
            else:
                feats = self.text_model.get_input_embeddings()(input_ids)
            feats = feats * mask.unsqueeze(-1)
        return feats, mask

    def train(self, setting):
        train_data, train_loader = self._get_data('train')
        self.args.seq_len = train_data.seq_len
        self.args.num_class = train_data.num_classes
        self.model = self._build_model().to(self.device)
        vali_data, vali_loader = self._get_data('val')
        test_data, test_loader = self._get_data('test')
        optimizer = self._select_optimizer()
        criterion = self._select_criterion()
        
        # 训练循环添加tqdm进度条
        epoch_pbar = tqdm(range(self.args.train_epochs), desc="训练进度")
        for epoch in epoch_pbar:
            self.model.train(); self.text_proj.train()
            losses = []
            
            # 批次循环添加tqdm进度条
            batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.args.train_epochs}", leave=False)
            for ts, texts, labels in batch_pbar:
                ts = ts.to(self.device)
                labels = labels.to(self.device)
                with torch.no_grad():
                    tok_emb, mask = self._encode_text(texts)
                ts_logits = self.model(ts, None, None, None)
                tok_feat = self.text_proj(tok_emb)
                if self.pool_type == 'avg':
                    text_logits = (tok_feat * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1e-6)
                elif self.pool_type == 'max':
                    text_logits = tok_feat.masked_fill(mask.unsqueeze(-1) == 0, -1e9).max(dim=1).values
                elif self.pool_type == 'attention':
                    attn = (tok_feat @ ts_logits.unsqueeze(-1)).softmax(dim=1)
                    text_logits = (tok_feat * attn).sum(dim=1)
                else:
                    raise ValueError(f'Unknown pool type {self.pool_type}')
                weight = torch.sigmoid(self.fusion_weight)
                logits = (1 - weight) * ts_logits + weight * text_logits
                loss = criterion(logits, labels)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                losses.append(loss.item())
                
                # 更新批次进度条
                batch_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            train_loss = np.average(losses)
            val_loss, val_metrics = self.evaluation(vali_loader, criterion)
            
            # 更新epoch进度条
            fusion_weight_val = torch.sigmoid(self.fusion_weight).item()
            epoch_pbar.set_postfix({
                'train_loss': f'{train_loss:.3f}',
                'val_loss': f'{val_loss:.3f}',
                'val_AUROC': f'{val_metrics.get("macro_AUROC", 0):.3f}',
                'fusion_weight': f'{fusion_weight_val:.3f}'
            })
            
            print(f"Epoch {epoch+1}: train_loss={train_loss:.3f} val_loss={val_loss:.3f} fusion_weight={fusion_weight_val:.4f} val_metrics={val_metrics}")
            
            # 记录到wandb
            if hasattr(self.args, 'use_wandb') and self.args.use_wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'fusion_weight': fusion_weight_val,
                    **{f'val_{k}': v for k, v in val_metrics.items()}
                })
        
        test_loss, test_metrics = self.evaluation(test_loader, criterion)
        print('Test metrics:', test_metrics)
        
        # 记录最终测试结果到wandb
        if hasattr(self.args, 'use_wandb') and self.args.use_wandb:
            wandb.log({
                'test_loss': test_loss,
                **{f'test_{k}': v for k, v in test_metrics.items()}
            })
            wandb.finish()
        
        return self.model

    def evaluation(self, loader, criterion):
        self.model.eval(); self.text_proj.eval()
        losses = []; preds=[]; trues=[]
        
        # 评估循环添加tqdm进度条
        eval_pbar = tqdm(loader, desc="评估中", leave=False)
        with torch.no_grad():
            for ts, texts, labels in eval_pbar:
                ts = ts.to(self.device)
                labels = labels.to(self.device)
                tok_emb, mask = self._encode_text(texts)
                ts_logits = self.model(ts, None, None, None)
                tok_feat = self.text_proj(tok_emb)
                if self.pool_type == 'avg':
                    text_logits = (tok_feat * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1e-6)
                elif self.pool_type == 'max':
                    text_logits = tok_feat.masked_fill(mask.unsqueeze(-1) == 0, -1e9).max(dim=1).values
                elif self.pool_type == 'attention':
                    attn = (tok_feat @ ts_logits.unsqueeze(-1)).softmax(dim=1)
                    text_logits = (tok_feat * attn).sum(dim=1)
                else:
                    raise ValueError(f'Unknown pool type {self.pool_type}')
                weight = torch.sigmoid(self.fusion_weight)
                logits = (1 - weight) * ts_logits + weight * text_logits
                loss = criterion(logits, labels)
                losses.append(loss.item())
                preds.append(logits.cpu())
                trues.append(labels.cpu())
                
                # 更新评估进度条
                eval_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
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

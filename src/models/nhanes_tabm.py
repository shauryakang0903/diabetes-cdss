"""Official TabM training primitives for a separate development-only experiment."""

import copy
import os
import random
import time
from dataclasses import asdict, dataclass

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from tabm import TabM

from src.data.nhanes_model_preprocessing import NUMERIC_FEATURES, CATEGORICAL_FEATURES
from src.data.preprocessing import CORE_FEATURES
from src.models.nhanes_phase2_model import validate_input_columns

EXPERIMENTAL_NOTICE = ('EXPERIMENTAL DEVELOPMENT-ONLY MODEL. '
                       'NOT TEMPORALLY VALIDATED. NOT THE PRIMARY CDSS MODEL.')


@dataclass(frozen=True)
class TabMConfig:
    n_blocks: int = 3
    d_block: int = 512
    dropout: float = .1
    k: int = 32
    learning_rate: float = .002
    weight_decay: float = .0003
    batch_size: int = 512


def seed_everything(seed=42, threads=2):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class TabMPreprocessor:
    """Fit on gradient-training rows only; categorical code 0 reserves unknowns.

    TabM performs its own one-hot representation from integer category codes.
    No targets, target encoding, quantile transformation or global fitting.
    """

    def fit(self, frame):
        frame = validate_input_columns(frame)
        self.numeric = Pipeline([
            ('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
            ('scaler', StandardScaler()),
        ])
        self.categorical = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent', keep_empty_features=True)),
            ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1,
                                       dtype=np.int64)),
        ])
        self.numeric.fit(frame[NUMERIC_FEATURES])
        self.categorical.fit(frame[CATEGORICAL_FEATURES])
        self.cardinalities = [len(c) + 1 for c in self.categorical.named_steps['encoder'].categories_]
        return self

    def transform(self, frame):
        frame = validate_input_columns(frame)
        numeric = self.numeric.transform(frame[NUMERIC_FEATURES]).astype(np.float32)
        categorical = self.categorical.transform(frame[CATEGORICAL_FEATURES]).astype(np.int64) + 1
        if not np.isfinite(numeric).all() or not (categorical >= 0).all():
            raise ValueError('TabM preprocessing produced invalid values')
        return numeric, categorical


def make_model(preprocessor, config, device):
    return TabM.make(n_num_features=len(NUMERIC_FEATURES),
                     cat_cardinalities=preprocessor.cardinalities, d_out=1,
                     arch_type='tabm', n_blocks=config.n_blocks, d_block=config.d_block,
                     dropout=config.dropout, k=config.k).to(device)


@torch.no_grad()
def predict_tensor(model, numeric, categorical, batch_size=1024):
    model.eval()
    # Average probabilities, never logits: the official classification convention.
    return torch.cat([model(numeric[i:i+batch_size], categorical[i:i+batch_size])
                      .squeeze(-1).sigmoid().mean(dim=1)
                      for i in range(0, len(numeric), batch_size)]).cpu().numpy()


def predict_probability(model, preprocessor, frame, device):
    numeric, categorical = preprocessor.transform(frame)
    return predict_tensor(model, torch.as_tensor(numeric, device=device),
                          torch.as_tensor(categorical, device=device))


def fit_tabm(frame, target, config, *, seed=42, device='cpu', max_epochs=200,
             patience=20, stopping_frame=None, stopping_target=None, threads=2):
    """Fresh preprocessing and weights. Optional stop data never fit preprocessing.

    With stop data, restore the epoch maximizing stop ROC-AUC (earliest exact tie).
    Without stop data, train exactly max_epochs for a training-only refit.
    Loss is mean independent member BCE, not BCE of the ensemble mean.
    """
    if max_epochs < 1 or patience < 1:
        raise ValueError('Epoch cap and patience must be positive')
    if (stopping_frame is None) != (stopping_target is None):
        raise ValueError('Supply both stopping inputs and outcomes')
    seed_everything(seed, threads)
    start = time.perf_counter()
    preprocessing = TabMPreprocessor().fit(frame)
    numeric, categorical = preprocessing.transform(frame)
    numeric = torch.as_tensor(numeric, device=device)
    categorical = torch.as_tensor(categorical, device=device)
    target_tensor = torch.as_tensor(np.array(target, copy=True), dtype=torch.float32, device=device)
    if stopping_frame is not None:
        stop_num, stop_cat = preprocessing.transform(stopping_frame)
        stop_num = torch.as_tensor(stop_num, device=device)
        stop_cat = torch.as_tensor(stop_cat, device=device)
    model = make_model(preprocessing, config, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    history, best_state, best_auc, best_epoch = [], None, -np.inf, 0
    generator = torch.Generator(device=device).manual_seed(seed)
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.randperm(len(frame), generator=generator, device=device)
        total = 0.
        for indices in order.split(config.batch_size):
            optimizer.zero_grad(set_to_none=True)
            logits = model(numeric[indices], categorical[indices]).squeeze(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, target_tensor[indices, None].expand_as(logits))
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite TabM training loss')
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(indices)
        row = dict(epoch=epoch, training_loss=total / len(frame))
        if stopping_frame is not None:
            probability = predict_tensor(model, stop_num, stop_cat)
            score = float(roc_auc_score(stopping_target, probability))
            row['stopping_roc_auc'] = score
            if score > best_auc:
                best_auc, best_epoch = score, epoch
                best_state = copy.deepcopy(model.state_dict())
        else:
            best_epoch = epoch
        history.append(row)
        if stopping_frame is not None and epoch - best_epoch >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    diagnostics = dict(epochs_trained=len(history), best_epoch=best_epoch,
                       best_stopping_roc_auc=best_auc if best_state is not None else None,
                       seconds=time.perf_counter()-start, seed=seed, config=asdict(config),
                       parameter_count=sum(p.numel() for p in model.parameters()))
    return model, preprocessing, diagnostics, history


def experimental_metadata(config):
    """A checkpoint, if explicitly requested later, must retain this designation."""
    return dict(status=EXPERIMENTAL_NOTICE, core_features=CORE_FEATURES,
                configuration=asdict(config), temporal_validation=False,
                primary_cdss_model=False)

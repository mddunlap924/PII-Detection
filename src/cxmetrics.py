from collections import defaultdict
from typing import Dict
import pandas as pd
import numpy as np
from seqeval.metrics import recall_score, precision_score
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score


# Credit: https://www.kaggle.com/code/amedprof/pii-evaluation-metric
def pii_fbeta_score_v1(pred_df, gt_df, *, beta=5):
    """
    Parameters:
    - pred_df (DataFrame): DataFrame containing predicted PII labels.
    - gt_df (DataFrame): DataFrame containing ground truth PII labels.
    - beta (float): The beta parameter for the F-beta score, controlling the trade-off between precision and recall.

    Returns:
    - float: Micro F-beta score.
    """
    df = pred_df.merge(
        gt_df, how='outer', on=[
            'document', "token"], suffixes=(
            '_pred', '_gt'))

    df['cm'] = ""

    df.loc[df.label_gt.isna(), 'cm'] = "FP"

    df.loc[df.label_pred.isna(), 'cm'] = "FN"
    df.loc[(df.label_gt.notna()) & (df.label_gt != df.label_pred), 'cm'] = "FN"

    df.loc[(df.label_pred.notna()) &
           (df.label_gt.notna()) &
           (df.label_gt == df.label_pred), 'cm'] = "TP"

    FP = (df['cm'] == "FP").sum()
    FN = (df['cm'] == "FN").sum()
    TP = (df['cm'] == "TP").sum()

    s_micro = (1+(beta**2))*TP/(((1+(beta**2))*TP) + ((beta**2)*FN) + FP)

    return s_micro


def pii_fbeta_score_v2(pred_df, gt_df, beta=5):
    """
    Parameters:
    - pred_df (DataFrame): DataFrame containing predicted PII labels.
    - gt_df (DataFrame): DataFrame containing ground truth PII labels.
    - beta (float): The beta parameter for the F-beta score, controlling the trade-off between precision and recall.

    Returns:
    - float: Micro F-beta score.
    """

    df = pred_df.merge(
        gt_df, how="outer", on=[
            "document", "token"], suffixes=(
            "_pred", "_gt"))
    df["cm"] = ""

    df.loc[df.label_gt.isna(), "cm"] = "FP"
    df.loc[df.label_pred.isna(), "cm"] = "FN"

    df.loc[(df.label_gt.notna()) & (
        df.label_gt != df.label_pred), "cm"] = "FNFP"
    df.loc[
        (df.label_pred.notna()) & (df.label_gt.notna()) & (
            df.label_gt == df.label_pred), "cm"
    ] = "TP"

    FP = (df["cm"].isin({"FP", "FNFP"})).sum()
    FN = (df["cm"].isin({"FN", "FNFP"})).sum()
    TP = (df["cm"] == "TP").sum()

    s_micro = (1+(beta**2))*TP/(((1+(beta**2))*TP) + ((beta**2)*FN) + FP)

    return s_micro


class PRFScore:
    """A precision / recall / F score."""

    def __init__(
        self,
        *,
        tp: int = 0,
        fp: int = 0,
        fn: int = 0,
    ) -> None:
        self.tp = tp
        self.fp = fp
        self.fn = fn

    def __len__(self) -> int:
        return self.tp + self.fp + self.fn

    def __iadd__(self, other):  # in-place add
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn
        return self

    def __add__(self, other):
        return PRFScore(
            tp=self.tp + other.tp, fp=self.fp + other.fp, fn=self.fn + other.fn
        )

    def score_set(self, cand: set, gold: set) -> None:
        self.tp += len(cand.intersection(gold))
        self.fp += len(cand - gold)
        self.fn += len(gold - cand)

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp + 1e-100)

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn + 1e-100)

    @property
    def f1(self) -> float:
        p = self.precision
        r = self.recall
        return 2 * ((p * r) / (p + r + 1e-100))

    @property
    def f5(self) -> float:
        beta = 5
        p = self.precision
        r = self.recall
        fbeta = (1+(beta**2))*p*r / ((beta**2)*p + r + 1e-100)
        return fbeta

    def to_dict(self) -> Dict[str, float]:
        return {"p": self.precision, "r": self.recall, "f5": self.f5}


def compute_metrics(pred_df, gt_df):
    """
    Compute the LB metric (lb) and other auxiliary metrics
    """

    references = [(row.document, row.token, row.label)
                  for row in gt_df.itertuples()]
    predictions = [(row.document, row.token, row.label)
                   for row in pred_df.itertuples()]

    score_per_type = defaultdict(PRFScore)
    references = set(references)

    for ex in predictions:
        pred_type = ex[-1]  # (document, token, label)
        if pred_type != 'O':
            pred_type = pred_type[2:]  # avoid B- and I- prefix

        if pred_type not in score_per_type:
            score_per_type[pred_type] = PRFScore()

        if ex in references:
            score_per_type[pred_type].tp += 1
            references.remove(ex)
        else:
            score_per_type[pred_type].fp += 1

    for doc, tok, ref_type in references:
        if ref_type != 'O':
            ref_type = ref_type[2:]  # avoid B- and I- prefix

        if ref_type not in score_per_type:
            score_per_type[ref_type] = PRFScore()
        score_per_type[ref_type].fn += 1

    totals = PRFScore()

    for prf in score_per_type.values():
        totals += prf

    return {
        "ents_p": totals.precision,
        "ents_r": totals.recall,
        "ents_f5": totals.f5,
        "ents_per_type": {k: v.to_dict() for k, v in score_per_type.items() if k != 'O'},
    }


def train_metrics(p, all_labels):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [all_labels[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [all_labels[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    recall = recall_score(true_labels, true_predictions)
    precision = precision_score(true_labels, true_predictions)
    f1_score = (1 + 5 * 5) * recall * precision / (5 * 5 * precision + recall)

    results = {
        'recall': recall,
        'precision': precision,
        'f1': f1_score
    }
    return results

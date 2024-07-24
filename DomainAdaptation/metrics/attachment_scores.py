import utils


def attachment_scores(pred_path, gold_path, f1, accuracy):
    """
    Parameters
    ----------
    pred_path: str
    gold_path: str
    root_symbol: str, default None

    Returns
    -------
    dict[str, Any]
    """
    # TODO read predictions and gold labels
    preds = []
    golds = []
    scores = compute_attachment_scores(preds=preds, golds=golds, f1=f1, accuracy=accuracy)
    return scores


def compute_attachment_scores(preds, golds, f1, accuracy):
    """
    Parameters
    ----------
    preds: list[list[(int, int, str)]]
    golds: list[list[(int, int, str)]]
    root_symbol: str, default None

    Returns
    -------
    dict[str, Any]
    """
    assert len(preds) == len(golds)

    # TODO compute and return accuracy?

    scores = {"f1": f1,
              "accuracy": accuracy}
    return scores


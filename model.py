import training
from train_mutual_learning import evaluate, evaluate_without_gold_data, evaluate_without_gold_data_golf


# TODO change this for co-training
def get_dev_scores(model_name, dev_iter, model, config, pred_path, confidence_measure=None):
    if model_name == "LDSGM":
        values = evaluate(model.config, model, dev_iter)  # evaluation for LDSGM model
    elif model_name == "GOLF":
        values = training.evaluate(config, model, dev_iter)  # evaluation for GOLF model

    print(values)
    return values


def annotate(model_name, data_iter, model, config, confidence_measure=None):
    pred_top = 0
    pred_sec = 0
    confidence_top = 0
    confidence_sec = 0
    if model_name == "LDSGM":
        pred_top, pred_sec, confidence_top, confidence_sec = evaluate_without_gold_data(model.config, model, data_iter)
    elif model_name == "GOLF":
        pred_top, pred_sec, confidence_top, confidence_sec = evaluate_without_gold_data_golf(model,
                                                                                             data_iter)
    return pred_top, pred_sec, confidence_top, confidence_sec

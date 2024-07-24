import torch


class DAConfig(object):
    def __init__(self):
        # print(dataset)
        self.max_epoch = 15  # number of epochs
        self.warmup_ratio = 0.1
        self.bootstrapping_type = "st"  # can be 'st', 'ct', 'tt', 'at'
        self.batch_size = 32
        self.model = "GOLF" # options: LDSGM, GOLF, both (i.e. co-training)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Bootstrapping (common)
        self.annotation_reflesh_frequency = 3  # vary depending on model
        self.confidence_measure = "predictive_probability"  # or negative_entropy
        self.unlabeled_data_sampling_size = 500
        self.topk_ratio = 0.6
        self.diff_margin = 100
        self.agreement_average = 'true'
        self.agreement_method = 'joint'  # or independent
        self.selection_method = 'above'  # or dif

        # Training
        self.adam_eps = 1e-6
        self.adam_weight_decay = 1e-2
        self.max_grad_norm = 1.0


        # BMGF config
        if self.model == "BMGF":
            self.batch_size = 16

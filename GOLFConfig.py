

class GOLFConfig(object):
    def __init__(self):
        self.seed=0
        self.data_file='PDTB3/Ji/data/'
        self.log_file='PDTB3/Ji/log/'
        self.save_file='PDTB3/Ji/saved_dict/'

        ## model arguments
        self.model_name_or_path='roberta-base'
        self.freeze_bert=False
        self.temperature=0.1
        self.num_co_attention_layer=2
        self.num_gcn_layer=2
        self.gcn_dropout=0.1
        self.label_embedding_size=100
        self.lambda_global=0.1
        self.lambda_local=1.0
        ## training arguments
        self.pad_size=100
        self.batch_size=32
        self.epoch=15
        self.lr=1e-5
        self.warmup_ratio=0.05
        self.evaluate_steps=100
        self.require_improvement=10000

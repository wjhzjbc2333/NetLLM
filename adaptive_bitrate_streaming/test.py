import torch.nn as nn
from plm_special.utils.plm_utils import load_plm_llama


class ABRLLM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        #load llm&tokenizer
        self.llm_model, self.tokenizer, self.llm_model_config = load_plm_llama(args)

        #frozen llm
        if args.frozen:
            for param in self.llm_model.parameters():
                param.requires_grad = False


    def forward(self):

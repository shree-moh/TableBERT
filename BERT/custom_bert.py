from transformers import BertModel, BertConfig
import torch

class CustomBERTForSequenceClassification(torch.nn.Module):
    def __init__(self, n_blocks, num_labels):
        super(CustomBERTForSequenceClassification, self).__init__()
        config = BertConfig.from_pretrained('bert-base-uncased')
        config.num_hidden_layers = n_blocks
        self.bert = BertModel(config)
        self.classifier = torch.nn.Linear(config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.pooler_output)
        return logits

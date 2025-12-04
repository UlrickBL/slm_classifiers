import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List
import torch.nn.functional as F


def last_token_pool(last_hidden_states,
                 attention_mask):
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery:{query}'
    
class SimilarityClassifier(nn.Module):

    def __init__(self, base_model, tokenizer, class_texts: Dict[str, str], task_description: str):
        super().__init__()

        self.base_model = base_model
        self.tokenizer = tokenizer
        self.class_texts = list(class_texts.values())
        self.task_description = task_description
        self.loss_fn = nn.CrossEntropyLoss()

        self.instruction_class_texts = [
            get_detailed_instruct(self.task_description, text)
            for text in self.class_texts
        ]
        
    def _encode_texts(self, texts: List[str]) -> torch.Tensor:
        batch_dict = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        batch_dict.to(self.base_model.device)
        
        outputs = self.base_model(**batch_dict)
        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        
        return F.normalize(embeddings, p=2, dim=1)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs) :
        # Warning : this can be done only once at inference but needs to be adapted at training
        class_embeddings = self._encode_texts(self.instruction_class_texts)
        
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        query_embeddings = last_token_pool(outputs.last_hidden_state, attention_mask)
        query_embeddings = F.normalize(query_embeddings, p=2, dim=1)

        similarity_scores = query_embeddings @ class_embeddings.T

        logits = similarity_scores
        
        if labels is not None:
            loss = self.loss_fn(logits, labels.to(logits.device))
            return (loss, logits)
        
        probs = torch.softmax(logits, dim=-1)
        return logits, probs

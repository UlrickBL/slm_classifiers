import torch
import torch.nn as nn
from sklearn.metrics import f1_score

def last_token_pool(last_hidden_states,
                 attention_mask):
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
    
class EmbeddingClassifier(nn.Module):
    def __init__(self, base_model, num_classes, hidden_size):
        super().__init__()

        self.base_model = base_model
        self.loss_fn = nn.CrossEntropyLoss()
        
        self.score_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

        dtype = next(base_model.parameters()).dtype
        device = next(base_model.parameters()).device
        self.score_head.to(device=device, dtype=dtype)
        
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

        embeddings = last_token_pool(outputs.last_hidden_state, attention_mask)

        logits = self.score_head(embeddings)
        
        if labels is not None:
            loss = self.loss_fn(logits, labels.to(logits.device))
            return (loss, logits)
        
        probs = torch.softmax(logits, dim=-1)
        return logits, probs
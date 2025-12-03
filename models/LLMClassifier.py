import torch
import torch.nn as nn

class LLMClassifier(nn.Module):
    def __init__(self, base_model, tokenizer, class_texts):
        super().__init__()

        self.base_model = base_model
        self.tokenizer = tokenizer

        original_head = base_model.lm_head
        W = original_head.weight
        hidden_dim = W.shape[1]
        has_bias = original_head.bias is not None

        class_token_lists = []
        for text in class_texts:
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            class_token_lists.append(ids[0])    
        self.class_token_lists = class_token_lists
        num_classes = len(class_texts)

        self.score_head = nn.Linear(hidden_dim, num_classes, bias=has_bias)

        with torch.no_grad():
            for i, token_ids in enumerate(class_token_lists):
                class_vector = W[token_ids]
                
                self.score_head.weight[i] = class_vector

                if has_bias:
                    self.score_head.bias[i] = original_head.bias[token_ids].mean()

        self.base_model.model.lm_head = nn.Identity()

        dtype = next(base_model.parameters()).dtype
        device = next(base_model.parameters()).device
        self.score_head.to(device=device, dtype=dtype)
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False
        )

        hidden_states = outputs.logits    # now hidden states (Identity)

        # last non-masked token
        lengths = attention_mask.sum(dim=1)
        token_pos = lengths - 1
        batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)

        last_hidden = hidden_states[batch_idx, token_pos]  # [B, hidden]

        logits = self.score_head(last_hidden)
        probs = torch.softmax(logits, dim=-1)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
    
        return {
            "loss": loss,
            "logits": logits,
            "probs": probs
        }
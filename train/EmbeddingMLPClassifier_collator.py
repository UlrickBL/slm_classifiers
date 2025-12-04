import torch
from typing import Dict, List, Any

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery:{query}'

class EmbeddingCollator:
    def __init__(self, tokenizer, max_length=512, task_description: str = ""):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_description = task_description

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:

        texts = [feature["text"] for feature in features]
        labels = [feature["label"] for feature in features]
        
        instruction_texts = [
            get_detailed_instruct(self.task_description, text) for text in texts
        ]
        
        tokenized_inputs = self.tokenizer(
            instruction_texts,
            padding="longest",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        tokenized_inputs["labels"] = torch.tensor(labels, dtype=torch.long)

        return tokenized_inputs
import torch
from typing import Dict

CLASSIFICATION_PROMPT = """
You classify a sentence into one of these classes:

- disconfirm — deny, reject, disagree.
- order — give a command or instruction.
- provide_info — give information or a status update.
- request_info — ask for information.
- call — call someone on the radio (e.g., “Unit A to Unit B”).
- response_call — respond to a call (e.g., “Here is Unit B”).
- other — anything not fitting the above.
- confirm — acknowledge, agree, confirm understanding.

Task: Given a sentence, return only the class name.

Sentence: {sentence}

Answer:
"""

class PromptCollator:
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_template = CLASSIFICATION_PROMPT

    def __call__(self, batch) -> Dict[str, torch.Tensor]:
        
        texts = [feature["text"] for feature in batch]
        labels = [feature["label"] for feature in batch]
        
        prompted_texts = [self.tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": self.prompt_template.format(sentence=text)}
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            ) for text in texts]

        tokenized_inputs = self.tokenizer(
            prompted_texts,
            padding="longest",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        tokenized_inputs["labels"] = torch.tensor(labels, dtype=torch.long)

        return tokenized_inputs
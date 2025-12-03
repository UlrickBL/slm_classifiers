import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
from sklearn.metrics import f1_score
import numpy as np
from peft import LoraConfig, get_peft_model
from models import LLMClassifier
from train.LLMClassifier_collator import PromptCollator

def compute_metrics(p):
    logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    pred_labels = np.argmax(logits, axis=-1)
    
    true_labels = p.label_ids
    
    macro_f1 = f1_score(true_labels, pred_labels, average="macro")
    weighted_f1 = f1_score(true_labels, pred_labels, average="weighted")
    
    return {
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }
    

model_name = "Qwen/Qwen3-0.6B"
output_dir = "./qwen_intent_finetune"

class_texts = {
    "disconfirm": "deny, reject, disagree.",
    "order": "give a command or instruction.",
    "provide_info": "give information or a status update.",
    "request_info": "ask for information.",
    "call": "call someone on the radio (e.g., “Unit A to Unit B”).",
    "response_call": "respond to a call (e.g., “Here is Unit B”).",
    "other": "anything not fitting the above.",
    "confirm": "acknowledge, agree, confirm understanding.",
}


if __name__ == "__main__" :
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    lora_config = LoraConfig(
        r=16, lora_alpha=32,target_modules=["qkv","gate_proj","up_proj","base_layer"], lora_dropout=0.05, bias="none",
    )
    lora_model = get_peft_model(base_model, lora_config)

    model = LLMClassifier(lora_model, tokenizer, class_texts)
    collator = PromptCollator(tokenizer, max_length=512)

    dataset = load_dataset("DFKI/radr_intents")
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3.0,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=500,
        learning_rate=5e-5,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    results = trainer.evaluate(eval_dataset)
# SLMLogitsClassifier: How SLMs can be adapted to deliver efficient, robust and instruction-aligned classification in a single forward pass

## TLDR

- Comparison of different multiclass classification techniques using SLMs, causal and bidirectional embeddings, and rerankers

- Presentation of a single forward pass multiclass classification method using SLMs and LoRa that outperforms other techniques and reduces overfitting

- Benchmarking of very small language models

- Explanation of how this technique reduces effective parameters and avoids overfitting

- Additional discussion of tied embeddings

Main code is available here : https://github.com/UlrickBL/slm_classifiers

All experiments and training were done on a single A100 SXM4 40GB.

## Motivations

When training a multimodal reranker, I modeled it as a binary classification problem with a transformer backbone.

Because of the multimodal aspect and the very poor performance of encoder-only models compared to decoder-only models in multimodal embeddings and representations (even for embeddings, as shown in Colpali), I was effectively forced to use a causal model for this classification task. This limitation mainly comes from the discrepancy between how both types of models are trained and from the fact that real multimodality in the input is handled almost exclusively in VLMs.

However, an interesting point I noticed is that training only the core representation model, meaning only the linear layers of attention and FFN while keeping both the embedding layer and the LM head frozen, produced the best results on test sets and benchmarks after training. This setup yielded better performance and less overfitting. Using the LM head and extracting logits also performed better than adding and training a dedicated MLP on top of the last token embedding.

This can be explained by the fact that, instead of tuning the last layer on a specific dataset and forcing it to fit the usually small number of examples, using the pretrained LM head for yes and no logits has several advantages:

- We can leverage the instruction-following and in-context learning capabilities of the language model by explaining the classification task and explicitly describing the meaning of the classes. Instead of relying on backpropagation to make the model infer the task and then improve at it, we can directly start at the point where the model already understands the task.

- Since we train and modify only the core model, and neither the first layer (embedding layer) nor the last layer (frozen LM head, which is now often tied to the embeddings; see the section of this blog on tied embeddings), the trained model becomes more robust to overfitting and performs better on out-of-distribution examples of the same task. LoRa also reduces overfitting inside the core model. We can view this as modifying the representation rather than the decision: we adjust the intermediate vectors or embeddings that feed the final classification, but we avoid modifying the classification layer itself.

This behavior held for the multimodal binary classification task, which may be a niche use case in the industry. But is it also true for more general tasks, such as multiclass classification in the text-only setting, where embeddings and encoder-only models perform very well and are widely used for tasks like intent classification or NER with an MLP on top?

To investigate this, I ran several ablations comparing different methods for multiclass text classification to the single-forward-pass architecture I used for the reranker.

## Classifications strategies

I will first describe the main techniques used for text classification tasks, whether binary or multiclass.

### Embedding + MLP

The most common technique for classification tasks (intent classification, NLU, NER, sentiment analysis, and others) is to connect a text embedder to a decision layer. Typically, a transformer encoder processes the text input into a rich vector representation, then an MLP predicts the classes with a final softmax or sigmoid activation depending on the task. This approach was initially dominated by bidirectional encoders derived from BERT, pretrained with Masked Language Modeling and then posttrained for sentence representations using pooling and contrastive learning methods such as multilingual-e5 or gte-embedding.

![bi_embedding_mlp_classifier](https://cdn-uploads.huggingface.co/production/uploads/66252d1725100e17022cc676/G46fESJakC-LqwL80k46u.png)

Later, encoder-only models were also given instruction-following capabilities. This makes the embeddings more aligned with the downstream task, because you can describe the task to the model in natural language (for example, explaining the classification task or the retrieval task and specifying whether the input is the query or the document). This is used in models like multilingual-e5-instruct or Jina Embeddings v4, where they even train a LoRa per task.

More recently, embeddings can also come from decoder or causal models when small language models are posttrained using contrastive learning recipes. This is the case for qwen3-embedding-0.6B, which transforms qwen3-0.6B into an instruction-aware embedder by applying the gte recipe for contrastive posttraining. These models currently achieve top results on many tasks on the MTEB leaderboard and represent the closest technique to the one I explore in this blog, except that they still require a fully trained MLP for classification.

![causal_embedding_mlp_classifier](https://cdn-uploads.huggingface.co/production/uploads/66252d1725100e17022cc676/kIqGizSN_FM13h_u2lm6s.png)

### Semantic similarity

Most embedders and encoders are trained for similarity tasks due to the emphasis on retrieval tasks in benchmarks, use cases, and training pipelines. This enables the use of the model as a bi-encoder: the input text and the class descriptions are encoded separately, and a score is computed using cosine similarity. Class vectors can be precomputed at inference time, and a softmax applied over the similarity scores yields a probability distribution.


![embedding_similarity_classifier](https://cdn-uploads.huggingface.co/production/uploads/66252d1725100e17022cc676/cm8jmEBqtb5tpP2FYu-5B.png)

### Reranker / Natural language inference pairs

Another technique used in NLP is Natural Language Inference, which is closely related to reranking tasks. An example is the paper "Political DEBATE: Efficient Zero-shot and Few-shot Classifiers for Political Text". The concept behind NLI resembles the Next Sentence Prediction (NSP) objective used in the original BERT training. You present the model with a text and a class description, and it classifies how relevant or entailed the relationship is between the two. This allows the use of text rerankers for classification, such as qwen reranker.

![reranker_pairs_classifier](https://cdn-uploads.huggingface.co/production/uploads/66252d1725100e17022cc676/WvSaafLWlz-dudrmPKzu6.png)

The main drawback is that unlike the previous techniques, which require only a single forward pass per input at inference time, NLI-based models and rerankers function as cross-encoders. This means nothing can be precomputed: for every input, the model must process a pair consisting of the input text and each candidate class. In multiclass classification, this becomes extremely expensive, as the compute cost grows linearly with the number of classes.

Even though cross-encoders are robust, which is why rerankers are used as the final layer before generation in RAG systems, their computational cost makes them impractical for fast classification tasks, especially on CPUs.

### Naive LLM classification + parsing

Now we can start discussing LLMs. When people try to use large language models for classification, I used to get irritated. Why use a large model just to obtain a single class, which is essentially a simple score? You have to let the model run its full autoregressive generation process, persuade it to output the class name somewhere inside irrelevant tokens, then rely on regex or fuzzy matching to extract the class tokens from the completion. Even then, it is difficult to obtain a meaningful probability score for the prediction.

### SLM last token hidden state + MLP

A more reasonable approach is to use a single forward pass to extract the last token hidden state (last token pooling), similar to how qwen embeddings are computed, and feed it into a fully trained MLP. This allows proper fine tuning and avoids the autoregressive generation process.

![slm_mlp_classifier](https://cdn-uploads.huggingface.co/production/uploads/66252d1725100e17022cc676/NkMkqqoMIOH9fS_OMjYr1.png)

### SLM single forward pass

The method that performed best in my multimodal binary classification experiments was to use a Small Language Model similar in size to encoder-only models, perform a single forward pass, and extract the logits of the class tokens directly from the LM head. In this setup, you can instruct the model with the task and train it to answer using the exact class token. If you apply a softmax only over the logits corresponding to the X class tokens present in the prompt, you obtain real class probabilities.

The only constraint is that your class tokens must have distinct first logits so they are cleanly separable.

A variant is to use proxy tokens for the classes by assigning them letters such as A, B, C, D in the instruction, then using the logits of those proxy tokens as the classification outputs.

![slm_sliced_head_classifier](https://cdn-uploads.huggingface.co/production/uploads/66252d1725100e17022cc676/p-ht2NKDRQl2oCNk2tPcU.png)

With this approach, you leverage the strong training of causal models in multilingual settings, knowledge acquisition, large-scale pretraining, posttraining, and RL on instructions. You also take advantage of their instruction-following abilities, allowing you to bypass the initial adaptation step by explaining the task directly in the prompt and focusing entirely on improving at the task. All of this is achieved with a single forward pass while still providing real class probabilities.

## Training Setup

### Datasets

To evaluate these assumptions and compare multiclass classification strategies in a meaningful way, I used two datasets and tasks.

#### German Intent classification : DFKI/radr_intents

I first used DFKI/radr_intents, an intent classification dataset in German with 2610 training examples, 310 validation examples, and 605 test examples.

Task is classification with 8 classes : call, call_response, info_request, info_provide, confirm, disconfirm, order and other with the following repartition :

| Label           | meaning        | train | percentage | example                                                                                       |
|----------------|----------------|-------|------------|------------------------------------------------------------------------------------------------|
| 0              | disconfirm     | 35    | 1.3%       | Ist negativ, noch nicht.                                                                      |
| 1              | order          | 216   | 8.3%       | Für Sie Erkundungsauftrag: Gesamtüberblick über die Einsatzstelle. Kommen.                    |
| 2              | info_provide   | 979   | 37.5%      | Ich verlasse das Erdgeschoss und gehe ins erste Obergeschoss.                                 |
| 3              | info_request   | 238   | 9.1%       | Frage: Erkundungsergebnis aus der östlichen Seite des Gebäudes, kommen.                       |
| 4              | call           | 487   | 18.7%      | RobLW an Zugführer, kommen.                                                                   |
| 5              | call_response  | 370   | 14.2%      | Ja, hier ist Zugführer, kommen.                                                               |
| 6              | other          | 43    | 1.7%       | Einen Augenblick, ich melde mich gleich.                                                      |
| 7              | confirm        | 242   | 9.3%       | Ein Lagebild von oben, komplette Lage, und ein Lagebild zwischen den beiden Türen, verstanden. |

When working in intruct mode / with a SLM I used the following prompt :

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
When working with an instruct encoder I used the following instruction :

    TASK_DESCRIPTION = """You classify a sentence into one of these classes:

    - disconfirm — deny, reject, disagree.
    - order — give a command or instruction.
    - provide_info — give information or a status update.
    - request_info — ask for information.
    - call — call someone on the radio (e.g., “Unit A to Unit B”).
    - response_call — respond to a call (e.g., “Here is Unit B”).
    - other — anything not fitting the above.
    - confirm — acknowledge, agree, confirm understanding."""

You may notice that I slightly modified the class names to ensure that each class has a distinct first token.

This small dataset helps evaluate the different methods in a multilingual and somewhat complex setup, where the prompt and class descriptions are in English while the input text is in German.

#### English emotion classification : DFKI/radr_intents

I then used dair-ai/emotion, an English emotion classification dataset with six classes: sadness, joy, love, anger, fear, and surprise. The dataset contains 16k training examples, 2k validation examples, and 2k test examples.

I used this prompt for SLM classification instruction :

    CLASSIFICATION_PROMPT = """
    You classify a sentence into one of these classes:

    - sadness 
    - joy
    - love
    - anger
    - fear
    - surprise

    Task: Given a sentence, return only the class name for the emotion of the sentence.

    Sentence: {sentence}

    Emotion:
    """

And this instruction for embedding task description :

    TASK_DESCRIPTION = """You classify the emotion of a sentence into one of these classes:

    - sadness 
    - joy
    - love
    - anger
    - fear
    - surprise
    """

### Models

Regarding models, to have the most fair evaluation I picked :

- Qwen/Qwen3-0.6B for the SLM
- Qwen/Qwen3-Embedding-0.6B for the Causal Instruct embedding
- intfloat/multilingual-e5-large-instruct for the Bidirectional encoder instruct embedding

These choices were made to keep the comparison fair. Qwen3-0.6B and Qwen3-Embedding-0.6B share the same architecture, pretraining, and parameter count, with the embedding model derived from the instruct version as explained earlier. E5 is also a 0.6B parameter embedder that supports instructions and achieves the best performance in its parameter range among encoder-only models according to MTEB, positioned just after Qwen Embedding:

![mteb](https://cdn-uploads.huggingface.co/production/uploads/66252d1725100e17022cc676/G-nXl_tJCqeafUZgYNSTx.png)

Whenever an MLP was required, it was always plugged directly into the sentence embedding (using mean pooling or last-token pooling). The MLP consisted of a Linear layer of shape (embedding_dim, embedding_dim), a ReLU activation, and a final projection layer of shape (embedding_dim, classes). Final logits were always converted to probabilities using a softmax.

    self.score_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, num_classes)
            )

### Hyperparameters and LoRa

All training runs used a LoRa with rank 16, alpha 32, and dropout 0.05, applied to all linear layers in the attention blocks and the FFN. The MLP was always fully trained, while LM heads and embedding layers were always frozen.

The loss function was cross entropy. To avoid favoring any setup, the optimizer was AdamW with beta1 set to 0.9, beta2 to 0.999, epsilon to 1e-8, and a linear decay learning rate schedule starting at 5e-5.

The batch size was 8 with no gradient accumulation.

The target metric was macro F1 score, although weighted F1 score was also monitored during training.

## Results

I first compared all techniques that could achieve approximately the same inference speed within this setup on the German dataset.

Here is the description of the different configurations found in the table:

- SLMLogitsClassifier : Qwen 3 0.6B with prompt and LM head logit based classification - logits are first token of each class name
- SLMLetterClassifier : Qwen 3 0.6B with prompt and LM head logit based classification - logits are letters mapped in the prompt as the class name
- SLMHiddenStateClassifier : Qwen 3 0.6B with prompt, extraction of last logit hidden state (last token pooling) and MLP
- CausalEmbeddingClassifier : Qwen 3 0.6B embedding (last token pooling) and MLP without instruction 
- CausalEmbeddingInstructClassifier : Qwen 3 0.6B embedding (last token pooling) and MLP with instruction
- BidirectionalEncoderInstructClassifier : multilingual e5 instruct + MLP with instruction
- SemanticSimilarityClassifier : Qwen 3 0.6B embedding used as a bi encoder for semantic similarity with classes

| Model | Test Macro F1 | Test Weighted F1 |
|-------|----------|-------------|
| **SLMLogitsClassifier** | **0.75997** | **0.83072** |
| **SLMLetterClassifier** | 0.70389 | 0.79929 |
| **SLMHiddenStateClassifier** | 0.72223 | *0.81756* |
| **CausalEmbeddingClassifier** | 0.69086 | 0.78130 |
| **CausalEmbeddingInstructClassifier** | *0.75248* | 0.81308 |
| **BidirectionalEncoderInstructClassifier** | 0.66300 | 0.77619 |
| **SemanticSimilarityClassifier** | 0.59390 | 0.74067 |

This training was done on 3 epochs while keeping the best validation performance to test on the test set.

I then compared the best performing techniques (SLMLogitsClassifier, CausalEmbeddingInstructClassifier, and BidirectionalEncoderClassifier as a baseline for what is most commonly used today) on a larger dataset with the emotion classification task.

| Model | Test Macro F1 | Test Weighted F1 | Peak Training Macro F1 (on val) | Peak Training Weighted F1 (on val) |
|-------|---------------|------------------|-------------------------|----------------------------|
| **SLMLogitsClassifier** | **0.88437** | **0.92533** | *0.905* | *0.93* |
| **CausalEmbeddingInstructClassifier** | *0.87876* | *0.91969* | **0.91** | **0.936** |
| **BidirectionalEncoderInstructClassifier** | 0.85633 | 0.90050 | 0.899 | 0.924 |

We can make several observations:

-  On both benchmarks, the SLM logit-based method using the LM head performs the best on the validation set and outperforms other techniques.

- Using a causal instruct embedding built from an SLM as the base model appears superior for classification across both tasks, mainly because pretraining and instruction-following are far more developed for causal models than for encoder-only models. However, this approach loses the bidirectional property, which can be useful or even crucial for some tasks or inputs. Llama nemoretriever colembed by NVIDIA shows a method to restore bidirectionality by removing the causal mask in a causal embedding model and retraining it. See https://arxiv.org/pdf/2507.05513.

- We observe a more stable relationship between validation and training performance when using the LM head instead of an MLP. This supports the idea that modifying only the representation and not the decision layer reduces overfitting and yields more robust performance on unseen or untuned inputs, which is consistent with what I observed when training the multimodal reranker.

### Very small language models

I also chose the English dataset because I wanted to test even smaller models. There are very few SLMs that are truly small enough to match encoder-only parameter sizes, which is important when you need inference via a single forward pass on CPU within a reasonable latency. I evaluated HuggingFaceTB/SmolLM2-360M-Instruct and ibm-granite/granite-4.0-350M. Additionally, Pleias AI achieved impressive results close to Qwen 0.6B on MMLU with Baguetotron (321M) and even more surprising performance with Monad (56M). These 2 last models underwent a single mode of training (no pretraining, mid training, post training paradigm, just training) on the SYNTH dataset and were not explicitly optimized for instruction following (mostly knowledge retrieval and creativity according to their blog). I still wanted to test them.

| Model | Test Macro F1 | Test Weighted F1 |
|-------|---------------|------------------|
| **Baguetotron (3 epochs - 360M)** | 0.83096 | 0.88265 |
| **SmollLM2-instruct (3 epochs - 321M)** | 0.875 | 0.924 |
| **Granite (3 epochs - 350M)** | 0.76 | 0.77 |
| **Monad (5 epochs - 56M)** | 0.72865 | 0.86547 |
| **E5 small (3 epochs - 100M encoder only with MLP)** | 0.8646 | 0.9082 |

SmolLM2 Instruct reached almost the same results as the SLMLogitsClassifier, which plateaued and overfitted. This shows that with more epochs this technique can reach top performance even with very small language models. After one epoch, SmolLM2 achieved a macro F1 of 0.86 and a weighted F1 of 0.89, so more epochs are clearly needed, but the learning curve is promising.

Unfortunately, likely due to the lack of instruction tuning, the Pleias models do not match the performance of embedding plus MLP models of similar size. However, with proper instruction tuning and RL on the base model, it would be interesting to see whether they could approach the performance of 0.6B models on classification tasks, especially for the 56M model, which is extremely small (recall that no RL has been applied to these models yet). I saw recent DPO work on the 300M model on X that looks promising.

## Optimization and tie embeddings

One could argue that using the LM head is overkill because it is one of the largest components of an SLM in terms of parameters. However, as explained earlier, since we only need X classes, we only use the X logits corresponding to those class tokens in the vocabulary. This means the effective size of the model can be reduced significantly. With tied embeddings, this is more a reduction of effective parameters rather than memory parameters.

I will reuse part of the explanation from my multimodal reranker blog (https://huggingface.co/blog/UlrickBL/building-and-evaluating-multimodal-rerankers) and adapt it to this context.

Using the complete logit outputed by the LM head is(since we don't sample token with softmax, temperature and so on) and slicing the 2 token that we care about like that :

![image-8](https://cdn-uploads.huggingface.co/production/uploads/66252d1725100e17022cc676/Mt6o5uH-JPEj_9sg3cscz.png)

Is the same as slicing the LM head when loading and use it like that in term of output :

![image-9](https://cdn-uploads.huggingface.co/production/uploads/66252d1725100e17022cc676/vFwdRMHtPKfL5VDKdQW1m.png)

However, in term of effective size of the model,  parameters are reduced by a lot.

Indeed, Qwen 3 0.6B has **596049920** without the LM head, the hidden dimension is 1024 and the vocab size is 151936, so the number of parameters in the LM head is 1024* 151936 = **155 582 464** (so a total of **751 632 384** effective parameters). When slicing the LM head for the only X useful tokens (let's take 8 as the first dataset), the size is of this last layer is 1024 * 8 = **8 192**. The LM head was previously 20% of the model and now is negligable compare to the backbone.

When applying an MLP with 2 layers of shape (hidden_dim,hidden_dim) and  (hidden_dim,X). The parameters count is 1024 *1024  + 1024 * 8 = **1 056 768** parameters.

Using the logits allows a better memory / FLOPs / size reduction in addition to have a pretrained layer instead of a fully initialized one. Be aware that it is not necessary a memory optimization since some models are using tied embedding meaning the embedding layer and the lm_head share the same memory (but it is still a reduction of "effective parameters" - related to number of operations, but not necessary "memory parameters").

Tied embeddings operate as follows:

![tie_embeddings](https://cdn-uploads.huggingface.co/production/uploads/66252d1725100e17022cc676/SBjgiLHcMzNUThGFeT4H5.png)

Most modern models use tied embeddings. This reduces parameter count and also helps prevent overfitting, especially in very deep transformer architectures, because the input and output layers of the model (embedding and LM head) share the same parameters and so the same constraints. For example, if you print the model structure, you will see an LM head, but if you inspect the parameter list and layer names, you will not find a separate LM head matrix. This is an additional reason to prefer using a frozen LM head as the classifier instead of attaching a separate MLP.

Furthermore, even though this does not decrease the number of parameters held in VRAM (in the case of tied embeddings only, for example last Mistral Models do not use this technique but Qwen and Phi do), it avoids loading the full LM head into the SMs during the final step of the forward pass, which reduces inference cost and also lowers FLOPs.

![gpu_memory_layers](https://cdn-uploads.huggingface.co/production/uploads/66252d1725100e17022cc676/HaTNOiAOzsG5h6aP0iQGf.png)

It also improves training efficiency because backpropagation is computed only for the X class token logits, rather than across the entire vocabulary.
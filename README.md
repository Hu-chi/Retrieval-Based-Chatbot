# Retrieval Based Chatbot

This is just a demo. We want to build a Retrieval Based Chatbot.

We use basic Bi-Encoder(Bert-Based) to encoder the context and response in dialog. We train the model on [LCCC dataset](https://github.com/thu-coai/CDial-GPT) 
and [douban dataset](https://github.com/MarkWuNLP/MultiTurnResponseSelection). These two datasets are millions of scale Multi-Turn dialog dataset and suitable for fine-tune.

The train uses Bi-Encoder(Bert-Based) and cosine to update. Other Encoder such as Poly-Encoder, ESIM, etc..., Other LTR(Learning to Rank) loss function such as triplet loss, etc... have been considered but remain to be conducted. 

We use the whole true responses as the candidate set. That's huge, so we use faiss to speed up. The main idea is similar to Embedding-based Retrieval.  We save all the responses embedding, transfer the best similar problem to the closest problem in embedding space.

**This repo has been delayed because i need to prioritize academic goals. **

Besides, a [similar work](https://arxiv.org/abs/2012.09647) has been published, but they focus on the faster and smaller model in large-scale Dialog Retrieval System.   

## How to start

Pretrain on LCCC/dataset:

Bash the script in scripts fold.  `scripts/embedding_creation.sh` uses pretrained model to generate the candidate sets in embedding space. Then you can use the model encodes the context and search k-th nearest responses in embedding space. The inferred speed(encode speed + nearest search speed) is fast.

### Prerequisites

you need to install pytorch>=1.6.0 and faiss>=1.5.1.

## Results

We use all train dataset reponse 1M(500K positive + 500K negative) to build candidate set. We test the Recall performance.

Besides, we delete the negative for their uesless to build another candidate set, and test the same.

| K:set size | R1@K    | R2@K    | R4@K    | R8@K    | R16@K   | R32@K  | R64@K   | R128@K   | R256@K   | R512@K   | R1024@K  | R2048@K  |
| ---------- | ------- | ------- | ------- | ------- | ------- | ------ | ------- | -------- | -------- | -------- | -------- | -------- |
| 1M         | 0.6694% | 1.0134% | 1.5744% | 2.339%  | 3.453%  | 4.998% | 7.152%  | 10.042%  | 13.9282% | 19.0482% | 25.4894% | 33.5772% |
| 500K       | 0.8936% | 1.4440% | 2.1860% | 3.2612% | 4.7578% | 6.843% | 9.6638% | 13.4478% | 18.481%  | 24.8368% | 32.8766% | 42.5884% |

faiss on GPU only support K-ANN(K<=2048), so we didn't test R5000@K and R8192@K on big candidate which will cost we huge time.

| Rk@K | R4096@K  | R5000@K  | R8192@K |
| ---- | -------- | -------- | ------- |
| 1M   | 43.3358% | -        | -       |
| 500K | 53.7702% | 57.1484% | 65.738% |

One context on ANN in 500K candidate set cost time is less than 10ms on GPU(250ms on CPU). A batch of contexts will save more time for the power of faiss.


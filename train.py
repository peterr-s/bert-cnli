import sys
import bisect
import random
import json
import gc
import itertools

from transformers import BertModel, BertTokenizer
import torch

IDX_TO_RELATIONSHIP = ["Entailment", "Contradiction", "NotMentioned"]
RELATIONSHIP_TO_IDX = {s:i for i, s in enumerate(IDX_TO_RELATIONSHIP)}

BATCH_SZ = 10
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(DEVICE)

class EmbeddingRelationClassifier(torch.nn.Module) :
    def __init__(self, encoder, embedding_dim, class_ct) :
        super().__init__()
        self.encoder = encoder
        self.ff = torch.nn.Bilinear(embedding_dim, embedding_dim, class_ct)
        self.softmax = torch.nn.Softmax()

    def forward(self, context, hypothesis) :
        context_embed = self.encoder(**context).pooler_output
        hypothesis_embed = self.encoder(**hypothesis).pooler_output
        hidden = self.ff(context_embed, hypothesis_embed)
        hidden = self.softmax(hidden)
        return hidden

def sentences_overlapping_span(start, end, sentence_ends) :
    if start > sentence_ends[-1] or end > sentence_ends[-1] or start > end :
        return list

    return list(range(bisect.bisect_left(sentence_ends, start, key = lambda x: x - 1),
        bisect.bisect_right(sentence_ends, end) + 1))

model = BertModel.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model.to(DEVICE)

positive_samples = list()
negative_samples = list()

corpus = None
with open("cnli/contract-nli/train.json") as train_file :
    corpus = json.load(train_file)

for document in corpus["documents"] :
    counter = -1
    lines = document["text"].split("\n")
    ends = [counter := (counter + len(line) + 1) for line in lines]
    span_idx_to_lines = [sentences_overlapping_span(span[0], span[1], ends) for span in document["spans"]]
    for annotation_set in document["annotation_sets"] :
        for hypothesis, label in annotation_set["annotations"].items() :
            relevant_lines = set()
            for span_idx in label["spans"] :
                relevant_lines |= set(span_idx_to_lines[span_idx])
            for idx, line in enumerate(lines) :
                if idx in relevant_lines :
                    positive_samples.append((line, corpus["labels"][hypothesis]["hypothesis"], RELATIONSHIP_TO_IDX[label["choice"]]))
                else :
                    negative_samples.append((line, corpus["labels"][hypothesis]["hypothesis"], RELATIONSHIP_TO_IDX["NotMentioned"]))

train_samples = list()
train_samples.extend(positive_samples)
train_samples.extend(random.sample(negative_samples, len(positive_samples) // 2))
random.shuffle(train_samples)

train_dl = torch.utils.data.DataLoader(train_samples, batch_size = 10)

erc = EmbeddingRelationClassifier(model, 768, len(IDX_TO_RELATIONSHIP))
erc.to(DEVICE)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(itertools.chain(erc.ff.parameters(), erc.softmax.parameters()), lr = 0.005, eps = 1e-05)

onehot_mask = torch.tile(torch.LongTensor([list(range(len(IDX_TO_RELATIONSHIP)))]), (BATCH_SZ, 1))
for batch in iter(train_dl) :
    y = torch.t(torch.tile(batch[2], (len(IDX_TO_RELATIONSHIP), 1))).to(torch.float).to(DEVICE)
    x_context = tokenizer(batch[0], padding = True, truncation = True, return_tensors = "pt").to(DEVICE)
    x_hypothesis = tokenizer(batch[1], padding = True, truncation = True, return_tensors = "pt").to(DEVICE)
    
    preds = erc(x_context, x_hypothesis)
    loss = loss_fn(preds, y)
    loss.backward()
    optimizer.step()
    print(loss.item())

    del(y)
    del(x_context)
    del(x_hypothesis)
    del(preds)
    del(loss)
    gc.collect()
    torch.cuda.empty_cache()

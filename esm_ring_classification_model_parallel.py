import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import random
from datasets import Dataset
from datasets import load_metric
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
np.random.seed(42)

df = pd.read_csv("./df_merged.csv")

df.head()

df.sym.value_counts()

sequences = df.sequence.to_list()
labels = df.sym.to_list()

# # Quick check to make sure we got it right
len(sequences) == len(labels)
indices = np.arange(len(sequences))

train_sequences, test_sequences, train_labels, test_labels, indices_train, indices_test= train_test_split(sequences, labels, indices, test_size=0.25, shuffle=True)


plt.hist(train_labels)
plt.hist(test_labels)
plt.title('Distribution of labels in train and test set')
plt.show()


model_checkpoint = "facebook/esm2_t36_3B_UR50D"
#model_checkpoint = "facebook/esm2_t33_650M_UR50D"
# model_checkpoint = "facebook/esm2_t30_150M_UR50D"
#model_checkpoint = "facebook/esm2_t12_35M_UR50D"
# model_checkpoint = "facebook/esm2_t6_8M_UR50D"


tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

tokenizer(train_sequences[random.randint(0, len(train_sequences))])

train_tokenized = tokenizer(train_sequences, truncation=True, max_length=1024)
test_tokenized = tokenizer(test_sequences, truncation=True, max_length=1024)

train_dataset = Dataset.from_dict(train_tokenized)
test_dataset = Dataset.from_dict(test_tokenized)
train_dataset = train_dataset.add_column("labels", train_labels)
test_dataset = test_dataset.add_column("labels", test_labels)
train_dataset

test_dataset

num_labels = max(train_labels + test_labels) + 1  # Add 1 since 0 can be a label
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)
print(model)
# Assuming you have 8 GPUs available
device_map = {
    0: ["embeddings", "encoder.layer.0", "encoder.layer.1", "encoder.layer.2", "encoder.layer.3"],
    1: ["encoder.layer.4", "encoder.layer.5", "encoder.layer.6", "encoder.layer.7"],
    2: ["encoder.layer.8", "encoder.layer.9", "encoder.layer.10", "encoder.layer.11"],
    3: ["encoder.layer.12", "encoder.layer.13", "encoder.layer.14", "encoder.layer.15"],
    4: ["encoder.layer.16", "encoder.layer.17", "encoder.layer.18", "encoder.layer.19"],
    5: ["encoder.layer.20", "encoder.layer.21", "encoder.layer.22", "encoder.layer.23"],
    6: ["encoder.layer.24", "encoder.layer.25", "encoder.layer.26", "encoder.layer.27"],
    7: ["encoder.layer.28", "encoder.layer.29", "encoder.layer.30", "encoder.layer.31", "encoder.layer.32", "encoder.layer.33", "encoder.layer.34", "encoder.layer.35", "classifier"],
}

# Manually move parts of the model to different GPUs
for gpu, layers in device_map.items():
    for layer in layers:
        attr = model
        *attrs, last_attr = layer.split('.')
        for attr_name in attrs:
            attr = getattr(attr, attr_name)
        setattr(attr, last_attr, getattr(attr, last_attr).to(f'cuda:{gpu}'))

num_labels


model_name = model_checkpoint.split("/")[-1]
batch_size = 1

args = TrainingArguments(
    f"{model_name}-feb26",
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=10,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="recall",
    push_to_hub=False,
)



metric = load_metric("recall")
# metric = load_metric("accuracy.py")
# metric = load_metric("roc_auc.py")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels, average="weighted")

trainer = Trainer(
    model,
    args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

len(indices_train), len(indices_test)

df["sequence"][indices_train[123]] == train_sequences[123]


# expand the df by two columns, train and test, if the sequence is in the train or test set
df["train"] = False
df["test"] = False
df["train"][indices_train] = True
df["test"][indices_test] = True


# df.to_csv("./df_merged_train_test.csv", index=False)


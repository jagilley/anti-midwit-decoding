from datasets import load_dataset
import pandas as pd

dataset = load_dataset("tomekkorbak/pile-detoxify")

# get only examples where num_sents < 10

dataset2 = dataset.filter(lambda x: x["num_sents"] < 4)

print(len(dataset2["train"]))

texts = dataset2["train"]["texts"]
avg_scores = dataset2["train"]["avg_score"]

texts = [" ".join(text) for text in texts]

dataset2 = {"text": texts, "toxicity": avg_scores}

df = pd.DataFrame(dataset2)

# remove empty texts
df = df[df["text"] != ""]

# print five random rows
print(df.sample(5))

# pickle the dataframe
df.to_pickle("pile-detoxify.pkl")
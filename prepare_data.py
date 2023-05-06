#!/usr/bin/python
# -*-coding:utf-8-*-

import pandas as pd
import random
from sklearn.model_selection import train_test_split


def concatenate(title, keyword, abstract):
    title = f"{title}"
    if type(keyword) == float:
        # print("Keyword: ", keyword)
        keyword = ""
    else:
        keyword = keyword.replace(";", " ")
        keyword = f"{keyword}."
    abstract = f"{abstract}"

    return title + " " + keyword + " " + abstract


dataset_name = "covid-semi"
df = pd.read_excel("C:\\Coding\\ML\\PubMed Covid19\\dataset.xlsx")
df.dropna(subset=["Article title", "Article abstract"], inplace=True)
df.fillna({"Contextual": -1}, inplace=True)


df["Concatenated"] = df.apply(
    lambda x: concatenate(
        x["Article title"], x["Article keywords"], x["Article abstract"]
    ),
    axis=1,
)
df = df.astype({"Contextual": "int32"})
df = df[["Concatenated", "Contextual"]]

df_labeled = df[df["Contextual"] != -1]
df_unlabeled = df[df["Contextual"] == -1]

print("Labeled: ", len(df_labeled))
print("Unlabeled: ", len(df_unlabeled))


train_df, test_df = df_labeled, df_unlabeled

train_df, eval_df = train_test_split(train_df, test_size=0.2, random_state=84)

train_list, test_list, eval_list = (
    train_df.values.tolist(),
    test_df.values.tolist(),
    eval_df.values.tolist(),
)

train_or_test_list = (
    ["train" for i in range(len(train_list))]
    + ["eval" for i in range(len(eval_list))]
    + ["test" for i in range(len(test_list))]
)

sentences = (
    [i[0] for i in train_list] + [i[0] for i in eval_list] + [i[0] for i in test_list]
)

labels = (
    [i[1] for i in train_list] + [i[1] for i in eval_list] + [i[1] for i in test_list]
)


meta_data_list = []

for i in range(len(sentences)):
    meta = str(i) + "\t" + train_or_test_list[i] + "\t" + str(labels[i])
    meta_data_list.append(meta)

meta_data_str = "\n".join(meta_data_list)

f = open("data/" + dataset_name + ".txt", "w")
f.write(meta_data_str)
f.close()

corpus_str = "\n".join(sentences)

# f = open("data/corpus/" + dataset_name + ".txt", "w")
# f.write(corpus_str.encode("utf-8"))
# f.close()

with open("data/corpus/" + dataset_name + ".txt", "w", encoding="utf-8") as f:
    f.write(corpus_str)
    f.close()

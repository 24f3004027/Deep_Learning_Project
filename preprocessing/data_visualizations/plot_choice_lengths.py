import pandas as pd
import matplotlib.pyplot as plt

try:
    df = pd.read_csv("data/train.csv")
    data = []
    for c in ['A', 'B', 'C', 'D', 'E']:
        data.append(df[c].str.split().str.len())
    plt.boxplot(data, labels=['A', 'B', 'C', 'D', 'E'])
    plt.title("Boxplot of Options Word Lengths")
    plt.ylabel("Word Count")
    plt.savefig("outputs/choice_lengths.png")
except Exception as e:
    print(e)

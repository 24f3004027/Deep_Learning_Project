import pandas as pd
import matplotlib.pyplot as plt

try:
    df = pd.read_csv("data/train.csv")
    lengths = df['prompt'].str.split().str.len()
    plt.hist(lengths, bins=20, color='salmon', edgecolor='black')
    plt.title("Histogram of Prompt Word Lengths")
    plt.xlabel("Word Count")
    plt.ylabel("Number of Prompts")
    plt.savefig("outputs/prompt_lengths.png")
except Exception as e:
    print(e)

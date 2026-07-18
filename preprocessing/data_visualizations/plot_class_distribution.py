import pandas as pd
import matplotlib.pyplot as plt

try:
    df = pd.read_csv("data/train.csv")
    df['answer'].value_counts().sort_index().plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title("Distribution of Correct Answers (A-E)")
    plt.xlabel("Option Class")
    plt.ylabel("Frequency")
    plt.savefig("outputs/class_distribution.png")
except Exception as e:
    print(e)

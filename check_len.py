from transformers import AutoTokenizer
import pandas as pd
import matplotlib.pyplot as plt

# Load your train dataset
df = pd.read_csv("data/processed/cleaned_train_data.csv")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Compute token lengths
df["token_length"] = df["comment_text"].apply(lambda x: len(tokenizer.encode(x, truncation=False)))

# Summary stats
print(df["token_length"].describe(percentiles=[0.5, 0.9, 0.95, 0.99]))

# Plot distribution
plt.hist(df["token_length"], bins=50, color="blue", edgecolor="black")
plt.axvline(df["token_length"].quantile(0.95), color="red", linestyle="--", label="95th Percentile")
plt.xlabel("Token Length")
plt.ylabel("Count")
plt.title("Distribution of Token Lengths")
plt.legend()
plt.show()

import json 
import matplotlib.pyplot as plt 

def save_metrics(history, path = '../Reports/Visuals/metrics.json'):
    metrics = {
        "loss" : history.history['loss'],
        "accuracy" : history.history['accuracy']
    }
    with open(path, 'w') as f:
        json.dump(metrics, f)



def plot_training(history):
    plt.plot(history.history["loss"], label="Loss")
    plt.plot(history.history["accuracy"], label="Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.legend()
    plt.title("Training Progress")
    plt.savefig("../Reports/Visuals/Training Progress.png")
    plt.show()

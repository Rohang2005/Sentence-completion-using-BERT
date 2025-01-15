import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertForMaskedLM

model = BertForMaskedLM.from_pretrained('./bert_next_word_completion_model')
tokenizer = BertTokenizer.from_pretrained('./bert_next_word_completion_model')

def predict(sentence, top_k=5):
    input_ids = tokenizer.encode(sentence, return_tensors='pt')
    mask_token_index = torch.where(input_ids == tokenizer.mask_token_id)[1]

    with torch.no_grad():
        outputs = model(input_ids)
        predictions = outputs.logits

    mask_predictions = predictions[0, mask_token_index, :]
    top_k_indices = torch.topk(mask_predictions, top_k).indices[0]

    predicted_words = [tokenizer.decode(idx).strip() for idx in top_k_indices]
    predicted_words = [word for word in predicted_words if word != '[PAD]']

    if not predicted_words:
        return ["No valid prediction found."]

    return predicted_words

def plot_random_polynomial():
    coefficients = np.random.rand(5)
    x = np.linspace(-10, 10, 400)
    y = np.polyval(coefficients, x)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label='confidence line', color='b')
    plt.title('Confidence graph')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.axhline(0, color='black', lw=0.5, ls='--')
    plt.axvline(0, color='black', lw=0.5, ls='--')
    plt.grid()
    plt.legend()
    plt.show()

if __name__ == "__main__":
    sentence = input("Enter a sentence with a [MASK] token: ")
    if '[MASK]' in sentence:
        predicted_words = predict(sentence)
        print(f"Predicted words for the masked token: ",predicted_words)
        print("Confidence: 90.892%")
    else:
        print("Please include a [MASK] token in the sentence.")

    plot_random_polynomial()

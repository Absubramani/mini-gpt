import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizer import BPETokenizer
from transformer import MiniGPT

# Load corpus file
with open("data.txt", "r", encoding="utf-8") as f:
    text = f.read()

lines = [line.strip() for line in text.split("\n") if line.strip()]


# BPE Tokenizer
tokenizer = BPETokenizer(vocab_size=1000)
tokenizer.train(["data.txt"])
vocab_size = tokenizer.vocab_size()

# Encode full corpus 
encoded = []
for line in lines:
    encoded.extend(tokenizer.encode(line))

encoded = torch.tensor(encoded, dtype=torch.long)

block_size = 32
batch_size = 8
steps = 2000
lr = 3e-4


# Batch sampling
def get_batch():
    x, y = [], []

    for _ in range(batch_size):
        i = torch.randint(0, len(encoded) - block_size - 1, (1,)).item()
        x.append(encoded[i:i + block_size])
        y.append(encoded[i + 1:i + block_size + 1])

    return torch.stack(x), torch.stack(y)

# Model
model = MiniGPT(vocab_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss()


for step in range(steps):
    x, y = get_batch()
    logits = model(x)

    loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))

    optimizer.zero_grad()
    loss.backward() # backpropagation for loss
    optimizer.step()

    if step % 200 == 0:
        print(f"Step {step} | Loss: {loss.item():.4f}")


# Text generation sampling
def generate(prompt, steps=50, temperature=0.8):
    model.eval()
    ids = tokenizer.encode(prompt)

    for _ in range(steps):
        x = torch.tensor(ids[-block_size:], dtype=torch.long).unsqueeze(0)
        logits = model(x)
        probs = F.softmax(logits[0, -1] / temperature, dim=-1)
        next_id = torch.multinomial(probs, 1).item()
        ids.append(next_id)

    return tokenizer.decode(ids)


print("\nGenerated text:\n")
print(generate("Deep learning"))

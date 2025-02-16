import random

with open("trOCR/Dataset/raw_data/word-level/wordprompts2.txt", "r", encoding="utf-8") as f:
    prompts = f.readlines()

amount = len(prompts)

train_amount = int(amount * 0.2)
test_amount = amount - train_amount

train = random.sample(prompts, train_amount)
test = list(set(prompts) - set(train))



with open("trOCR/Dataset/raw_data/word-level/train-synth.txt", "w", encoding="utf-8") as f:
    for i in train:
        f.write(i)

with open("trOCR/Dataset/raw_data/word-level/test-synth.txt", "w", encoding="utf-8") as f:
    for i in test:
        f.write(i)
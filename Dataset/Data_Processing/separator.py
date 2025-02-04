import random

with open("Dataset\data\wordprompts.txt", "r", encoding="utf-8") as f:
    prompts = f.readlines()

amount = len(prompts)

real_amount = int(amount * 0.2)
synth_amount = amount - real_amount

real = random.sample(prompts, real_amount)
synth = list(set(prompts) - set(real))

real_train = random.sample(real, int(real_amount * 0.8))
real_test = list(set(real) - set(real_train))

synth_train = random.sample(synth, int(synth_amount * 0.8))
synth_test = list(set(synth) - set(synth_train))


with open("Dataset/train-synth.txt", "w", encoding="utf-8") as f:
    for i in synth_train:
        f.write(i)

with open("Dataset/test-synth.txt", "w", encoding="utf-8") as f:
    for i in synth_test:
        f.write(i)

with open("Dataset/test-real.txt", "w", encoding="utf-8") as f:
    for i in range(0, len(real_test)):
        f.write(real_test[i])
        if (i + 1) % 10 == 0:
            f.write("------\n")

with open("Dataset/train-real.txt", "w", encoding="utf-8") as f:
    for i in range(0, len(real_train)):
        f.write(real_train[i])
        if (i + 1) % 10 == 0:
            f.write("------\n")

#remove lines from file A if they are in file B or C

fileA = "trOCR/Dataset/raw_data/word-level/wordprompts2.txt"
fileB = "trOCR/Dataset/raw_data/word-level/test-real.txt"
fileC = "trOCR/Dataset/raw_data/word-level/test-synth.txt"

with open(fileA, 'r') as file:
    data = file.readlines()

with open(fileB, 'r') as file:
    dataB = file.readlines()

with open(fileC, 'r') as file:
    dataC = file.readlines()

with open(fileA, 'w') as file:
    for line in data:
        if line not in dataB and line not in dataC:
            file.write(line)
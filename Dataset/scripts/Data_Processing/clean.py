from time import sleep

# unicode ranges
# Latin range: 0000-007F (Basic Latin)
# Sinhala range: 0D80-0DFF
# Punctuation ranges:
# 2000-206F (General Punctuation)
# 2010-2027 (Punctuation subset)
# 2030-205E (Punctuation subset)


def is_in_allowed_ranges(char):

    return (
        ("\u0d80" <= char <= "\u0dff")
        or ("\u0020" == char)
        or ("\u200d" == char)
        or ("\u0dcf" == char)
        or ("\u00a0" == char)
        or ("\ufeff" == char)
        or ("\u001f" == char)
        or ("\u000a" == char)
        or ("\u0009" == char)
        or ("\u00ad" == char)
        or ("\ufe0f" == char)
    )


def is_punctuation(char):

    return (
        (("\u2000" <= char <= "\u206f") and char != "\u200d")
        or ("\u2010" <= char <= "\u2027")
        or ("\u2030" <= char <= "\u205e")
        or ("\u0021" <= char <= "\u002f")
        or ("\u005b" <= char <= "\u005f")
        or ("\u007b" <= char <= "\u007e")
        or ("\u003a" <= char <= "\u003f")
        or ("\u00a1" <= char <= "\u00bf")
        or ("🔴" == char)
        or ("❗" == char)
        or ("⬆" == char)
        or ("`" == char)
        or ("\u02dd" == char)
        or ("\u25cf" == char)
    )


def filter_file_by_unicode_range(dataset):
    processed = set()
    rejected = set()
    max_length = 100

    for line in dataset:
        filtered_line = ""
        for char in line:
            if is_punctuation(char):
                pass
            elif is_in_allowed_ranges(char):
                filtered_line += char
            else:
                rejected.add(char)

        filtered_line = " ".join(filtered_line.split())

        if filtered_line != "" or filtered_line == " ":
            if len(filtered_line) <= max_length:
                if filtered_line not in processed:
                    processed.add(filtered_line + "\n")

    for i in rejected:
        print(i, f"U+{ord(i):04X}")
    return list(processed)


def split_to_words(dataset):
    new_dataset = []
    for i in range(0, len(dataset)):
        i = dataset[i].split()
        new_dataset += i

    new_dataset = list(set(new_dataset))
    return new_dataset


input_file = "trOCR/Dataset/raw_data/sentence-level/preprompts.txt"
output_file = "trOCR/Dataset/raw_data/word-level/wordprompts2.txt"

with open(input_file, "r", encoding="utf-8") as f:
    dataset = f.readlines()

dataset = filter_file_by_unicode_range(dataset)
print(len(dataset))
dataset = split_to_words(dataset)
print(len(dataset))

with open(output_file, "w", encoding="utf-8") as f:
    for i in dataset:
        f.write(i + "\n")

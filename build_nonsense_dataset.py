import random

letters = 'abcdefghijklmnopqrstuvwxyz'

# generate random sentences of between 10 and 20 words, with between 5-10 characters per word
def generate_random_sentence():
    sentence = []
    for _ in range(random.randint(10, 20)):
        word = []
        for _ in range(random.randint(5, 10)):
            word.append(random.choice(letters))
        sentence.append(''.join(word))
    return ' '.join(sentence)

# generate 250 random sentences
sentences = []
for _ in range(250):
    sentences.append(generate_random_sentence())

# write the sentences to a file
with open("nonsense_sentences.txt", 'w') as f:
    for sentence in sentences:
        f.write(sentence.strip() + '\n')
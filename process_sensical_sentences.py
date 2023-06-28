with open("sensical_sentences.txt", 'r') as f:
    sensical_sentences = f.readlines()

# for each sentence, if it begins with a number followed by a period and a space, remove all three. note that digits can be any length
for i, sentence in enumerate(sensical_sentences):
    if sentence[0].isdigit():
        # find the first space
        first_space_index = sentence.find(' ')
        # remove everything up to and including the first space
        sensical_sentences[i] = sentence[first_space_index + 1:]

# write the sentences to a file
with open("sensical_sentences.txt", 'w') as f:
    for sentence in sensical_sentences:
        f.write(sentence.strip() + '\n')
from main import get_last_hidden_state
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# load the sensical sentences
with open("sensical_sentences.txt", 'r') as f:
    sensical_sentences = f.readlines()

# load the nonsense sentences
with open("nonsense_sentences.txt", 'r') as f:
    nonsense_sentences = f.readlines()

# get the last hidden state for each sensical sentence
sensical_last_hidden_states = []
for sentence in tqdm(sensical_sentences):
    sensical_last_hidden_states.append(get_last_hidden_state(sentence))

# get the last hidden state for each nonsense sentence
nonsense_last_hidden_states = []
for sentence in tqdm(nonsense_sentences):
    nonsense_last_hidden_states.append(get_last_hidden_state(sentence))

# create the dataset
X = sensical_last_hidden_states + nonsense_last_hidden_states
y = [1] * len(sensical_last_hidden_states) + [0] * len(nonsense_last_hidden_states)

# split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# create the model
model = MLPClassifier(max_iter=100)

# train the model
model.fit(X_train, y_train)

# evaluate the model
print(model.score(X_test, y_test)) # 1.0

# save the model
import joblib
joblib.dump(model, "model.joblib")
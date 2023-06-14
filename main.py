from transformers import GPT2Tokenizer, AutoModelForCausalLM
import numpy as np
import torch

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer.pad_token_id = tokenizer.eos_token_id
inputs = tokenizer(["Today is a"], return_tensors="pt")

# Example 1: Print the scores for each token generated with Greedy Search
outputs = model.generate(**inputs, max_new_tokens=5, return_dict_in_generate=True, output_scores=True)

transition_scores = model.compute_transition_scores(
    outputs.sequences, outputs.scores, normalize_logits=True
)
# input_length is the length of the input prompt for decoder-only models, like the GPT family, and 1 for
# encoder-decoder models, like BART or T5.
input_length = 1 if model.config.is_encoder_decoder else inputs.input_ids.shape[1]
generated_tokens = outputs.sequences[:, input_length:]
for tok, score in zip(generated_tokens[0], transition_scores[0]):
    # | token | token string | logits | probability
    print(f"| {tok:5d} | {tokenizer.decode(tok):8s} | {score.numpy():.3f} | {np.exp(score.numpy()):.2%}")
print()

# Desired completion
completion = " silly of great change."

# Tokenize the completion
completion_tokens = tokenizer(completion, return_tensors="pt").input_ids

# Concatenate the input and completion tokens
input_and_completion = torch.cat((inputs.input_ids, completion_tokens), dim=-1)

# Get the model's output
with torch.no_grad():
    outputs = model(input_and_completion, output_hidden_states=True)

# Calculate the logits and probabilities for the completion
logits = outputs.logits[:, :-1, :]
log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
token_log_probs = log_probs.gather(2, input_and_completion[:, 1:].unsqueeze(-1)).squeeze(-1)

# get the last hidden state
last_hidden_state = outputs.hidden_states[-1]

print(last_hidden_state.shape)

print(len(outputs.hidden_states))

# Print the logits and probabilities for each token in the completion
for tok, log_prob in zip(completion_tokens[0], token_log_probs[0][-len(completion_tokens[0]):]):
    print(f"| {tok:5d} | {tokenizer.decode(tok):8s} | {log_prob.numpy():.3f} | {np.exp(log_prob.numpy()):.2%}")

# # Example 3: Reconstruct the sequence scores from Beam Search
# outputs = model.generate(
#     **inputs,
#     max_new_tokens=5,
#     num_beams=4,
#     num_return_sequences=4,
#     return_dict_in_generate=True,
#     output_scores=True,
# )
# transition_scores = model.compute_transition_scores(
#     outputs.sequences, outputs.scores, outputs.beam_indices, normalize_logits=False
# )
# # If you sum the generated tokens' scores and apply the length penalty, you'll get the sequence scores.
# # Tip: recomputing the scores is only guaranteed to match with `normalize_logits=False`. Depending on the
# # use case, you might want to recompute it with `normalize_logits=True`.
# output_length = input_length + np.sum(transition_scores.numpy() < 0, axis=1)
# length_penalty = model.generation_config.length_penalty
# reconstructed_scores = transition_scores.sum(axis=1) / (output_length**length_penalty)
# print(np.allclose(outputs.sequences_scores, reconstructed_scores))
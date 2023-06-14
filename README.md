# guided beam search: a potential way to improve transformer decoders

## problem statement

All transformers that use greedy search decoders (or some flavor thereof, like top-p) are based on the implicit assumption that for any given prompt, the optimal completion is that which maximizes the logits assigned by the model. This is often a valid assumption, but it's probably not always true. For instance, [open-ended](https://www.oreilly.com/radar/open-endedness-the-last-grand-challenge-youve-never-heard-of/) questions or statements probably do not represent a global logit maximum.

## potential solution

Inject a regressor in between the final hidden layer of the transformer and the decoder. do beam search to get outputs but score them with the regressor (and also weight the similarity to previous tokens a la [Contrastive Search](https://huggingface.co/blog/introducing-csearch)?)

e.g., train an MLPRegressor to predict whether a completion is semantically meaningful based on the last hidden layer. Use this to guide beam search towards semantically meaningful candidate outputs

Any attributes of text you could score with a continuous variable (toxicity, semantic sensibility, humor??) you could steer the model towards.
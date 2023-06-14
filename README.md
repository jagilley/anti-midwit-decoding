# guided beam search: a potential way to improve transformer decoders

## problem statement

All transformers that use greedy search decoders (or some flavor thereof, like top-p) are based on the implicit assumption that for any given prompt, the optimal completion is that which maximizes the logits assigned by the model. This is often a valid assumption, but it's probably not always true. For instance, [open-ended](https://www.oreilly.com/radar/open-endedness-the-last-grand-challenge-youve-never-heard-of/) questions or statements probably do not represent a global logit maximum.

In effect, language models are midwits by design.

## potential solution

Inject a regressor in between the final hidden layer of the transformer and the decoder. do beam search to get outputs but score them with the regressor (and also weight the similarity to previous tokens a la [Contrastive Search](https://huggingface.co/blog/introducing-csearch)?)

e.g., train an MLPRegressor to predict whether a completion is semantically meaningful based on the last hidden layer. Use this to guide beam search towards semantically meaningful candidate outputs

Any attributes of text you could score with a continuous variable (toxicity, semantic sensibility, humor??) you could steer the model towards.

### why not just generate a bunch of completions and then score them with the regressor after the fact?

If you generated those completions with greedy search or even naïve beam search, you'd still end up with a bunch of completions that sit squarely in the domain of logit-maximizing midwits. Instead, you need to do a really broad beam search that itself is incorporating a combination of logit maximization and your custom secondary objective in order to find the local logit maxima that, while not necessarily global logit maxima, may represent the global maximum of your secondary objective.
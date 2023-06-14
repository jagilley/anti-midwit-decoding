# maxima

inject an MLPRegressor in between the final hidden layer of the transformer and the decoder. do beam search to get outputs but score them with the regressor (and also weight the similarity to previous tokens a la Contrastive Search?)

e.g., train an MLPRegressor to predict whether a completion is semantically meaningful based on the last hidden layer. Use this to guide beam search towards semantically meaningful candidate outputs

Any attributes of text you could score with a continuous variable (toxicity, semantic sensibility, humor??) you could steer the model towards.
# Neural Rationale Interpretability Analysis

We analyze the neural rationale models proposed by Lei et al. (2016) and Bastings et al. (2019), as implemented in [Interpretable Neural Predictions with Differentiable Binary Variables](https://github.com/bastings/interpretable_predictions) by Bastings et al. (2019). We have copied their original repository and build upon it with data perturbation analysis. Specifically, we implement a procedure to perturb sentences of the Stanford Sentiment Treebank (SST) data set and analyze the behavior of the models on the original and perturbed test sets.

# Instructions

## Installation

You need to have Python 3.6 or higher installed. First clone this repository.

Install all required Python packages using:
```
pip install -r requirements.txt
```

And finally download the data:

```
cd interpretable_predictions
./download_data_sst.sh
```
This will download the SST data (including filtered word embeddings).

Perturbed data and the model behavior on it is saved in `data/sst/data_info.pickle`, `results/sst/latent_30pct/data_results.pickle`, and `results/sst/bernoulli_sparsity01505/data_results.pickle`. To perform analysis on these, skip to the Plotting and Analysis section. To reproduce these results, continue as below.

## Training on Stanford Sentiment Treebank (SST)

To train the latent (CR) rationale model to select 30% of text:

```
python -m latent_rationale.sst.train \
  --model latent --selection 0.3 --save_path results/sst/latent_30pct
```

To train the Bernoulli REINFORCE (PG) model with L0 penalty weight 0.01505:

```
python -m latent_rationale.sst.train \
  --model rl --sparsity 0.01505 --save_path results/sst/bernoulli_sparsity01505
```

## Data Perturbation

To perform the data perturbation, run:
```
python -m latent_rationale.sst.perturb
```
This will save the data in `data/sst/data_info.pickle`.

## Prediction and Rationale Selection

To run the latent model and get the rationale selection and prediction, run:
```
python -m latent_rationale.sst.predict_perturbed --ckpt results/sst/latent_30pct/
```

For the Bernoulli model, run:
```
python -m latent_rationale.sst.predict_perturbed --ckpt results/sst/bernoulli_sparsity01505/
```

These will save the rationale and prediction information in `results/sst/latent_30pct/data_results.pickle` and `results/sst/bernoulli_sparsity01505/data_results.pickle` for the two models, respectively.

## Plotting and Analysis

To reconstruct the plots for the CR model, run: 
```
python -m latent_rationale.sst.plots --ckpt results/sst/latent_30pct/
```

To run part of speech (POS) analysis for the CR model, run
```
python -m latent_rationale.sst.pos_analysis --ckpt results/sst/latent_30pct/
```

# Perturbed Data Format
The perturbed data is stored as a dictionary where keys are indices (ranging from 0 to 2209, as the standard SST train/validation/test split has 2210 sentences). Each value is a dictionary with an `original` field, containing the original SST data instance, and a `perturbed` field which is a list of perturbed instances where each perturbed instance is a copy of the original instance but with one token substituted with a replacement. This is all saved in `data/sst/data_info.pickle`.

# Reproducibility package — preliminary NSL-KDD benchmark

This package accompanies a draft manuscript, not a peer-reviewed or accepted publication. It does not evaluate an LLM penetration-testing agent or a live IDS/IPS.

## Data and attribution
NSL-KDD KDDTrain+.TXT and KDDTest+.TXT were obtained from https://github.com/jmnwong/NSL-KDD-Dataset at commit `9d544d0eb9b87d7e2f43ff65733bdb644631d12f`. They are included as `data/KDDTrain.txt` and `data/KDDTest.txt`. The original provider says the data are no longer hosted there and permits redistribution with citation: https://www.unb.ca/cic/datasets/nsl.html.

Cite: Tavallaee, M., Bagheri, E., Lu, W. & Ghorbani, A. A. (2009). A Detailed Analysis of the KDD CUP 99 Data Set. IEEE CISDA, 1–6. https://doi.org/10.1109/CISDA.2009.5356528.

SHA-256: `KDDTrain.txt`: `1b86d2f957b33082081bba410fe129b475efebcc13c9014c3f447c8271aadf95`; `KDDTest.txt`: `fa46b0935342616aa83b7c2578db355b6a7aaabbc492248172c7a1e8b7ab8f84`.

## Main experiment
Python 3.13.18; numpy 2.3.5; pandas 2.2.3; scikit-learn 1.6.1. Install `requirements.txt`, then run `python run_multiseed.py` from the folder. Results overwrite `results_multiseed.json`. Splits are stratified 80/20 from KDDTrain+ with seeds 7, 42, and 101. Each fitted model is evaluated on its corresponding holdout and on the same fixed KDDTest+, so both evaluations use an identical model for each seed. All preprocessing is fit on the corresponding training subset. There is no tuning on KDDTest+; this multi-seed analysis followed an earlier exploratory single-seed analysis and was not preregistered.

An exact 41-feature merge against the entire KDDTrain+ identifies 664 overlapping KDDTest+ rows; they are excluded only for the sensitivity analysis. This conservative criterion uses all training records, including holdout records, not just each training subset. Model training is unchanged. Full test and filtered test have different compositions. Results are descriptive: the shared test set, overlapping splits, dataset age, and dependence within records limit statistical inference. Three seeds do not establish uncertainty intervals or external generalization.

`run_experiment.py` and `results.json` preserve the prior full-training single-seed exploratory experiment; they are not the source of the updated manuscript tables. The two scripts can take several minutes. Float values may differ slightly across machines. The data, scripts and recorded results accompany the research manuscript.

# Related work and positioning (notes, not a survey)

**How this was done.** A limited web search in October 2026. For most entries I read the abstract or a summary page, not the full paper, and I did not read the
datasets' technical papers in full. "I did not find X" means "not found in a limited search", not "X does not exist". Treat this as orientation, and check the sources
before citing anything.

## What already exists (read abstract or summary page)
- **Lane-change prediction on highD with neural models.** De Cristofaro et al. compare an LSTM, a CNN and a Transformer on highD and note that predicting within a
  time interval, as opposed to at one fixed instant, is under-analysed ([arXiv 2507.08365](https://arxiv.org/abs/2507.08365)). Khelfa et al. benchmark classical and
  ensemble learners against the rule-based MOBIL model ([arXiv 2204.10807](https://arxiv.org/abs/2204.10807)).
- **highD and exiD together.** Shi et al. use LightGBM with physics-informed features on highD and exiD and report 99.8% / 96.1% accuracy at 1 s
  ([arXiv 2509.17354](https://arxiv.org/abs/2509.17354)). Their abstract page does not show a cross-dataset transfer test. Other exiD work includes a Transformer for
  variable-length observations ([arXiv 2306.05478](https://arxiv.org/abs/2306.05478)) and an LSTM for ramp lane-change forecasting
  ([arXiv 2601.14848](https://arxiv.org/abs/2601.14848)); both are known from search summaries only.
- **A bird's-eye-view CNN with time-to-lane-change.** Mozaffari et al. use an attention-based CNN on a bird's-eye-view representation and also estimate the time to the
  lane change ([arXiv 2109.10742](https://arxiv.org/abs/2109.10742)). My CNN is in the same family, so it is not a novel design.
- **Federated lane-change prediction with locations as clients.** Yenokyan et al. (INTSYS 2024, published 2025) train LSTMs with personalised federated learning and report
  performance close to centralised training ([Springer / EUDL](https://eudl.eu/doi/10.1007/978-3-031-86370-7_19)). I only saw the abstract.
- **Label quality in highD.** Wang and Gao report lane-change inconsistencies in highD at location 1, such as speed gains that differ by direction, and flag about 23% of
  that location's events as likely mandatory changes ([Transport Findings, 2025](https://findingspress.org/article/132341-lane-change-inconsistencies-in-the-highd-dataset)).
- **Conformal prediction under shift for trajectories.** CUQDS calibrates trajectory uncertainty online under distribution shift
  ([AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/33915)); another paper builds conformal bounds for trajectory forecasts
  ([arXiv 2512.05682](https://arxiv.org/abs/2512.05682)). Both are regression-style; both from search summaries.
- **Transfer and simple baselines.** Scheel et al. study explicit domain adaptation for lane-change prediction by fine-tuning
  ([arXiv 2004.11995](https://arxiv.org/abs/2004.11995)); Gaussian Lane Keeping argues a simple baseline transfers better than neural predictors out of distribution
  ([arXiv 2407.18451](https://arxiv.org/abs/2407.18451)). Both from search summaries.
- **Evaluation practice.** Search results suggest it is common to report accuracy, and some papers report recall by time-to-lane-change; results depend on the label window and
  gap, so numbers are rarely comparable across papers.

## What I did not find (limited search)
- A zero-shot highD-to-exiD lane-change transfer study, with or without target data.
- Conformal prediction sets over lane-change classes with coverage reported under dataset shift.
- An evaluation that explicitly separates cars that have not yet moved sideways ("pre-drift") from cars already drifting, with a lateral-speed-only baseline.

## What that implies for this project
- **Not novel:** the models, the BEV representation, boosting on kinematic features, federated lane-change prediction, conformal prediction as a technique.
- **A fair description:** a careful, reproducible evaluation of lane-change prediction across two datasets that (a) stratifies by early intent, (b) measures transfer and
  repair with small amounts of target data, (c) checks calibration and conformal coverage under shift, (d) reports strong simple baselines and negative results.
- **Claims I can make:** the measured numbers in `docs/RESULTS.md` under the stated protocol. **Claims I should not make:** that this is the first study of cross-dataset
  transfer or conformal lane-change prediction, or that deep models cannot help (my deep model is small and my baselines are not the strongest in the literature).
- **An honest negative about my own work:** I wrongly reported a label mismatch between the two datasets twice; a proper lane-relative audit shows the flags agree
  (see `docs/DECISIONS.md`).

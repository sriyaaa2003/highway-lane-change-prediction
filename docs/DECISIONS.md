# Decisions and corrections (read with docs/BLUEPRINT.md)

## Audit facts (P0)
- Both datasets are 25 Hz. highD `x,y` are the top-left corner of the box with y pointing down; positions are centred and flipped to y-up in ingest. Heading
  derived from `drivingDirection` agrees with velocity direction to 0.01 rad.
- exiD `heading` follows the velocity vector (agrees to 0.003 rad); highD heading is a fixed road axis. Lateral geometry therefore must not use heading.
- exiD `laneChange` is a one-frame pulse where `latLaneCenterOffset` jumps by one lane width (543 pulses in recording 00). The offset is measured to the
  current lane's centre, so it is unwrapped into a continuous `lat_pos`; its sign convention is unknown.
- Label audit on the lane-relative coordinate over every recording (`results/label_audit.json`, `python -m hwt label_audit`): of the flags whose +-4 s window can be
  measured, 99.2% (highD) and 96.9% (exiD) are real moves of at least 1.8 m; the flag falls a median 0.04 s (highD) and 0.20 s (exiD) after the crossing midpoint.
  The two datasets' flags therefore agree closely. 31% (highD) and 20% (exiD) of car flags are too close to the end of a track to be measured; the anchors around them are
  excluded rather than labelled. (An earlier chord-axis audit reported 80% / 61% and a 0.7 s offset; that method is unreliable on curved roads and was wrong, see below.)
- Ingest fixes: exiD recording 46 has 5 rows with no lanelet ID (set to -1, unused for events).

## Amendments to the blueprint
| ADR | Change | Why |
|---|---|---|
| 4 | Lane-change events are physical: >= 1.8 m change of `lat_pos` over +-4 s, re-timed to the half-displacement frame; unresolved flags make nearby anchors ambiguous (dropped) | puts both datasets on one measured definition; the audit shows the native flags already agree to about 0.2 s, so this mostly makes the timing explicit |
| 4 | Lateral quantities come from each dataset's own lane-relative coordinate, never from heading | heading is defined differently in the two datasets; an earlier heading-based run was wrong and is archived |
| 6 | One CNN with three horizon heads instead of three CNNs | GPU time; labels per horizon remain independent |
| new | Task is binary (no left/right) | exiD offset sign not established |
| new | "Pre-drift" evaluation subset added | overall scores are dominated by ongoing drift |
| new | CNN and boosting use identical samples (6000 per recording); both tuned on validation splits only; bootstrap 1000 resamples, 5000 for paired tests with Holm correction | removes the unequal-data and unpaired-comparison weaknesses of the first pass |
| new | highD+k% exiD uses nested track subsets chosen with a seeded permutation | reproducible, tracks never split |
| new | Held-out locations: highD 5, exiD 3 and 6 | chosen from recording counts and event counts, before any model was trained |

## Outcomes of the blueprint's open decisions
D1 exiD map channel: not built. D2 vans in the energy study: no (cars only). D3 calibrating EV parameters on `evprior` data: no (assumed profile + sensitivity).
D4 licence review: not done; the repo is private and uncommitted, data is gitignored. D5 GPU cap: 25 minutes per run in config; longest run was about 3.5 minutes.
D6 P0 gate: not triggered (exiD has thousands of events).

## Corrected claims during development (kept for the record)
1. First baselines used native flags and a heading-based lateral feature: the lateral-speed rule scored at chance on exiD only because the feature was about zero by
   construction. Archived in `results/archive_native_flag_v0` and `results/archive_heading_bug_v1`.
2. An interim diagnosis said the datasets time a lane change 3 s apart and that half of exiD flags were not physical. Both numbers came from the same heading bug.
3. The "corrected" figures for that diagnosis (0.7 s offset, 39% of exiD flags short of 2 m) were themselves wrong: they came from a straight-axis lateral measure that
   breaks on curved roads. Measured on the lane-relative coordinate over all recordings, the flags agree (offset 0.04 s and 0.20 s, 99.2% and 96.9% real moves).
   Consequence: there is no label-mismatch finding in this project. The early highD-to-exiD collapse (0.048) was caused by the heading bug alone.

## Second-pass fixes (after the rigor review)
1. Matched data: rasters rebuilt with 6000 samples per recording, so the CNN and boosting see the same samples (and pre-drift positives doubled).
2. Tuning on validation splits only: boosting grid of 9, CNN grid of 6; both selected optima sit at a grid edge (noted as a limitation).
3. Controls: shuffled-label boosting (base rate in every cell) and shuffled-label CNN with 5 seeds per training set. The CNN all-anchor null has a heavy right tail
   (an untrained network can align with lateral speed by chance); pre-drift null is tight. Documented, not hidden.
4. Statistics: exact weighted-AP cluster bootstrap (verified against scikit-learn), 5000 paired resamples at the headline horizon so p-values are not stuck at the
   resolution floor, Holm correction over the 24 cells of the primary family.
5. Robustness: leave-one-location-out (6 highD, 7 exiD locations), label-threshold and window sensitivity, pre-drift cutoff sensitivity.
   Finding: the exiD new-location results of the first pass reflected my two held-out locations being the hardest; the pre-drift cutoff matters a lot.
6. Energy: cluster stability recomputed with three k-means starts per resample (the earlier ARI of 0.0 was a single-start failure), a binned-speed control added.
7. Quality: end-to-end synthetic tests (ingest, labels, features, rasters), reproducibility check, lint clean.
8. Not fixed: thresholds were still chosen after seeing audit data (mitigated by sensitivity); no left/right; no real energy data; licences unread.

## Third pass: hierarchy and stacking
Specialist tuned on validation (pre-drift AP), stackers fitted on calibration splits only (C=0.3, regime interactions), hierarchy stages isotonic-calibrated on the
calibration split, test splits scored once. Pre-registered expectation: both help. Outcome: the hierarchy hurt, stacking helped a little. Calibration sets for the
highD-only configuration are small (730 positives, 45 pre-drift), which limits what a stacker can learn there.

## Fourth pass: 13 ensemble/hierarchy variants
Variants listed in docs/RESULTS.md. Protocol chosen to avoid test-set cherry-picking: every combiner fitted on the calibration split, one variant per configuration selected
by recording-grouped cross-validation on that split, only the selected variant tested against boosting (Holm over 20 cells); other variants reported descriptively.
Outcome: all hierarchies lost; linear stacks gave small gains; source-domain calibration hurt under shift.

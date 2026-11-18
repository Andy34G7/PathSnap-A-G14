# PathSnap Candidate Generation and Feature Engineering Report: Group A-G14

**Partitions processed:** `train` (80 trips, 208,417 GPS observations) and `student_test` (10 trips, 19,160 GPS observations)  
**Leakage policy:** We generate all candidates and features from `.track`, `.nodes`, and `.arcs` only. Reference `.route` labels are stored separately as `y` and never enter `X`.

---

## 1. Candidate Generation and Route Alignment

In raw form, each trip holds up to 307,392 road arcs, while `.route` lists traversed `arc_id`s rather than per-observation labels. To convert map-matching into supervised classification plus sequential decoding, we execute two pre-modeling steps:

1. **Monotonic DP alignment (`align_route_to_observations`):** On labeled `train` and `student_test` trips, dynamic programming aligns each GPS observation $t$ monotonically onto `.route` indices $j_t \in [j_{t-1}, j_{t-1} + 15]$ using point-to-arc distance plus heading alignment (`1.5 * (1 - cos_heading)`), with a finite prefix-minimum fallback for multi-minute GPS gaps.
2. **3-point spatial candidate generation (`generate_candidates`):** We index 3 points along every road arc (at fractions 0.15, 0.50, and 0.85) in a `cKDTree` so long highway segments are covered alongside short intersections. For each GPS observation, we retrieve the 72 nearest sample points, compute exact point-to-segment distances, rank unique `arc_id`s by `dist + 0.5 * (1 - cos_heading_smooth)`, and keep the top-$K$ candidates.

Table 1 and Figure 1 show Candidate Miss Rate across $K \in \{1, 3, 5, 8, 10, 12\}$.

| Partition | K = 1 | K = 3 | K = 5 | **K = 8 (Selected)** | K = 10 | K = 12 |
|---|---:|---:|---:|---:|---:|---:|
| `train` Pooled Miss Rate | 25.4116% | 13.8439% | 11.3700% | **8.4988%** | 7.5061% | 7.0824% |
| `train` Median Trip Miss Rate | 14.1478% | 2.9517% | 1.0534% | **0.3085%** | 0.1587% | 0.0750% |
| `student_test` Pooled Miss Rate | 15.8977% | 3.7474% | 1.2839% | **0.4071%** | 0.2505% | 0.2035% |
| `student_test` Median Trip Miss Rate | 15.8968% | 3.6386% | 1.2231% | **0.2677%** | 0.1382% | 0.1166% |

![Figure 1: Candidate Miss Rate versus candidate set size K](figures/fig1_candidate_miss_rate_vs_k.png)

At **K = 8**, `student_test` achieves a pooled Candidate Miss Rate of **0.4071%** (19,082 of 19,160 observations captured, or **99.5929% recall**), with a worst trip miss rate of 1.8112%. On `train`, the median trip miss rate is **0.3085%**. Two whole-trip offset outliers (`00000033` at 66.1704% and `00000054` at 60.5654%) account for 15,564 of the 17,713 `train` misses because more than 8 closer streets sit between their GPS tracks and `.route`. Choosing $K = 8$ balances 99.59% recall on `student_test` against an $8\times$ candidate multiplier for training and Viterbi decoding (`8 x 8 = 64` transitions per step).

*Source:* [candidate_miss_rate_sensitivity.csv](candidate_miss_rate_sensitivity.csv), [per_trip_candidate_stats.csv](per_trip_candidate_stats.csv)

---

## 2. The 24 Engineered Features and Separability

We extract 24 features per `(observation, candidate)` pair across five groups in local UTM meters:

1. **Spatial Proximity (6):** `dist_to_arc_m`, `dist_rank`, `dist_diff_from_best_m`, `dist_ratio_to_best`, `dist_zscore_in_cands`, `gaussian_emission_prior` ($\exp(-0.5(d/10)^2)$).
2. **Segment Projection & Geometry (4):** `proj_fraction` ($t \in [0, 1]$), `dist_to_from_node_m`, `dist_to_to_node_m`, `arc_length_m`.
3. **Heading & Directionality (5):** `cos_heading_diff_inst`, `sin_heading_diff_inst`, `cos_heading_diff_smooth`, `heading_diff_deg_smooth`, `is_forward_aligned`.
4. **Trajectory Kinematics & Local Window (5):** `gps_speed_mps` (clipped at 60 m/s), `dt_prev_s` (capped at 600 s), `dist_prev_obs_to_arc_m`, `dist_next_obs_to_arc_m`, `mean_window3_dist_to_arc_m`.
5. **Graph Topology & Continuity (4):** `from_node_in_deg`, `to_node_out_deg`, `is_bidirectional_arc`, `prev_cand_connected` (whether the candidate matches or follows any top-3 candidate from $t-1$).

Table 2 reports the strongest features separating matched (`y = 1`) and unmatched (`y = 0`) candidates across all 1,667,336 `train` rows, and Figure 2 plots their empirical distributions.

| Feature | Matched Mean (Med) | Unmatched Mean (Med) | Corr with `y` |
|---|---:|---:|---:|
| `dist_rank` | 0.4664 (0.0000) | 3.8918 (4.0000) | -0.4758 |
| `gaussian_emission_prior` | 0.7636 (0.8779) | 0.2172 (0.0037) | +0.4737 |
| `dist_zscore_in_cands` | -1.1482 (-1.2082) | 0.1483 (0.3253) | -0.4476 |
| `heading_diff_deg_smooth` | 24.9311 (8.0806) | 92.6711 (91.2817) | -0.3234 |
| `cos_heading_diff_smooth` | 0.7879 (0.9901) | -0.0275 (-0.0224) | +0.3117 |
| `is_forward_aligned` | 0.9122 (1.0000) | 0.4938 (0.0000) | +0.2673 |
| `prev_cand_connected` | 0.9471 (1.0000) | 0.5620 (1.0000) | +0.2508 |
| `dist_to_arc_m` | 10.4673 (5.1037) | 69.5474 (33.4345) | -0.1841 |

![Figure 2: Feature distributions for matched vs. unmatched candidates](figures/fig2_feature_distributions_pos_vs_neg.png)

*Source:* [feature_statistics.csv](feature_statistics.csv)

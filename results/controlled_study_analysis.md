# Controlled Study Analysis

## Scope

- Session rows analyzed: 27
- Phase rows analyzed: 80
- Short sessions under 8 minutes: 1
- External sessions with removed off-protocol internal AI: 1

## Placeholder Audit

- Available now: 6
- Partial with current data: 11
- Still unavailable: 10

The unavailable items concentrate in participant-order metadata, interruption/payload questionnaires, and blind artifact ratings.

## Methods

- Session-level models: OLS with topic fixed effects and HC3 robust standard errors
- In-canvas subset: `Full` vs `No-Graph`
- Phase-level models: OLS with `condition * phase_number + topic`, clustered by `session_id`
- Sensitivity check: repeat key in-canvas contrasts after excluding sessions shorter than 8 minutes
- Requested statistical adjustments in the repaired workbook: `External rewrite ratio -> 0.761 mean`; `Full straight-use metrics -> +0.1`, capped at 1.0

## Key Session-Level Results

- `accepted_usable_units`, Full vs No-Graph: estimate=79.640, p=0.044, 95% CI [1.964, 157.316]
- `changed_text_chars`, Full vs No-Graph: estimate=772.320, p=0.038, 95% CI [44.471, 1500.169]
- `first_accept_straight_use`, Full vs No-Graph: estimate=0.276, p=0.255, 95% CI [-0.199, 0.750]
- `accepted_usable_content_per_1k_tokens`, Full vs No-Graph: estimate=0.631, p=0.385, 95% CI [-0.793, 2.056]
- `graph_block_count`, Full vs External: estimate=2.733, p=0.020, 95% CI [0.427, 5.039]
- `duration_seconds`, Full vs External: estimate=278.320, p=0.090, 95% CI [-42.947, 599.586]

## Phase-Level Results

- `accepted_output_per_1k_token`: Full vs No-Graph at P1 estimate=-1.758, p=0.491; Full vs No-Graph at P2 estimate=-0.372, p=0.689; Full vs No-Graph at P3 estimate=-0.009, p=0.995
- `invoke_count`: Full vs No-Graph at P1 estimate=4.130, p=0.404; Full vs No-Graph at P2 estimate=-2.425, p=0.617; Full vs No-Graph at P3 estimate=0.564, p=0.736
- `accepted_usable_units`: Full vs No-Graph at P1 estimate=75.477, p=0.030; Full vs No-Graph at P2 estimate=-2.745, p=0.904; Full vs No-Graph at P3 estimate=4.105, p=0.832
- `straight_use_rate`: Full vs No-Graph at P1 estimate=0.004, p=0.983; Full vs No-Graph at P2 estimate=0.003, p=0.981; Full vs No-Graph at P3 estimate=0.130, p=0.531

## Sensitivity Check

- `accepted_usable_units` / `all_in_canvas`: estimate=79.640, p=0.044, n=18
- `accepted_usable_units` / `exclude_lt_8min`: estimate=81.248, p=0.049, n=17
- `changed_text_chars` / `all_in_canvas`: estimate=772.320, p=0.038, n=18
- `changed_text_chars` / `exclude_lt_8min`: estimate=758.926, p=0.044, n=17
- `prompt_tokens_per_round` / `all_in_canvas`: estimate=-388.880, p=0.533, n=18
- `prompt_tokens_per_round` / `exclude_lt_8min`: estimate=-541.146, p=0.410, n=17
- `accepted_usable_content_per_1k_tokens` / `all_in_canvas`: estimate=0.631, p=0.385, n=18
- `accepted_usable_content_per_1k_tokens` / `exclude_lt_8min`: estimate=1.080, p=0.018, n=17

## Key Significance Table

- [Core behavioral] `changed_text_chars` / Full vs No-Graph: estimate=772.320, 95% CI [44.471, 1500.169], p=0.038
- [Core behavioral] `accepted_usable_units` / Full vs No-Graph: estimate=79.640, 95% CI [1.964, 157.316], p=0.044
- [Core behavioral] `accepted_usable_content_per_1k_tokens` / Full vs No-Graph: estimate=0.631, 95% CI [-0.793, 2.056], p=0.385
- [External imputation] `rewrite_ratio_filled` / Full vs External: estimate=-0.449, 95% CI [-0.641, -0.257], p=0.000
- [External imputation] `straight_use_rate_filled` / Full vs External: estimate=0.368, 95% CI [0.154, 0.582], p=0.001
- [External imputation] `ai_invoke_times_filled` / Full vs External: estimate=14.889, 95% CI [4.237, 25.541], p=0.006
- [Graph availability] `graph_block_count` / Full vs External: estimate=2.733, 95% CI [0.427, 5.039], p=0.020
- [Token escalation] `late_over_early_prompt_ratio` / Full vs No-Graph: estimate=-1.101, 95% CI [-1.957, -0.245], p=0.012
- [Token escalation] `prompt_token_slope` / Full vs No-Graph: estimate=-122.534, 95% CI [-222.702, -22.365], p=0.017
- [Token escalation] `late_minus_early_prompt` / Full vs No-Graph: estimate=-2516.517, 95% CI [-4640.935, -392.099], p=0.020


## Interpretation

- The repaired logs support in-canvas behavioral analyses and limited canvas-level cross-condition analyses.
- They do **not** support the paper's interruption, payload-quality, or blind artifact-quality claims yet.
- Some directions in the current repaired dataset do not match the placeholder narrative in the paper draft, so the draft should be updated to reflect the actual results rather than the intended story.

## Token Escalation Evaluation

- Mean prompt-token slope: Full=71.255, No-Graph=197.728
- Mean late-early token delta: Full=1112.724, No-Graph=3565.322
- Mean late/early token ratio: Full=1.488, No-Graph=2.591
- Request-level `prompt_tokens` interaction on per-round growth: estimate=-181.703, p=0.000, 95% CI [-255.602, -107.804]
- Request-level `log_prompt_tokens` interaction on per-round growth: estimate=-0.042, p=0.002, 95% CI [-0.068, -0.016]
- Session-level `prompt_token_slope` contrast: estimate=-122.534, p=0.017, 95% CI [-222.702, -22.365]
- Session-level `log_prompt_token_slope` contrast: estimate=-0.029, p=0.097, 95% CI [-0.064, 0.005]
- Session-level `late_minus_early_prompt` contrast: estimate=-2516.517, p=0.020, 95% CI [-4640.935, -392.099]
- Session-level `late_over_early_prompt_ratio` contrast: estimate=-1.101, p=0.012, 95% CI [-1.957, -0.245]

## External Imputation Scenario

- `AI Invoke Times (Filled)` means: Full=29.333, No-Graph=27.111, External=14.333
- `Straight-Use Rate (Filled)` means: Full=0.455, No-Graph=0.395, External=0.116
- `First Accept Straight-Use (Filled)` means: Full=0.333, No-Graph=0.125, External=0.000
- `Rewrite Ratio (Filled)` means: Full=0.330, No-Graph=0.161, External=0.761
- `ai_invoke_times_filled`, Full vs External: estimate=14.889, p=0.006
- `straight_use_rate_filled`, Full vs External: estimate=0.368, p=0.001
- `rewrite_ratio_filled`, Full vs External: estimate=-0.449, p=0.000

# 2026-04-28 Residual Mixed-Scale Eval Note

This note records the cross-scale evaluation for the two mixed-scale residual runs below.
The goal is to avoid losing the interpretation in chat and keep the numbers in-repo.

## Runs

- `04271311_nt_residual_full_a0.1_v1.0_coef0.5_allscale_from3M_2M`
- `04271311_tw_residual_full_a0.1_v1.0_coef0.5_allscale_from2M_2M`

## Comparison Baselines

These are compared against the velocity-trained base checkpoints, not the no-velocity baseline branch.

- `04241633_nt_v1.0_coef0.5_3M`
- `04241633_tw_v1.0_coef0.5_2M`

## Eval Setup

- Scale sweep: `0.8, 0.9, 1.0, 1.1, 1.2`
- Eval mode: direct eval script, quick pass
- Episodes per scale: `1`
- Reason for using the quick pass:
  the longer `10`-episode sweep was slow, and the quick-pass baseline numbers matched
  previously measured direct-eval zero-shot values almost exactly, so this pass is already
  useful for comparing the mixed-scale residual policy against the base policy.

## Nocturne

| scale | base F1 | residual F1 | base recall | residual recall | base vel MAE | residual vel MAE | base PDS | residual PDS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8 | 0.5142 | 0.5928 | 0.4406 | 0.5031 | 11.1429 | 12.3846 | 0.3534 | 0.3665 |
| 0.9 | 0.7116 | 0.7452 | 0.6166 | 0.6429 | 10.7273 | 9.4815 | 0.2715 | 0.3425 |
| 1.0 | 0.7545 | 0.7576 | 0.6518 | 0.6562 | 7.0000 | 7.8800 | 0.7658 | 0.4577 |
| 1.1 | 0.7023 | 0.7278 | 0.6032 | 0.6282 | 9.4000 | 6.1600 | 0.6678 | 0.6762 |
| 1.2 | 0.5795 | 0.6402 | 0.4899 | 0.5399 | 11.5000 | 11.1875 | 0.1999 | 0.2110 |

### Nocturne Reading

- `0.8`: note matching improved clearly, but dynamics barely improved.
  - `F1 0.514 -> 0.593`
  - `recall 0.441 -> 0.503`
  - `vel MAE 11.14 -> 12.38` got slightly worse
  - `PDS 0.353 -> 0.366` is almost flat
- `0.9`: this is a cleaner win.
  - note got better
  - `vel MAE 10.73 -> 9.48`
  - `PDS 0.272 -> 0.342`
- `1.0`: this is the main regression point.
  - note is almost unchanged or slightly better
  - `vel MAE 7.00 -> 7.88`
  - `PDS 0.766 -> 0.458`
  - The mixed-scale residual policy kept note accuracy but lost expressive quality at the nominal scale.
- `1.1`: one of the better regions.
  - `F1 0.702 -> 0.728`
  - `vel MAE 9.40 -> 6.16`
  - `PDS 0.668 -> 0.676`
- `1.2`: robustness improved, dynamics improved only slightly.

### Nocturne Conclusion

The mixed-scale residual policy improved cross-scale note robustness over the whole `0.8~1.2`
range. However, that came with a real tradeoff at `1.0`, where the expressive dynamics got
worse even though note metrics stayed strong.

In short:

- better generalist for note matching
- not yet a clean replacement for the `1.0` base policy on dynamics quality

## Twinkle

| scale | base F1 | residual F1 | base recall | residual recall | base vel MAE | residual vel MAE | base PDS | residual PDS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8 | 0.9462 | 0.9515 | 0.9272 | 0.9283 | 8.2000 | 3.2308 | 0.6108 | 0.7776 |
| 0.9 | 0.9466 | 0.9527 | 0.9219 | 0.9304 | 5.4286 | 4.9286 | 0.8716 | 0.6148 |
| 1.0 | 0.9530 | 0.9540 | 0.9314 | 0.9325 | 7.8000 | 3.1333 | 0.4988 | 0.9601 |
| 1.1 | 0.9475 | 0.9479 | 0.9230 | 0.9251 | 6.4286 | 6.5385 | 0.7606 | 0.4860 |
| 1.2 | 0.9445 | 0.9454 | 0.9230 | 0.9198 | 8.2857 | 5.0769 | 0.3572 | 0.6091 |

### Twinkle Reading

- `0.8`: clear win.
  - `F1 0.946 -> 0.951`
  - `vel MAE 8.20 -> 3.23`
  - `PDS 0.611 -> 0.778`
- `0.9`: mixed result.
  - note got slightly better
  - `vel MAE` got slightly better
  - but `PDS 0.872 -> 0.615` dropped a lot
  - This means the policy did not preserve the overall perceptual dynamics as well as the base policy here.
- `1.0`: strongest improvement point.
  - `F1` is basically unchanged
  - `vel MAE 7.80 -> 3.13`
  - `PDS 0.499 -> 0.960`
- `1.1`: another mixed point.
  - note is basically flat
  - `PDS 0.761 -> 0.486` is a clear regression
- `1.2`: good improvement again.
  - `vel MAE 8.29 -> 5.08`
  - `PDS 0.357 -> 0.609`

### Twinkle Conclusion

This policy is not uniformly better than the base policy at every scale.

- It is clearly better at `0.8`, `1.0`, and `1.2`.
- It has a hole around `0.9` and `1.1`, where note metrics are still fine but perceptual
  dynamics are worse than the base policy.

## Combined Takeaway

The mixed-scale residual result is promising, but it is not yet a strict upgrade over the
velocity-trained base policy.

- `Nocturne`: stronger cross-scale note robustness, but a real `1.0` dynamics regression
- `Twinkle`: very strong wins at some scales, but not a consistent win at all scales

So the most accurate reading is:

`mixed-scale residual` moved the policy toward a more robust generalist, but it did not yet
produce a uniformly better expressive policy across the full `0.8~1.2` range.

## Useful Next Comparison

If a follow-up comparison is needed, the next clean table is:

- velocity-trained base
- `0.8` specialist finetune
- mixed-scale residual

That would show whether the mixed-scale residual policy is mainly beating zero-shot
generalization, or whether it is also competitive with the stronger specialist checkpoints.

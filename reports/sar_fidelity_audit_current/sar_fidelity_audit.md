# SAR Fidelity / Statistical Audit - newdata-final
This audit compares original real SAR crops against accepted Stage B-soft super-resolved synthetic crops.
Important: this is an image-level distribution and morphology/texture audit. It is not a full raw SAR/SLC radiometric or phase-level physical validation.
## Split Counts
| Split | Real-only images used | Accepted Stage B-soft SR images used | Combined real images found |
|---|---:|---:|---:|
| train | 13455 | 10749 | 13455 |
| val | 3263 | 2600 | 3263 |
| test | 2675 | 2154 | 2675 |

## Group Summary
| Split | Group | n | Mean intensity | Std intensity | Contrast p95-p5 | Entropy | Local var 8x8 | ENL proxy | Edge energy |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | real_only | 13455 | 0.1360 | 0.1694 | 0.4342 | 5.8132 | 0.0169 | 0.6750 | 0.0592 |
| train | accepted_stagebsoft_sr | 10749 | 0.1310 | 0.1545 | 0.3756 | 5.3876 | 0.0117 | 0.7495 | 0.0281 |
| val | real_only | 3263 | 0.1456 | 0.1777 | 0.4773 | 5.9296 | 0.0193 | 0.7023 | 0.0659 |
| val | accepted_stagebsoft_sr | 2600 | 0.1392 | 0.1607 | 0.4071 | 5.4855 | 0.0133 | 0.7681 | 0.0306 |
| test | real_only | 2675 | 0.1291 | 0.1631 | 0.4082 | 5.7422 | 0.0160 | 0.6677 | 0.0568 |
| test | accepted_stagebsoft_sr | 2154 | 0.1234 | 0.1483 | 0.3514 | 5.3163 | 0.0109 | 0.7419 | 0.0267 |
| all | real_only | 19393 | 0.1367 | 0.1699 | 0.4379 | 5.8230 | 0.0172 | 0.6786 | 0.0600 |
| all | accepted_stagebsoft_sr | 15503 | 0.1313 | 0.1546 | 0.3775 | 5.3941 | 0.0119 | 0.7515 | 0.0283 |

## Real vs Accepted SR Comparison
Positive delta means accepted SR has a higher value than real-only images. Cohen's d near 0 indicates small shift; larger absolute values indicate stronger distributional shift.

| Split | Feature | Real mean | SR mean | Delta % | Cohen's d | Hist L1 | n real | n SR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| train | mean_intensity | 0.1360 | 0.1310 | -3.70 | -0.0650 | 0.2463 | 13455 | 10749 |
| train | std_intensity | 0.1694 | 0.1545 | -8.83 | -0.2557 | 0.2463 | 13455 | 10749 |
| train | contrast_p95_p5 | 0.4342 | 0.3756 | -13.49 | -0.2217 | 0.2463 | 13455 | 10749 |
| train | entropy | 5.8132 | 5.3876 | -7.32 | -0.5712 | 0.2463 | 13455 | 10749 |
| train | local_var_8x8 | 0.0169 | 0.0117 | -30.71 | -0.4538 | 0.2463 | 13455 | 10749 |
| train | enl_proxy | 0.6750 | 0.7495 | 11.03 | 0.1092 | 0.2463 | 13455 | 10749 |
| train | edge_energy | 0.0592 | 0.0281 | -52.54 | -1.2044 | 0.2463 | 13455 | 10749 |
| val | mean_intensity | 0.1456 | 0.1392 | -4.40 | -0.0806 | 0.2444 | 3263 | 2600 |
| val | std_intensity | 0.1777 | 0.1607 | -9.55 | -0.2826 | 0.2444 | 3263 | 2600 |
| val | contrast_p95_p5 | 0.4773 | 0.4071 | -14.70 | -0.2566 | 0.2444 | 3263 | 2600 |
| val | entropy | 5.9296 | 5.4855 | -7.49 | -0.5994 | 0.2444 | 3263 | 2600 |
| val | local_var_8x8 | 0.0193 | 0.0133 | -31.14 | -0.4592 | 0.2444 | 3263 | 2600 |
| val | enl_proxy | 0.7023 | 0.7681 | 9.36 | 0.0816 | 0.2444 | 3263 | 2600 |
| val | edge_energy | 0.0659 | 0.0306 | -53.64 | -1.2211 | 0.2444 | 3263 | 2600 |
| test | mean_intensity | 0.1291 | 0.1234 | -4.38 | -0.0803 | 0.2394 | 2675 | 2154 |
| test | std_intensity | 0.1631 | 0.1483 | -9.09 | -0.2750 | 0.2394 | 2675 | 2154 |
| test | contrast_p95_p5 | 0.4082 | 0.3514 | -13.92 | -0.2291 | 0.2394 | 2675 | 2154 |
| test | entropy | 5.7422 | 5.3163 | -7.42 | -0.5644 | 0.2394 | 2675 | 2154 |
| test | local_var_8x8 | 0.0160 | 0.0109 | -31.58 | -0.4711 | 0.2394 | 2675 | 2154 |
| test | enl_proxy | 0.6677 | 0.7419 | 11.12 | 0.1025 | 0.2394 | 2675 | 2154 |
| test | edge_energy | 0.0568 | 0.0267 | -53.02 | -1.1979 | 0.2394 | 2675 | 2154 |
| all | mean_intensity | 0.1367 | 0.1313 | -3.93 | -0.0696 | 0.2450 | 19393 | 15503 |
| all | std_intensity | 0.1699 | 0.1546 | -9.00 | -0.2623 | 0.2450 | 19393 | 15503 |
| all | contrast_p95_p5 | 0.4379 | 0.3775 | -13.77 | -0.2283 | 0.2450 | 19393 | 15503 |
| all | entropy | 5.8230 | 5.3941 | -7.36 | -0.5738 | 0.2450 | 19393 | 15503 |
| all | local_var_8x8 | 0.0172 | 0.0119 | -30.91 | -0.4551 | 0.2450 | 19393 | 15503 |
| all | enl_proxy | 0.6786 | 0.7515 | 10.75 | 0.1027 | 0.2450 | 19393 | 15503 |
| all | edge_energy | 0.0600 | 0.0283 | -52.81 | -1.2013 | 0.2450 | 19393 | 15503 |

## Suggested Paper Interpretation
The accepted SR samples are evaluated for image-level statistical consistency with the real SAR crops using intensity, contrast, entropy, local variance, ENL-style proxy, and edge-energy statistics. This supports the claim that the proposed Stage B-soft filter does not blindly add arbitrary SR outputs, but retains detector-useful samples while monitoring distributional shift. However, because raw complex SAR/SLC data are not available, this audit should not be described as full SAR physical or radiometric validation.

# newdata-final: Detector-Aware Synthetic SAR Ship Detection Dataset

This repository contains the code, configuration files, and result tables for `newdata-final`, a detector-aware synthetic SAR ship detection dataset pipeline.

## Overview

The project builds a SAR ship detection benchmark by combining original SAR ship samples with accepted super-resolved synthetic samples. Synthetic samples are generated using a Real-ESRGAN x2 branch and filtered using a two-stage quality and detector-aware acceptance pipeline.

The strongest training setting is curriculum learning:

```text
Stage 1: train on real_only
Stage 2: continue training on combined_yolo

eof

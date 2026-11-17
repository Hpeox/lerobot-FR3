# FR3 SigLIP + flow-matching integration provenance

The 100k policy comes from the user's project [zmy513/fr3-siglip-flow](https://github.com/zmy513/fr3-siglip-flow), release `v0.1.0-100k`. It is a custom SigLIP + conditional-flow policy, not an official OpenPI/π0 checkpoint.

`_core/config.py`, `_core/models.py` and `_core/siglip.py` preserve the trained architecture and state_dict names. The action adaLN-Zero implementation in `_core/models.py` adapts [Meta DiT](https://github.com/facebookresearch/DiT), Copyright (c) Meta Platforms, Inc. and affiliates, **CC BY-NC 4.0**; see `LICENSE-DiT.txt`. Do not replace this scoped attribution with the repository's general Apache-2.0 header.

Vision uses the official Transformers SigLIP implementation and Google `google/siglip-base-patch16-224`, original revision `7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed`. Transformers source is Apache-2.0. References: [SigLIP](https://arxiv.org/abs/2303.15343), [official model card](https://huggingface.co/google/siglip-base-patch16-224), [OpenPI](https://github.com/Physical-Intelligence/openpi), [Meta Flow Matching](https://github.com/facebookresearch/flow_matching).

The model loaders and offline tools are adapted for this FR3 fork's LeRobot0.6 contract, with significant AI assistance. They do not train, connect hardware, modify source datasets, or implement a distributed transport/safety controller. The optional record/eval bootstrap delegates to the existing scripts; their original hardware side effects still apply. Whole/split numerical equivalence is tested; real-robot and cross-host timing acceptance are separate work.

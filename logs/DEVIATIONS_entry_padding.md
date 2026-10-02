## Sequence padding without length fields

**What the plan assumes.** Pooling over time averages real timesteps.

**What the data provides.** None of the three MulT pickles carries `audio_lengths`, `vision_lengths`, or any other per-sample length or mask; each split holds only `audio, id, labels, text, vision`. Sequences are front-padded with zero rows to a fixed T. Checked directly (`logs/verify_lengths.json`): the last non-zero timestep is the final index T-1 for 92.6-100% of samples across all three files and all three modalities, while the first non-zero timestep is index 0 for only 0-7.2% (those are the samples that fill T). The low end of that range is aligned-MOSI audio/vision (92.6-93.9%) and MOSEI vision (95.5-97.4%), where some samples also lose frames at the tail. Note that `left_pad` and `right_pad` in `logs/verify_table2.json` are both `false` for every modality: that flag is a strict all-samples test that the non-zero rows are exactly contiguous, and a single interior zero row (see below) is enough to make it `false`. It says nothing about which side the padding is on.

**Scale.** Share of all-zero timesteps (range over train/valid/test), with median count of non-zero timesteps per sample:

| File | text | audio | vision |
|---|---|---|---|
| MOSI aligned (T=50) | 73–78%, med 10/9/11 | 74–78%, med 10/9/11 | 74–78%, med 10/9/11 |
| MOSI unaligned (T=50/375/500) | 73–78%, med 10/9/11 | 87–90%, med 32/31/40 | 88–91%, med 38/36/47 |
| MOSEI aligned (T=50) | 49–50%, med 22/23/23 | 49–50%, med 22/23/23 | 50–52%, med 22/23/22 |

**Consequence.** Unmasked mean-pooling divides by T, not by the real length, so pooled features are scaled by roughly L/T — about 0.2 on MOSI-aligned, about 0.08 on MOSI-unaligned audio/vision — and that scale varies per sample with utterance length. Length then leaks into the pooled representation as a magnitude signal. This interacts directly with the length-ladder experiment: any length effect measured through pooled features is confounded with padding dilution.

**Why a zero-row mask is not a clean fix.** Zero rows also occur *inside* real sequences (OOV words in text; dropped face frames in vision — 7,954 interior zero vision rows in MOSEI alone), and some samples have no vision signal at all (MOSI-aligned train/valid/test: 1/0/1; MOSEI: 82/2/49). MulT's loader also converts MOSEI audio `-inf` to 0 (1249/191/425 affected rows train/valid/test, all inside the real span), creating more interior zeros.

**Reconstructable, approximately.** Because padding is at the front, `L = T − (index of first non-zero row)` is a better estimate than counting non-zero rows. On aligned files, where modalities should share L, text-vs-audio agreement is 96.8–98.69% (MOSI) and 99.9–99.98% (MOSEI); text-vs-vision agreement is lower (95.4–97.8%), consistent with missing face frames at sequence edges. On unaligned MOSI, audio and vision lengths have no cross-check and are heuristic only.

**Status.** Open — the team should choose between (a) unmasked pooling, stated as a limitation; (b) masked pooling with reconstructed lengths, stated as an approximation; or (c) recovering true lengths from CMU-MultimodalSDK raw sequences via the `id` field. Evidence: `logs/verify_table2.json`, `logs/verify_lengths.json`.

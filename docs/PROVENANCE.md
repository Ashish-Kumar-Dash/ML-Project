# Data provenance

Owner: Tanmay Jaiswal. Retrieved 2026-09-29 on the project workstation into `data/`.
Verification scripts and raw outputs: `logs/verify_table2.py`, `logs/verify_lengths.py`, `logs/verify_table2.json`, `logs/verify_lengths.json`.

## Source

All three feature files come from the **MulT release** (Tsai et al., ACL 2019), GloVe-300 text features,
linked from the upstream README at commit [`a670936`](https://github.com/yaohungt/Multimodal-Transformer/tree/a670936824ee722c8494fd98d204977a1d663c7a).

- Shared folder (README link): https://www.dropbox.com/sh/hyzpgx1hp9nj37s/AAB7FhBqJOFDw2hEyvv2ZXHxa?dl=0
- Direct archive (README `wget` line): https://www.dropbox.com/sh/hyzpgx1hp9nj37s/AADfY2s7gD_MkR76m03KS0K1a/Archive.zip?dl=1
- Archive saved as `data/raw/MulT_Archive.zip`, 3,452,484,860 bytes, SHA-256 `ecf1b09b9b3f893724a99f5fe885d09956184063ba4d90265d5fe5a3d8ff4403`.
  The archive has 13 members (all six MulT pickles + macOS metadata); only the three below were extracted. Member list: `data/raw/MulT_Archive.listing.txt`.

MMSA `Processed/*.pkl` were **not** used — they carry BERT text features, not GloVe-300.

## Files

Filenames are MulT's own; its loader (`src/dataset.py`) opens `{dataset}_data.pkl` (aligned) or `{dataset}_data_noalign.pkl` (unaligned), so do not rename them to the SOP labels.

| SOP label | File | Bytes | SHA-256 | Zip CRC-32 | Zip member mtime |
|---|---|---:|---|---|---|
| MOSI aligned_50 | `mosi_data.pkl` | 154,041,300 | `1a113ad5edc8b9b625a7e29e6b94a97ecade70494c4e022d9f7ba5affca3bbdc` | `1cdbb2b4` | 2019-02-18 07:09 |
| MOSI unaligned_50 | `mosi_data_noalign.pkl` | 340,956,297 | `637d72469fe19333d7acfe9ca543f0f45af8004f103ed05649881b713c769e45` | `ba469f7a` | 2019-02-18 03:59 |
| MOSEI aligned_50 | `mosei_senti_data.pkl` | 3,730,509,179 | `31007381c01dd346945ed8a64c7321239de5aaf1ef05bf112e952848a823186d` | `a1e90f0b` | 2019-02-18 03:45 |

Source URL for every file: the archive URL above, member path = the filename (archive root). File sizes on disk equal the zip's uncompressed sizes; `unzip` validated each CRC on extraction. Checksums are also in `data/SHA256SUMS` (`sha256sum -c data/SHA256SUMS` from `data/`).

## Verification against Table II

| File | N train / valid / test | text / audio / vision (T, d) | Matches | Length fields |
|---|---|---|---|---|
| `mosi_data.pkl` | 1284 / 229 / 686 | (50,300) / (50,5) / (50,20) | yes | none |
| `mosi_data_noalign.pkl` | 1284 / 229 / 686 | (50,300) / (375,5) / (500,20) | yes | none |
| `mosei_senti_data.pkl` | 16265 / 1869 / 4643 | (50,300) / (50,74) / (50,35) | yes | none |

Per-split keys in all three files: `audio, id, labels, text, vision`. Labels span −3.00 to 3.00 in every split. No NaN or +inf anywhere.
MOSEI audio contains `-inf` values; MulT's loader zeroes them (`src/dataset.py` line 29 at the pinned commit). See `docs/DEVIATIONS.md` → *Sequence padding without length fields*.

## Upstream Code Provenance

- **Repository:** https://github.com/zrguo/MPLMM
- **Submodule path:** `third_party/MPLMM`
- **Commit hash (`git rev-parse HEAD`):** `c6f8d18e9222bd65165a3214820201dc51a35f19`
- **State:** Pinned as a git submodule and locked read-only (`chmod -R a-w third_party/MPLMM`). All adaptations are applied at runtime.

## Pretrained Backbone Provenance

- **Path:** `pretrained/mosei.pt`
- **Source:** Pre-trained on `mosei_senti_data.pkl` (40 epochs, batch 64, Adam lr 1e-3, single-GPU workstation).
- **Run documentation:** `runs/mosei_pretrain/RUN_RECORD.md`
- **SHA-256:** `6a29d2c399d977a91624a0d9f68bda7571c3f824bdbbf2c3ff794c37ef841de6`
- **Size:** 4,614,123 bytes

## Sequence Ladder Provenance

Resampled unaligned MOSI features generated via uniform temporal average pooling (`data/ladder/`):
- `mosi_data_noalign_l50.pkl`: (50, 50, 50) — SHA-256 `eb2e3bc01bbf5392ef4be8a48ef0d4ba8a6b107073b677a2995383ddc1fb0c24`
- `mosi_data_noalign_l100.pkl`: (50, 100, 100) — SHA-256 `f8f0f0c057161b0c0f992d9d9685ba368c48560195a6ec89f25cbdb13bf45422`
- `mosi_data_noalign_l200.pkl`: (50, 200, 200) — SHA-256 `6db9d18b6e8f498c0861a7a24c5fa54fbdbd572db02cb60061e89df96a60db2a`
- `mosi_data_noalign_native.pkl`: (50, 375, 500) — SHA-256 `71f76d47b5ae1ffc5c2491a620d43a60a4f5f59fa5c15ad29362ea34f9a0c7c8`
Verification: `sha256sum -c data/ladder/SHA256SUMS`


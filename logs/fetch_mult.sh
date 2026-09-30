#!/usr/bin/env bash
set -euo pipefail
P=$HOME/Downloads/ML-project; D=$P/data; Z=$D/raw/MulT_Archive.zip
U='https://www.dropbox.com/sh/hyzpgx1hp9nj37s/AADfY2s7gD_MkR76m03KS0K1a/Archive.zip?dl=1'
echo "[$(date)] start"
wget -c --tries=20 --timeout=60 --progress=dot:giga -O "$Z" "$U"
ls -l "$Z"
unzip -l "$Z" > $D/raw/MulT_Archive.listing.txt
cat $D/raw/MulT_Archive.listing.txt
for f in mosi_data.pkl mosi_data_noalign.pkl mosei_senti_data.pkl; do
  m=$(awk '{print $4}' $D/raw/MulT_Archive.listing.txt | grep -E "(^|/)$f$" | head -1)
  [ -n "$m" ] || { echo "MISSING $f"; exit 2; }
  unzip -o -j "$Z" "$m" -d "$D"
done
cd $D && sha256sum mosi_data.pkl mosi_data_noalign.pkl mosei_senti_data.pkl | tee SHA256SUMS
echo "[$(date)] DONE" ; touch $D/.fetch_done

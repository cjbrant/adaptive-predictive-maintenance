#!/usr/bin/env bash
set -e

echo "=== Downloading all datasets ==="

echo ""
echo "--- CWRU Bearing Dataset ---"
python3 -c "
import sys; sys.path.insert(0, '.')
from datasets.cwru.download import download_cwru_data
download_cwru_data(subset='minimal')
"

echo ""
echo "--- IMS Bearing Dataset (~6 GB) ---"
python3 -c "
import sys; sys.path.insert(0, '.')
from datasets.ims.download import download_ims_data
download_ims_data()
"

echo ""
echo "--- FEMTO/PRONOSTIA Dataset ---"
python3 -c "
import sys; sys.path.insert(0, '.')
from datasets.femto.download import download_femto_data
download_femto_data()
"

echo ""
echo "--- C-MAPSS Dataset ---"
python3 -c "
import sys; sys.path.insert(0, '.')
from datasets.cmapss.download import download_cmapss_data
download_cmapss_data()
"

echo ""
echo "=== All downloads complete ==="

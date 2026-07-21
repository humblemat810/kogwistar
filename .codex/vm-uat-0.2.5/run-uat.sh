#!/bin/sh
set -eu
python -m venv /tmp/adr015-venv
/tmp/adr015-venv/bin/python -m pip install --upgrade pip
/tmp/adr015-venv/bin/python -m pip install /work/kogwistar-0.2.5-cp312-abi3-manylinux_2_34_x86_64.whl
/tmp/adr015-venv/bin/python -m pip check
/tmp/adr015-venv/bin/python -P /work/adr015_consumer_uat.py \
  --workdir /tmp/adr015-consumer-state \
  --report /work/adr015-consumer-uat-0.2.5.json

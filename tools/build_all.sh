#!/bin/sh
# Regenerates everything: full-detail Blockbench models (hi), in-game near meshes, in-game far meshes.
# All three come from the same sculpt (grid step 0.3) and are then simplified to different face budgets.
set -e
G=${GEN:-$(dirname $0)}; OUT=${1:-out}
rm -rf $OUT; mkdir -p $OUT
SDF_STEP=0.3 QUAL=hi   TARGET_BODY=18000 TARGET_HAND=7000 TARGET_FOOT=8000 python3 $G/sdf3.py $OUT/hi   > $OUT/hi.log 2>&1 &
SDF_STEP=0.3 QUAL=near TARGET_BODY=6000  TARGET_HAND=1700 TARGET_FOOT=1900 python3 $G/sdf3.py $OUT/near > $OUT/near.log 2>&1 &
SDF_STEP=0.4 QUAL=far  TARGET_BODY=2600  TARGET_HAND=520  TARGET_FOOT=650  python3 $G/sdf3.py $OUT/far  > $OUT/far.log 2>&1 &
wait
cat $OUT/hi.log $OUT/near.log $OUT/far.log
python3 $(dirname $0)/convert.py $OUT/near $OUT/mod
python3 $(dirname $0)/convert.py $OUT/far $OUT/mod _lod
rm -f $OUT/mod/*_lod.png
python3 $(dirname $0)/convert.py $OUT/hi $OUT/preview > /dev/null

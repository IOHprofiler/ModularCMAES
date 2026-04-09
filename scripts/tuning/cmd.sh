#!/bin/bash

fid=$1

jq '(.data | min_by(.cost)) as $b | {result: $b, config: .configs[$b.config_id|tostring]}' data/BBOB_F${fid}_5D_LRFalse/*/0/runhistory.json | sed 's/true/True/g' | sed 's/false/False/g'

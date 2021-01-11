#!/bin/bash
set -x

ARGS="$@"

if $(echo "$ARGS" | grep -q 'participants=auto'); then
    # set number of participants to num_cpus / 2
    NUM_PARTICIPANTS=$[ $(nproc) / 2]
    ARGS=$(echo $ARGS | sed "s/--participants=auto/--participants=$NUM_PARTICIPANTS/")
fi

# start selenium grid in background
./opt/bin/entry_point.sh &
sleep 2

cd /jitsi-meet-torture
./scripts/malleus.sh $ARGS

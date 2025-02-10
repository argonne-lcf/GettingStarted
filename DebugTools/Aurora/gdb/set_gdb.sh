#!/usr/bin/env bash

# debug issue on rank 0
if [ "${PALS_LOCAL_RANKID}" == "0" ]; then
	export INTELGT_AUTO_ATTACH_DISABLE=1
	gdb-oneapi --args "$@"
else
	"$@"
fi

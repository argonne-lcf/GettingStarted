#!/bin/bash

# Original version created by Nitin Dhamankar and Aditya Nishtala (Intel) to help debug hangs in LAMMPS on Aurora

GSTACK_WRAPPER_LOC=/soft/tools/gstack-gdb-oneapi/bin/gstack-gdb-oneapi-cpu.sh

# for consistent date formatting
function date_ () {
  TZ="UTC" \date +"%a %b %d %T UTC %Y"
}

# declare a function to wait for pid on each line of an input file
function wait_on_processes () {

    while IFS="" read -r p || [ -n "$p" ]
    do
        wait $p
    done < $1
}

# Name of logfile to monitor
LOGF="$1"

# Name of executable to kill
if [ ! -f $APP_EXE ]; then
  echo "[LOG MONITOR, $(date_)] ERROR: APP_EXE not defined! Exiting .."
  exit 1
fi

#Set the default nodefile
if [[ -z ${APP_NODEFILE+x} ]]; then
    APP_NODEFILE=$PBS_NODEFILE
fi

MY_PARENT=`ps -o ppid -p $$ | tail -1 | xargs`

# Set the default start delay and monitor frequency (both in seconds/minutes)
if [[ -z ${APP_LOGMON_START_DELAY+x} ]]; then
  APP_LOGMON_START_DELAY=5
fi
if [[ -z ${APP_LOGMON_MONFREQ+x} ]]; then
  APP_LOGMON_MONFREQ=10
fi

# Set the default slowdown threshold (in minutes)
if [[ -z ${APP_LOGMON_SLOWDOWN_THRESHOLD+x} ]]; then
  APP_LOGMON_SLOWDOWN_THRESHOLD=30
fi

# Initial wait
#sleep $APP_LOGMON_START_DELAY
sleep $(($APP_LOGMON_START_DELAY * 60))

# Check if LOGF is now available
if [ ! -f $LOGF ]; then
  echo "[LOG MONITOR, $(date_)] ERROR: Log file is still not available! Exiting .."
  exit 1
fi


# record main mpiexec process ID for the run ; what's a cleaner way to get this for arbitrarily named job script??
MPIEXEC_PROC=$MY_PARENT #`ps -o pid,ppid,cmd --no-headers | grep -e "$MY_PARENT" -e "submit.sh" | awk '{print $1}' | xargs`
NLINES_OLD=`wc -l $LOGF | awk '{print $1}'`

# Do this till the calling script kills the current process and while the calling script is still running...
while (( `ps -p $MY_PARENT | wc -l` == 2 )); do

  KILL_NOW=0
#  sleep $APP_LOGMON_MONFREQ
  sleep $(($APP_LOGMON_MONFREQ * 60))

  NLINES_NEW=`wc -l $LOGF | awk '{print $1}'`

  # Check if no update to log since last check
  if (( $NLINES_NEW == $NLINES_OLD )) && (( `ps -p $MY_PARENT | wc -l` == 2 )); then
    echo "[LOG MONITOR, $(date_)] ERROR: Log file $LOGF has not been updated in the last $APP_LOGMON_MONFREQ minutes!" | tee -a $LOGF
    KILL_NOW=1

    # collect call stacks at the hang point
    cp $APP_NODEFILE ./app_nodefile
    split -l 1024 app_nodefile app_nodefile_split_
    for i in app_nodefile_split_*; do
      clush -f 1024 -S -t 30 -u 420 --hostfile ${i} "pidof $(basename ${APP_EXE}) | xargs -n 1 ${GSTACK_WRAPPER_LOC} | gzip | base64 -w0" | dshbak -d callstacks_at_hang_point/ -f
    done
    rm -f app_nodefile app_nodefile_split_*

    for j in callstacks_at_hang_point/x4*; do
      mv $j ${j}_compressed
      cat ${j}_compressed | base64 -d | gunzip > ${j}
    done

    COUNT_PROCS=0
    FANOUT=1024

  fi

  if (( $KILL_NOW == 1 )); then

    echo "[LOG MONITOR, $(date_)] Issuing a kill of $APP_EXE processes across all nodes!" | tee -a $LOGF

    # Kill app
    clush -f 1024 -S -t 30 -u 60 --hostfile $APP_NODEFILE "ps -ef | grep -e ${APP_EXE} | awk '{print \$2}' | xargs kill -9"

    sleep 5
    kill $MPIEXEC_PROC
    sleep 10
    if (( `ps -p $MPIEXEC_PROC | wc -l` == 2 )); then
      kill -9 $MPIEXEC_PROC
    fi

  fi

  NLINES_OLD=$NLINES_NEW

done

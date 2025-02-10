#!/bin/bash

kill $1
sleep 5
if (( `ps -p $1 | wc -l` == 2 )); then
  kill -9 $1 
fi

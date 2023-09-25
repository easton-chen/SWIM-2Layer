#!/bin/bash
MAINSIMDIR=../../src/
MAINSIMEXEC=swim

if [ $# -lt 1 ]; then
	echo "usage: $0 config [run-number(s)|all [ini-file]]"
	echo example:
	echo "  "$0 Reactive 0..2
	exit 1
fi

RUNS=""
if [ "$2" != "" ] && [ "$2" != "all" ]; then
    RUNS="-r $2"
fi

CONTROLLER="NewMPCController.py"
if [ "$3" != "" ]; then
	if [ $3 == 0 ]; then
    	CONTROLLER="NewMPCController.py"
	elif [ $3 == 1 ]; then
		CONTROLLER="CobRaMPCController.py"
	elif [ $3 == 2 ]; then
		CONTROLLER="BaseMPCController.py"
	fi
fi

INIFILE="swim_test.ini"
if [ "$4" != "" ]; then
    INIFILE="$4"
fi


python $CONTROLLER >MPC.log &
PID=$!
sleep 5s
opp_runall -j1 $MAINSIMDIR/$MAINSIMEXEC $INIFILE -u Cmdenv -c $1 -n ..:$MAINSIMDIR:../../../queueinglib:../../src -lqueueinglib $RUNS

sleep 2s
kill -9 $PID
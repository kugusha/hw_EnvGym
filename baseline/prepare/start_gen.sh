# Number of tasks for each map
N_TASK=$1

SCRIPT=`realpath $0`
SCRIPT_DIR=`dirname $SCRIPT`

MAPS_DIR="$SCRIPT_DIR/../data/imported/maps"
PATHS_DIR="$SCRIPT_DIR/../data/imported/paths"

for map in $MAPS_DIR/*; do
    python2 gen_tasks.py -n $N_TASK $map $PATHS_DIR
done

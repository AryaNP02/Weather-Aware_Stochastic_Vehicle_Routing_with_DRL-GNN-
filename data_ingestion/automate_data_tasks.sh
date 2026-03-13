#!/bin/bash
# automated data tasks script

echo "Running Data Automation Script..."
cd "$(dirname "$0")"

if [ -z "$1" ]; then
    echo "Usage: ./automate_data_tasks.sh [generate|clean] [num_nodes] [num_instances]"
    echo "Defaulting to 'generate' 10 nodes, 1000 instances..."
    python3 data_manager.py --num_nodes 10 --num_instances 1000
    exit 0
fi

COMMAND=$1

case $COMMAND in
    generate)
        NODES=${2:-10}
        INSTS=${3:-1000}
        python3 data_manager.py --num_nodes "$NODES" --num_instances "$INSTS"
        ;;
    clean)
        echo "Cleaning all .pkl dataset files..."
        rm -f *.pkl
        echo "Clean complete."
        ;;
    *)
        echo "Unknown command: $COMMAND"
        ;;
esac

#!/bin/bash

echo "=== VGGish Embedding Generation Progress ==="
echo ""

# Check if process is still running
if ps -p 6185 > /dev/null 2>&1; then
    echo "Status: RUNNING (PID: 6185)"
else
    echo "Status: COMPLETED or STOPPED"
fi

echo ""
echo "=== Last 30 lines of log ==="
tail -30 vggish_final_output.log

echo ""
echo "=== Output files ==="
ls -lh work/tokenized/vggish_embeddings_*.csv 2>/dev/null || echo "No output files yet"

if [ -f "work/tokenized/vggish_embeddings_train_curated.csv" ]; then
    echo ""
    echo "=== Train embeddings file info ==="
    wc -l work/tokenized/vggish_embeddings_train_curated.csv
    head -1 work/tokenized/vggish_embeddings_train_curated.csv | tr ',' '\n' | wc -l
    echo "columns in file"
fi

if [ -f "work/tokenized/vggish_embeddings_test.csv" ]; then
    echo ""
    echo "=== Test embeddings file info ==="
    wc -l work/tokenized/vggish_embeddings_test.csv
fi

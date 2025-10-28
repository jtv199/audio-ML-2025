#!/bin/bash
# Run MobileNet Error Analysis
# This will take ~10-15 minutes to process all audio files

echo "Starting MobileNet Error Analysis..."
echo "This will process 853 audio files and may take 10-15 minutes."
echo ""

# Activate conda environment and run
~/miniconda3/envs/freesound/bin/python mobilenet_error_analysis.py 2>&1 | grep -v "^█"

echo ""
echo "✓ Analysis complete!"
echo ""
echo "Results saved to:"
echo "  - claude/lazypredict/findings/error_analysis_mobilenet.csv"
echo "  - claude/lazypredict/findings/mobilenet_vs_linearsvc_comparison.csv"

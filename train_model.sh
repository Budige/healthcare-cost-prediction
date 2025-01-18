#!/bin/bash
echo "Training healthcare cost prediction model..."
python src/model_training.py
echo "Model training complete. Results saved to output/models/"

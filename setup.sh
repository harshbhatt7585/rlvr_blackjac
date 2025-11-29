#!/bin/bash

# Setup script for RLVR Blackjack training

echo "=========================================="
echo "RLVR Blackjack Training Setup"
echo "=========================================="

# Check Python version
echo -e "\n1. Checking Python version..."
python_version=$(python3 --version 2>&1 | grep -oP '(?<=Python )\d+\.\d+')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    echo "✓ Python $python_version detected"
else
    echo "✗ Python 3.8+ required, found $python_version"
    exit 1
fi

# Check if pip is installed
echo -e "\n2. Checking pip..."
if command -v pip3 &> /dev/null; then
    echo "✓ pip is installed"
else
    echo "✗ pip is not installed"
    exit 1
fi

# Install dependencies
echo -e "\n3. Installing dependencies..."
pip3 install -r requirements.txt

# Check if installation was successful
if [ $? -eq 0 ]; then
    echo "✓ Dependencies installed successfully"
else
    echo "✗ Failed to install dependencies"
    exit 1
fi

# Test environment
echo -e "\n4. Testing environment..."
python3 test_env.py

if [ $? -eq 0 ]; then
    echo -e "\n✓ Environment tests passed"
else
    echo -e "\n✗ Environment tests failed"
    exit 1
fi

# Check Hugging Face login
echo -e "\n5. Checking Hugging Face authentication..."
if huggingface-cli whoami &> /dev/null; then
    echo "✓ Logged in to Hugging Face"
else
    echo "⚠️  Not logged in to Hugging Face"
    echo "   Run: huggingface-cli login"
    echo "   Required to download Gemma models"
fi

# Create directories
echo -e "\n6. Creating output directories..."
mkdir -p checkpoints
mkdir -p logs
echo "✓ Directories created"

echo -e "\n=========================================="
echo "Setup Complete!"
echo "=========================================="
echo -e "\nNext steps:"
echo "1. Login to Hugging Face (if not already): huggingface-cli login"
echo "2. Request access to Gemma: https://huggingface.co/google/gemma-2b-it"
echo "3. Run training: python3 train_rlvr.py"
echo "4. Watch trained model: python3 play_blackjack.py --model ./checkpoints/final_model"
echo ""

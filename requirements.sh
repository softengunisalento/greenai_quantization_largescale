#!/bin/bash

# Update package list and install pip if not already installed
if ! command -v pip &> /dev/null
then
    echo "pip could not be found, installing pip..."
    sudo apt update
    sudo apt install python3-pip -y
fi

# List of packages to install
packages=(
    "transformers"
    "quanto"
    "accelerate"
    "datasets"
    "setuptools"
    "bitsandbytes"
    "torch"
    "torchvision"
    "torchaudio"
)

result=0
failed_packages=()

# Install each package in the list
for package in "${packages[@]}"
do
    echo "📥 Installing $package..."
    pip install "$package"

    # Check if the exit code is zero
    exit_code=$?

    if [ $exit_code != 0 ]; then
        result=$exit_code
        failed_packages+=("$package")
    fi
done

if [ $result -eq 0 ]; then
    echo "Installation of packages is complete ✅"
else
    echo "Installation of packages failed ❌"
    echo "Failed packages:"
    for failed_package in "${failed_packages[@]}"
    do
        echo "- $failed_package"
    done
fi

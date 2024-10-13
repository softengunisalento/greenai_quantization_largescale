#!/bin/bash

# Check if the required argument (folder path) is passed
if [ "$#" -ne 1 ]; then
	echo "Usage: $0 /path/to/project"
	exit 1
fi

PROJECT_PATH=$1

# Clone the EnergiBridge repository and build the project
git clone https://github.com/tdurieux/EnergiBridge.git ~/EnergiBridge
cd ~/EnergiBridge
cargo build -r
cd

# Clone the pyEnergiBridge repository and install it
git clone https://github.com/luiscruz/pyEnergiBridge.git ~/pyEnergiBridge
pip install ~/pyEnergiBridge

# Create the pyenergibridge configuration file in /root/models_testing
echo '{"binary_path": "~/EnergiBridge/target/release/energibridge"}' > "$PROJECT_PATH/pyenergibridge_config.json"

# Create the pyenergibridge configuration file in /root
echo '{"binary_path": "~/EnergiBridge"}' > ~/.pyenergibridge_config.json

echo "Setup completed successfully!"
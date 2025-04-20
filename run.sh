#!/bin/bash

BASE_DIR="./xxx"
FILES=($(find "$BASE_DIR" -type f -name "db_qd_*.py"))
TOTAL=${#FILES[@]}

if [ "$TOTAL" -eq 0 ]; then
    echo "No matching files found."
    exit 1
fi

echo "Found $TOTAL script(s). Starting execution..."
echo

i=0
for file in "${FILES[@]}"; do
    ((i++))
    echo -ne "[$i/$TOTAL] Running $(basename "$file")... "

    python3 "$file" -m -q 4 > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo -e "\e[32mOK\e[0m"
    else
        echo -e "\e[31mFAIL\e[0m"
    fi
done

echo
echo "✅ All scripts processed."

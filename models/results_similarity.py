import re
import sys
import subprocess
import difflib

def extract_value(line):
    """
    Extracts the portion of the line that matches the regex '>\s*[^:]*:\s*(.*)'
    """
    line = line.replace('%', '')
    pattern = r'>\s*[^:]*:\s*(.*)'
    match = re.search(pattern, line)
    if match:
        return match.group(1).strip()
    return None

def is_number(s):
    """
    Determines if the extracted value is a number (integer or float)
    """
    try:
        float(s)
        return True
    except ValueError:
        return False

def compare_numbers(num1, num2):
    """
    Calculates the percentage similarity between two numbers
    """
    num1 = float(num1)
    num2 = float(num2)
    if num1 == num2:
        return 1
    max_val = max(num1, num2)
    min_val = min(num1, num2)
    similarity = (min_val / max_val)
    return similarity

def compare_text_charactes(text1, text2):
    """
    Calculates the text similarity using difflib
    """
    similarity = difflib.SequenceMatcher(None, text1, text2).ratio()
    return similarity * 100

def preprocess_string(s):
    """
    Preprocess the string: remove non-alphanumeric characters and convert to lowercase
    """
    s = re.sub(r'[^\w\s]', '', s)  # Remove punctuation
    s = s.lower()  # Convert to lowercase
    return s

def jaccard_similarity(str1, str2):
    """
    Calculates the Jaccard similarity between two strings
    """
    # Preprocess the strings
    str1 = preprocess_string(str1)
    str2 = preprocess_string(str2)
    
    # Split strings into sets of words
    set1 = set(str1.split())
    set2 = set(str2.split())
    
    # Calculate the Jaccard similarity: intersection / union
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    if union == 0:
        return 0.0  # To handle division by zero in case both sets are empty
    
    return intersection / union

def run_file(file):
    """
    Runs the specified Python file and returns its output as a list of lines
    """
    try:
        result = subprocess.run(['python3', file], capture_output=True, text=True, check=True)
        return result.stdout.splitlines()
    except subprocess.CalledProcessError as e:
        print(f"Error running {file}: {e.stderr}")
        return []

def process_files(file1, file2, file3):
    """
    Runs two Python files, compares their outputs after extracting values using regex
    """
    output1 = run_file(file1)
    output2 = run_file(file2)

    print(f"Output of {file1}:")
    for line in output1:
        print(line)
    print()

    print(f"Output of {file2}:")
    for line in output2:
        print(line)
    print()

    max_lines = max(len(output1), len(output2))
    i = 0
    while i < max_lines:
        
        value1 = extract_value(output1[i]) if i < len(output1) else None
        value2 = extract_value(output2[i]) if i < len(output2) else None

        if value1 and value2:
            if is_number(value1) and is_number(value2):
                similarity = compare_numbers(value1, value2)
                print(f"Line {i + 1}: Numbers compared: {value1} vs {value2} -> {similarity}% similarity")
            elif value1.startswith('tensor([') and value2.startswith('tensor(['):
                mse = float(run_file(file3)[0])
                similarity = 100 - (mse*100)
                print(f"Line {i}: Tensors compared: '{value1} ...]])' vs '{value2} ...]]' -> {similarity}% similarity")
            else:
                similarity = jaccard_similarity(value1, value2)
                print(f"Line {i + 1}: Text compared: '{value1}' vs '{value2}' -> {similarity*100}% similarity")
        else:
            print(f"Line {i + 1}: No matching values found or format incorrect.")

        i += 1

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 compare_files.py <file1.py> <file2.py> <logits_file.py>")
        sys.exit(1)

    file1 = sys.argv[1]
    file2 = sys.argv[2]
    file3 = sys.argv[3]

    process_files(file1, file2, file3)
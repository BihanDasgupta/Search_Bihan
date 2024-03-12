import pandas as pd
import os
from striprtf.striprtf import rtf_to_text

# Data structure
class fileLine:
    def __init__(self, fileName, textLine):
        self.fileName = fileName
        self.textLine = textLine

# Data files directories
data_dir_path = '/Users/bihan/Desktop/career/website/bihan_corpus'
output_dir_path = '/Users/bihan/Desktop/career/website'

# Get all lines of all RTF files in a giver directory
def getRTFLines():
    list = []
    # For each file in the given directory
    for filename in os.listdir(data_dir_path):
        filePath = os.path.join(data_dir_path, filename)
        # Check if it's an RTF file
        if os.path.isfile(filePath) and filePath.endswith('.rtf'):
            # Open the RTF file
            with open(filePath, encoding='utf-8', errors='ignore') as file:
                # Read all the RTF file's content
                text = file.read()
                # Decode the RTF file's content
                rtfText = rtf_to_text(text, encoding='utf-8')
                # For each line in the RTF file
                for line in rtfText.splitlines():
                    # Check if it's an empty line
                    if line:
                        # Insert the line in our data structure
                        list.append(fileLine(filename, line))
    return list

# Read data
list = getRTFLines()

# Convert data to csv file
data = [[x.fileName, x.textLine] for x in list]
df = pd.DataFrame(data, columns=['File Name', 'Text Line'])
df.to_csv(os.path.join(output_dir_path,r'rtfData-3.csv'), escapechar="")

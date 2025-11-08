import sys
import time
import openai
import pandas as pd
from tqdm import tqdm  # Import tqdm for the progress bar

def write_result(filename, result):
    with open(filename,'w') as f:
        for item in result:
            f.write(item.strip())
            f.write("\n")

def read_result(filename):
    output = []
    with open(filename,'r') as f:
        for item in f:
            #print(item.split(" - "), item.split(" - ")[-1])
            #classl = item.split(" - ")[-1].strip()
            classl = item.rsplit(' - ', 1)[-1].strip()
            output.append(classl)
    return output
            
# Example usage
if __name__ == "__main__":
    # List of paper titles to classify
    category = sys.argv[1]
    print(category)
    lbls = read_result(f"{category}.txt")
    write_result(f'{category}.lbl.txt', lbls)

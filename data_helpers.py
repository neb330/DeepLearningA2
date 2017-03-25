import numpy as np
import re
import itertools
import collections
import os
import nltk

UNIGRAM_SIZE = 10000


def prepare_data(path = "data/gutenberg/"):
    count = [['<unk>', -1]]
    # Load data from files
    with open((path + "train.txt"), "r",  encoding = "ISO-8859-1") as f:
        words = [word for line in f for word in line.split()]
        count.extend(collections.Counter(words).most_common(UNIGRAM_SIZE - 1))
        count = dict(count)
    for filename in os.listdir(path):
        text = open((path + filename), "r",  encoding = "ISO-8859-1").readlines()
        outfile = open((path + filename.split('.')[0] + "_clean.txt"), "w")
        for line in text:
            lst = line.split()
            no_unks = [w if w in count.keys() else '<unk>' for w in lst]
            no_nums = [w if not w.isdigit() else 'N' for w in no_unks]
            outfile.write(' '.join(no_nums) + '\n')

def main():
    prepare_data()

if __name__ == "__main__":
    main()

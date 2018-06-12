# Mark Evers
# 5/9/2018
# globals.py
# Global variables and functions

from sys import stdout
from time import time


# performance stuff
DEBUG_LEVEL = 1
DOC_BUFFER_SIZE = 1000
RANDOM_STATE = 666

# vectorizer options
MAX_FEATURES = 50000
MIN_DF = 3
MAX_DF = .5
N_GRAMS = 1

# nmf options
N_TOPICS = 42
MAX_ITER = 1000

# word cloud options
BACKGROUND_COLOR = "black"
MAX_WORDS = 250
WIDTH = 750
HEIGHT = 375



_PROGRESS_BAR_LAST_TIME = 0
def progress_bar(done, total, resolution=0.333, text=""):
    """
    Prints a progress bar to stdout.
    :param done: Number of items complete
    :param total: Total number if items
    :param resolution: How often to update the progress bar (in seconds).
    :return: None
    """

    global _PROGRESS_BAR_LAST_TIME

    time_now = time()
    if time_now - _PROGRESS_BAR_LAST_TIME < resolution and done < total:
        return

    # percentage done
    i = int(done / total * 100)

    stdout.write('\r')
    # print the progress bar
    stdout.write("[{}]{}%".format(("-" * int(i / 2) + (">" if i < 100 else "")).ljust(50), str(i).rjust(4)))
    # print the text figures
    stdout.write(" ({}/{})".format(done, total))
    if text:
        stdout.write(" " + text)
    stdout.flush()

    if i == 100:
        # print("\n")
        stdout.write('\r')
        stdout.write(' ' * 120)
        stdout.write('\r')

    _PROGRESS_BAR_LAST_TIME = time_now


def debug(text, level = 0):

    if level <= DEBUG_LEVEL:
        print(text)

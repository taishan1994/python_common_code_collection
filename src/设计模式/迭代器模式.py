__author__ = "Burgess Zheng"
# !/usr/bin/env python
# -*- coding:utf-8 -*-

'''
Interator
'''


def count_to(count):
    """Counts by word numbers, up to a maximum of five"""
    numbers = ["one", "two", "three", "four", "five"]
    # enumerate() returns a tuple containing a count (from start which
    # defaults to 0) and the values obtained from iterating over sequence
    for pos, number in zip(range(count), numbers):
        yield number


# Test the generator
count_to_two = count_to(2)
count_to_five = count_to(5)

print('Counting to two...')
for number in count_to_two:
    print(number)

print(" ")

print('Counting to five...')
for number in count_to_five:
    print(number)

print(" ")
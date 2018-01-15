"""
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD License
https://gist.github.com/karpathy/d4dee566867f8291f086
"""
import numpy as np

# data I/O
data = open('input.txt', 'r').read() # should be simple plain text file
chars = list(set(data))
chars.sort(key=lambda x:(not x.islower(), x))


print chars
#for char in chars:
#	print '%d, ' % ord(char)

data_size, vocab_size = len(data), len(chars)
print 'data has %d characters, %d unique.' % (data_size, vocab_size)
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

# Param
seq_length = 100
sample_size = 100000

max_sample_size = int(data_size/vocab_size)
sample_size = max_sample_size if (sample_size > max_sample_size) else sample_size

out_file = open("shake.txt", "w")
out_file.write('{}\n'.format(sample_size))
out_file.write('{}\n'.format(vocab_size))

p = 0
while p+seq_length < len(data):
	out_file.write(str(seq_length) + '\n')
	inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
	out_file.write('\t'.join(map(str, inputs)) + '\n')

	out_file.write(str(seq_length) + '\n')
	targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]
	out_file.write('\t'.join(map(str, targets)) + '\n')
	p += seq_length


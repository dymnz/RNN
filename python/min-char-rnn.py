"""
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD License
"""
import numpy as np

# data I/O
data = open('speeches.txt', 'r').read() # should be simple plain text file
chars = list(set(data))
chars.sort()
print chars
print chars.index('\n')
data_size, vocab_size = len(data), len(chars)
print 'data has %d characters, %d unique.' % (data_size, vocab_size)
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

seq_length = 100
max_sample_size = data_size / seq_length
sample_size = 100

if sample_size > max_sample_size:
  sample_size = max_sample_size

out_file = open("exp_trump.txt","w")
out_file.write('{}\n'.format(sample_size))
out_file.write('{}\n'.format(vocab_size))

n, p = 0, 0
# prepare inputs (we're sweeping from left to right in steps seq_length long)
while (n < sample_size) and (p+seq_length+1 <= len(data)) :
  inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
  targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]
  
  out_file.write(str(len(inputs)) + '\n')
  out_file.write('\t'.join(map(str, inputs)) + '\n')
  out_file.write(str(len(targets)) + '\n')
  out_file.write('\t'.join(map(str, targets)) + '\n')

  p += seq_length # move data pointer
  n += 1 # iteration counter 

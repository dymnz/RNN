```
thread 1
hidden cell 100
epochs 10 
reddit_14
print loss 1
learn adjust 1
gradient check 100000

> time(./rnn reddit_1000 reddit_1000 100 10 0.001 1 1 100000 4 1)
```
real    3m24.463s
user    3m24.335s
sys     0m0.105s

---

```
thread 8
hidden cell 100
epochs 10
reddit_14
print loss 1
learn adjust 1
gradient check 100000

> time(./rnn reddit_1000 reddit_1000 100 10 0.001 1 1 100000 4 8)
```
real    2m10.094s
user    3m43.735s
sys     0m0.397s

---
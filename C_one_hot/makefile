CC=gcc
CFLAGS=-Wall
LFLAGS=-lm
DEPS = common_math.h RNN.h file_process.h
TEST_OBJ = test.o common_math.o RNN.o file_process.o


%.o: %.c $(DEPS)
	$(CC) $(CFLAGS) -c -o $@ $< $(LFLAGS)

test: $(TEST_OBJ)
	gcc $(CFLAGS) -o $@ $^ $(LFLAGS)

clean:
	rm $(TEST_OBJ)
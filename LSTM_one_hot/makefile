CC=gcc
CFLAGS=-Wall -DFILE_NAME_LENGTH=255
LFLAGS=-lm
DEPS = common_math.h RNN.h file_process.h
DEPS_OBJ = common_math.o RNN.o file_process.o
RNN_OBJ = main.o

all: $(DEPS_OBJ) $(RNN_OBJ)
	$(CC) $(CFLAGS) -o rnn $^ $(LFLAGS)

%.o: %.c $(DEPS)
	$(CC) $(CFLAGS) -c -o $@ $< $(LFLAGS)

rnn: $(DEPS_OBJ) $(RNN_OBJ)
	$(CC) $(CFLAGS) -o rnn $^ $(LFLAGS)

test: $(TEST_OBJ) $(DEPS_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LFLAGS)

clean:
	rm $(TEST_OBJ) $(RNN_OBJ) $(DEPS_OBJ)

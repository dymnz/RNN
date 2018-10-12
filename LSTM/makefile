CC=gcc
CFLAGS=-Wall -DFILENAME_LENGTH=255 -Dpacket_size=4 -Din_packet_num=6 -Dout_packet_num=1
LFLAGS=-lm
DEPS = common_math.h RNN.h file_process.h serial.h
TEST_OBJ = test.o
DEPS_OBJ = common_math.o RNN.o file_process.o serial.o
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
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <termios.h>
#include <unistd.h>
#include <math.h>

#include <sys/time.h>

struct timeval last_time, current_time;

union float_packet {
	float float_value;
	char byte_array[4];
};

int set_interface_attribs(int fd, int speed);
void set_mincount(int fd, int mcount);
int serial_init(char *portname);
#include "util.h"


#ifdef GNU_TIME
double elapsed_time(const mytspec t2, const mytspec t1) {
	return 1.0 * (t2 - t1) / CLOCKS_PER_SEC;
}
#else
double elapsed_time(const mytspec t2, const mytspec t1) {
	return (((double)t2.tv_sec) + ((double)t2.tv_nsec / 1e9))
	       - (((double)t1.tv_sec) + ((double)t1.tv_nsec / 1e9));
}
#endif

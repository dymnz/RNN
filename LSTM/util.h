#include <time.h>	/* clock/clock_gettime */

/* ickiness in naming of time stuff */
#define GNU_TIME
#ifdef GNU_TIME
#  define __need_clock_t
#  define mytspec clock_t
#  define get_time(tspec) tspec = clock()
#else
#  define mytspec timespec_t
#  define get_time(tspec) clock_gettime(CLOCK_SGI_CYCLE,&tspec)
#endif

double elapsed_time(const mytspec t2, const mytspec t1);

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <timemory/ctimemory.h>

//======================================================================================//

intmax_t
fibonacci(intmax_t n)
{
    return (n < 2) ? 1L : (fibonacci(n - 1) + fibonacci(n - 2));
}

//======================================================================================//

intmax_t
timed_fibonacci(intmax_t n, intmax_t cutoff)
{
    if(n > cutoff)
    {
        void*    timer = TIMEMORY_BASIC_AUTO_TIMER("");
        intmax_t _n =
            (n < 2) ? n
                    : (timed_fibonacci(n - 1, cutoff) + timed_fibonacci(n - 2, cutoff));
        FREE_TIMEMORY_AUTO_TIMER(timer);
        return _n;
    }
    else
    {
        return (n < 2) ? n : (fibonacci(n - 1) + fibonacci(n - 2));
    }
}

//======================================================================================//

void*
get_timer(const char* func)
{
    return TIMEMORY_AUTO_TUPLE(func, WALL_CLOCK, SYS_CLOCK, CPU_CLOCK, CPU_UTIL,
                               CURRENT_RSS, PEAK_RSS, PRIORITY_CONTEXT_SWITCH,
                               VOLUNTARY_CONTEXT_SWITCH);
}

//======================================================================================//

void
free_timer(void* timer)
{
    FREE_TIMEMORY_AUTO_TUPLE(timer);
}

//======================================================================================//

int
main()
{
    int nfib = 44;
    int ncut = nfib - 20;
    printf("... \"%s\" : %s @ %i\n", __FILE__, __FUNCTION__, __LINE__);

    void*    timer0 = get_timer("[main (untimed)]");
    intmax_t n0     = fibonacci(nfib);
    free_timer(timer0);

    void*    timer1 = get_timer("[main (timed)]");
    intmax_t n1     = timed_fibonacci(nfib, ncut);
    free_timer(timer1);

    printf("... \"%s\" : %s @ %i --> n = %lli and %lli\n", __FILE__, __FUNCTION__,
           __LINE__, (long long int) n0, (long long int) n1);

    return 0;
}

//======================================================================================//

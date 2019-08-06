
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <timemory/ctimemory.h>

static long nlaps = 0;

//======================================================================================//

void*
get_timer(const char* func)
{
    return TIMEMORY_AUTO_TUPLE(func, WALL_CLOCK, SYS_CLOCK, USER_CLOCK, CPU_CLOCK,
                               CPU_UTIL, CURRENT_RSS, PEAK_RSS, PRIORITY_CONTEXT_SWITCH,
                               VOLUNTARY_CONTEXT_SWITCH);
}

//======================================================================================//

void*
get_fibonacci_timer(const char* func, long n)
{
    char* buffer = (char*) (malloc(64 * sizeof(char)));
    sprintf(buffer, "%s[%li]", func, n);
    return TIMEMORY_BLANK_AUTO_TUPLE(buffer, WALL_CLOCK, SYS_CLOCK, USER_CLOCK);
}

//======================================================================================//

void
free_timer(void* timer)
{
    FREE_TIMEMORY_AUTO_TUPLE(timer);
}

//======================================================================================//

intmax_t
_fibonacci(intmax_t n)
{
    return (n < 2) ? n : (_fibonacci(n - 1) + _fibonacci(n - 2));
}

//======================================================================================//

intmax_t
fibonacci(intmax_t n, intmax_t cutoff)
{
    if(n > cutoff)
    {
        nlaps += 3;
        void*    timer = get_fibonacci_timer(__FUNCTION__, n);
        intmax_t _n = (n < 2) ? n : (fibonacci(n - 1, cutoff) + fibonacci(n - 2, cutoff));
        free_timer(timer);
        return _n;
    }
    else
    {
        return _fibonacci(n);
    }
}

//======================================================================================//

int
main(int argc, char** argv)
{
    // default calc: fibonacci(43)
    int nfib = 43;
    if(argc > 1)
        nfib = atoi(argv[1]);

    // only record auto_timers when n > cutoff
    int cutoff = nfib - 20;
    if(argc > 2)
        cutoff = atoi(argv[2]);

    printf("... \"%s\" : %s @ %i. Running fibonacci(%i, %i)...\n", __FILE__, __FUNCTION__,
           __LINE__, nfib, cutoff);

    timemory_settings settings = TIMEMORY_SETTINGS_INIT;
    settings.auto_output       = 1;
    settings.cout_output       = 0;
    settings.json_output       = 1;
    TIMEMORY_INIT(argc, argv, settings);

    void*    timer0 = get_timer("[main (untimed)]");
    intmax_t n0     = _fibonacci(nfib);
    free_timer(timer0);

    printf("main (untimed): fibonacci(%i, %i) = %lli\n", nfib, nfib, (long long) n0);

    void*    timer1 = get_timer("[main (timed)]");
    intmax_t n1     = fibonacci(nfib, cutoff);
    free_timer(timer1);

    printf("main (timed): fibonacci(%i, %i) = %lli\n", nfib, cutoff, (long long) n1);
    printf("# laps = %li\n", nlaps);

    printf("\n... \"%s\" : %s @ %i --> n = %lli and %lli\n", __FILE__, __FUNCTION__,
           __LINE__, (long long int) n0, (long long int) n1);

    return 0;
}

//======================================================================================//

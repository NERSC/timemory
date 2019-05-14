
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <timemory/ctimemory.h>

//======================================================================================//

intmax_t
fibonacci(intmax_t n)
{
    void* timer = NULL;
    if(n > 34)
    {
        timer = TIMEMORY_BASIC_AUTO_TIMER("");
    }
    intmax_t _n = (n < 2) ? 1L : (fibonacci(n - 2) + fibonacci(n - 1));
    FREE_TIMEMORY_AUTO_TIMER(timer);
    return _n;
}

//======================================================================================//

int
main()
{
    printf("... \"%s\" : %s @ %i\n", __FILE__, __FUNCTION__, __LINE__);

    void* timer = c_timemory_create_auto_tuple(
        __FUNCTION__, __LINE__, 8, WALL_CLOCK, SYS_CLOCK, CPU_CLOCK, CPU_UTIL,
        CURRENT_RSS, PEAK_RSS, PRIORITY_CONTEXT_SWITCH, VOLUNTARY_CONTEXT_SWITCH);
    intmax_t n = fibonacci(44);

    c_timemory_delete_auto_tuple(timer);

    printf("... \"%s\" : %s @ %i --> n = %lli\n", __FILE__, __FUNCTION__, __LINE__,
           (long long int) n);

    return 0;
}

//======================================================================================//

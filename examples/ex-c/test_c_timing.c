
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
        int   length = snprintf(NULL, 0, "%lli", (long long int) n);
        char* str    = malloc(length + 1);
        snprintf(str, length + 3, "[%lli]", (long long int) n);
        timer = TIMEMORY_AUTO_TIMER(str);
        free(str);
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

    printf("... \"%s\" : %s @ %i\n", __FILE__, __FUNCTION__, __LINE__);
    intmax_t n = fibonacci(44);
    printf("... \"%s\" : %s @ %i --> n = %li\n", __FILE__, __FUNCTION__, __LINE__, n);
    TIMEMORY_PRINT();
    printf("... \"%s\" : %s @ %i\n", __FILE__, __FUNCTION__, __LINE__);
    TIMEMORY_REPORT("test_output/c_timing_report");
    printf("... \"%s\" : %s @ %i\n", __FILE__, __FUNCTION__, __LINE__);

    return 0;
}

//======================================================================================//

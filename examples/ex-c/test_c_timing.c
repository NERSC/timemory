
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <timemory/ctimemory.h>

#if defined(_WINDOWS)
#define __attribute_constructor__
#else
#define __attribute_constructor__ __attribute__((constructor))
#endif

extern void setup_timemory(void) __attribute_constructor__;

//============================================================================//

int64_t fibonacci(int64_t n)
{
    void* timer = NULL;
    if (n > 34)
    {
        int length = snprintf( NULL, 0, "%lli", (long long int) n );
        char* str = malloc( length + 1 );
        snprintf( str, length + 3, "[%lli]", (long long int) n );
        timer = TIMEMORY_AUTO_TIMER(str);
        free(str);
    }
    int64_t _n = (n < 2) ? 1L : (fibonacci(n-2) + fibonacci(n-1));
    FREE_TIMEMORY_AUTO_TIMER(timer);
    return _n;
}

//============================================================================//

int main(int argc, char** argv)
{
    printf("%s @ %i\n", __FUNCTION__, __LINE__);

    // modify recording memory
    if(argc > 1)
        TIMEMORY_RECORD_MEMORY(atoi(argv[1]));

    int64_t n = fibonacci(44);
    printf("\nANSWER = %lli\n", (long long int) n);

    TIMEMORY_PRINT();
    TIMEMORY_REPORT("test_output/c_timing_report");

    return 0;
}

//============================================================================//

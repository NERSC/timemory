
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <timemory/ctimemory.h>

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
    printf("... \"%s\" : %s @ %i\n", __FILE__, __FUNCTION__, __LINE__);

    // modify recording memory
    if(argc > 1)
        TIMEMORY_RECORD_MEMORY(atoi(argv[1]));

    printf("... \"%s\" : %s @ %i\n", __FILE__, __FUNCTION__, __LINE__);
    int64_t n = fibonacci(44);
    printf("... \"%s\" : %s @ %i\n", __FILE__, __FUNCTION__, __LINE__);
    TIMEMORY_PRINT();
    printf("... \"%s\" : %s @ %i\n", __FILE__, __FUNCTION__, __LINE__);
    TIMEMORY_REPORT("test_output/c_timing_report");
    printf("... \"%s\" : %s @ %i\n", __FILE__, __FUNCTION__, __LINE__);

    return 0;
}

//============================================================================//

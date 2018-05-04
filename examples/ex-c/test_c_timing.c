
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <timemory/ctimemory.h>

int64_t fibonacci(int64_t n)
{
    void* timer = NULL;
    if (n > 36)
    {
        int length = snprintf( NULL, 0, "%lli", n );
        char* str = malloc( length + 1 );
        snprintf( str, length + 3, "[%lli]", n );
        timer = TIMEMORY_C_AUTO_TIMER(str);
        free(str);
    }
    int64_t _n = (n < 2) ? 1L : (fibonacci(n-2) + fibonacci(n-1));
    FREE_TIMEMORY_C_AUTO_TIMER(timer);
    return _n;
}

int main(int argc, char** argv)
{
    printf("%s @ %i\n", __FUNCTION__, __LINE__);

    void* timer = TIMEMORY_C_AUTO_TIMER("");
    int64_t n = fibonacci(45);
    printf("\nANSWER = %lli\n", n);
    FREE_TIMEMORY_C_AUTO_TIMER(timer);

    TIMEMORY_C_REPORT("test_c.txt");
    return 0;
}

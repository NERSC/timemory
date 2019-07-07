
#include "kernel.hpp"
#include "rep.h"
#include <cstdio>
#include <cstdlib>

#include <sys/time.h>

#define ERT_ALIGN 64
#define ERT_FLOP 128
#define ERT_TRIALS_MIN 100
#define ERT_WORKING_SET_MIN 10
#define ERT_MEMORY_MAX 64 * 64 * 64

double
getTime()
{
    double time;

#ifdef ERT_OPENMP
    time = omp_get_wtime();
#elif ERT_MPI
    time = MPI_Wtime();
#else
    struct timeval tm;
    gettimeofday(&tm, NULL);
    time = tm.tv_sec + (tm.tv_usec / 1000000.0);
#endif
    return time;
}

void
initialize(uint64_t nsize, double* __restrict__ A, double value)
{
#ifdef ERT_INTEL
    __assume_aligned(A, ERT_ALIGN);
#elif __xlC__
    __alignx(ERT_ALIGN, A);
#endif

    uint64_t i;
    for(i = 0; i < nsize; ++i)
    {
        A[i] = value;
    }
}

void
kernel(uint64_t nsize, uint64_t ntrials, double* __restrict__ A, int* bytes_per_elem,
       int* mem_accesses_per_elem)
{
    *bytes_per_elem        = sizeof(*A);
    *mem_accesses_per_elem = 2;

#ifdef ERT_INTEL
    __assume_aligned(A, ERT_ALIGN);
#elif __xlC__
    __alignx(ERT_ALIGN, A);
#endif

    double   alpha = 0.5;
    uint64_t i, j;
    for(j = 0; j < ntrials; ++j)
    {
#pragma unroll(8)
        for(i = 0; i < nsize; ++i)
        {
            double beta = 0.8;
#if defined(TEMPLATES)
#    if(ERT_FLOP & 1) == 1 /* add 1 flop */
            KERNEL1(beta, A[i], alpha);
#    else
            // divide by two here because macros halve, e.g. ERT_FLOP == 4 means 2 calls
            constexpr size_t N_FLOP = ERT_FLOP / 2;
            unroll<N_FLOP>(
                [](double& a, const double& b, const double& c) { a = a * b + c; }, beta,
                A[i], alpha);
#    endif
#elif defined(MACROS)
#    if(ERT_FLOP & 1) == 1  // add 1 flop
            KERNEL1(beta, A[i], alpha);
#    endif
#    if(ERT_FLOP & 2) == 2  // add 2 flops
            KERNEL2(beta, A[i], alpha);
#    endif
#    if(ERT_FLOP & 4) == 4  // add 4 flops
            REP2(KERNEL2(beta, A[i], alpha));
#    endif
#    if(ERT_FLOP & 8) == 8  // add 8 flops
            REP4(KERNEL2(beta, A[i], alpha));
#    endif
#    if(ERT_FLOP & 16) == 16  // add 16 flops
            REP8(KERNEL2(beta, A[i], alpha));
#    endif
#    if(ERT_FLOP & 32) == 32  // add 32 flops
            REP16(KERNEL2(beta, A[i], alpha));
#    endif
#    if(ERT_FLOP & 64) == 64  // add 64 flops
            REP32(KERNEL2(beta, A[i], alpha));
#    endif
#    if(ERT_FLOP & 128) == 128  // add 128 flops
            REP64(KERNEL2(beta, A[i], alpha));
#    endif
#    if(ERT_FLOP & 256) == 256  // add 256 flops
            REP128(KERNEL2(beta, A[i], alpha));
#    endif
#    if(ERT_FLOP & 512) == 512  // add 512 flops
            REP256(KERNEL2(beta, A[i], alpha));
#    endif
#    if(ERT_FLOP & 1024) == 1024  // add 1024 flops
            REP512(KERNEL2(beta, A[i], alpha));
#    endif
#endif
            A[i] = beta;
        }
        alpha = alpha * (1.0 - 1.0e-8);
    }
}

int
ert_main(int, char**)
{
    int rank     = 0;
    int nprocs   = 1;
    int nthreads = 1;
    int id       = 0;
#ifdef ERT_MPI
    int provided = -1;
    int requested;

#    ifdef ERT_OPENMP
    requested = MPI_THREAD_FUNNELED;
    MPI_Init_thread(&argc, &argv, requested, &provided);
#    else
    MPI_Init(&argc, &argv);
#    endif  // ERT_OPENMP

    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* printf("The MPI binding provided thread support of: %d\n", provided); */
#endif  // ERT_MPI

    uint64_t TSIZE = ERT_MEMORY_MAX;
    uint64_t PSIZE = TSIZE / nprocs;

#ifdef ERT_INTEL
    double* __restrict__ buf = (double*) _mm_malloc(PSIZE, ERT_ALIGN);
#else
    double* __restrict__ buf = (double*) malloc(PSIZE);
#endif

    if(buf == NULL)
    {
        fprintf(stderr, "Out of memory!\n");
        return -1;
    }

#ifdef ERT_OPENMP
#    pragma omp parallel private(id)
#endif

    {
#ifdef ERT_OPENMP
        id       = omp_get_thread_num();
        nthreads = omp_get_num_threads();
#else
        id       = 0;
        nthreads = 1;
#endif

        uint64_t nsize = PSIZE / nthreads;
        nsize          = nsize & (~(ERT_ALIGN - 1));
        nsize          = nsize / sizeof(double);
        uint64_t nid   = nsize * id;

        // initialize small chunck of buffer within each thread
        initialize(nsize, &buf[nid], 1.0);

        double   startTime, endTime;
        uint64_t n, nNew;
        uint64_t t;
        int      bytes_per_elem;
        int      mem_accesses_per_elem;

        n = ERT_WORKING_SET_MIN;
        while(n <= nsize)
        {  // working set - nsize
            uint64_t ntrials = nsize / n;
            if(ntrials < 1)
                ntrials = 1;

            for(t = ERT_TRIALS_MIN; t <= ntrials; t = t * 2)
            {  // working set - ntrials

#ifdef ERT_MPI
#    ifdef ERT_OPENMP
#        pragma omp master
#    endif
                {
                    MPI_Barrier(MPI_COMM_WORLD);
                }
#endif  // ERT_MPI

#ifdef ERT_OPENMP
#    pragma omp barrier
#endif
                comp_roof_t* rl = nullptr;

                if((id == 0) && (rank == 0))
                {
                    static int64_t nhash = __LINE__;
                    nhash += 1;
                    rl =
                        new comp_roof_t(tim::str::join("_", __FUNCTION__, n, t), true, 1);
                    startTime = getTime();
                    rl->start();
                }

                kernel(n, t, &buf[nid], &bytes_per_elem, &mem_accesses_per_elem);

#ifdef ERT_OPENMP
#    pragma omp barrier
#endif

#ifdef ERT_MPI
#    ifdef ERT_OPENMP
#        pragma omp master
#    endif
                {
                    MPI_Barrier(MPI_COMM_WORLD);
                }
#endif  // ERT_MPI

                if((id == 0) && (rank == 0))
                {
                    endTime = getTime();
                    rl->stop();
                    double   seconds          = (double) (endTime - startTime);
                    uint64_t working_set_size = n * nthreads * nprocs;
                    uint64_t total_bytes =
                        t * working_set_size * bytes_per_elem * mem_accesses_per_elem;
                    uint64_t total_flops = t * working_set_size * ERT_FLOP;

                    // nsize; trials; microseconds; bytes; single thread bandwidth; total
                    // bandwidth
                    float ert_flops_per_sec = (float) total_flops / seconds;
                    float tim_second = std::get<0>(*rl).get_elapsed() * tim::units::sec;
                    float tim_flops_per_sec = std::get<0>(*rl).get_counted() / tim_second;
                    float perc_error        = (tim_flops_per_sec - ert_flops_per_sec) /
                                       ert_flops_per_sec * 100.;
                    printf("%8" PRIu64 "%6" PRIu64 " %12.4e %10" PRIu64 " %10" PRIu64
                           " flops/sec: %8.0f (ERT), %8.0f (TiM), err: %5.2f %s\n",
                           working_set_size * bytes_per_elem, t, seconds,  //* 1000000,
                           total_bytes, total_flops, ert_flops_per_sec, tim_flops_per_sec,
                           perc_error, "%");
                    delete rl;
                }  // print

            }  // working set - ntrials

            nNew = 1.1 * n;
            if(nNew == n)
            {
                nNew = n + 1;
            }

            n = nNew;
        }  // working set - nsize

#if ERT_GPU
        cudaFree(d_buf);

        if(cudaGetLastError() != cudaSuccess)
        {
            printf("Last cuda error: %s\n", cudaGetErrorString(cudaGetLastError()));
        }

        cudaDeviceReset();
#endif
    }  // parallel region

#ifdef ERT_INTEL
    _mm_free(buf);
#else
    free(buf);
#endif

#ifdef ERT_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

#ifdef ERT_MPI
    MPI_Finalize();
#endif

    printf("\n");
    printf("META_DATA\n");
    printf("FLOPS          %d\n", ERT_FLOP);

#ifdef ERT_MPI
    printf("MPI_PROCS      %d\n", nprocs);
#endif

#ifdef ERT_OPENMP
    printf("OPENMP_THREADS %d\n", nthreads);
#endif

#ifdef ERT_GPU
    printf("GPU_BLOCKS     %d\n", gpu_blocks);
    printf("GPU_THREADS    %d\n", gpu_threads);
#endif

    return 0;
}

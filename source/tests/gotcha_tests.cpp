// MIT License
//
// Copyright (c) 2020, The Regents of the University of California,
// through Lawrence Berkeley National Laboratory (subject to receipt of any
// required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//

#include "gotcha_tests_lib.hpp"

#include "gtest/gtest.h"

#include "timemory/timemory.hpp"

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

using namespace tim::component;
using tim::component_tuple_t;

// create a hybrid for inside the gotcha
using gotcha_tuple_t = component_tuple_t<wall_clock, cpu_clock, peak_rss>;
using gotcha_list_t =
    tim::component_list_t<papi_array_t, cpu_roofline_sp_flops, cpu_roofline_dp_flops>;
using gotcha_hybrid_t = tim::auto_hybrid_t<gotcha_tuple_t, gotcha_list_t>;

// create gotcha types for various bundles of functions
using mpi_gotcha_t    = tim::component::gotcha<1, gotcha_hybrid_t>;
using work_gotcha_t   = tim::component::gotcha<1, gotcha_hybrid_t, int>;
using memfun_gotcha_t = tim::component::gotcha<5, gotcha_tuple_t>;

using malloc_gotcha_t = malloc_gotcha::gotcha_type<gotcha_tuple_t>;

TIMEMORY_DEFINE_CONCRETE_TRAIT(start_priority, mpi_gotcha_t, priority_constant<256>)
TIMEMORY_DEFINE_CONCRETE_TRAIT(start_priority, malloc_gotcha_t, priority_constant<512>)

TIMEMORY_DEFINE_CONCRETE_TRAIT(stop_priority, mpi_gotcha_t, priority_constant<-256>)
TIMEMORY_DEFINE_CONCRETE_TRAIT(stop_priority, malloc_gotcha_t, priority_constant<-512>)

using comp_t  = component_tuple_t<wall_clock, cpu_clock, peak_rss>;
using tuple_t = component_tuple_t<comp_t, mpi_gotcha_t, work_gotcha_t, memfun_gotcha_t>;
using list_t  = gotcha_list_t;
using auto_hybrid_t = tim::auto_hybrid_t<tuple_t, list_t>;

template <typename Tp>
using vector_t = std::vector<Tp>;

static constexpr int64_t nitr      = 100000;
static const double      tolerance = 1.0e-2;

static int    _argc = 0;
static char** _argv = nullptr;
namespace details
{
//--------------------------------------------------------------------------------------//
//  Get the current tests name
//
inline std::string
get_test_name()
{
    return ::testing::UnitTest::GetInstance()->current_test_info()->name();
}

//--------------------------------------------------------------------------------------//

template <typename Tp>
inline vector_t<Tp>
generate(const int64_t& nsize)
{
    std::vector<Tp> sendbuf(nsize, 0.0);
    std::mt19937    rng;
    rng.seed(54561434UL);
    auto dist = [&]() { return std::generate_canonical<Tp, 10>(rng); };
    std::generate(sendbuf.begin(), sendbuf.end(), [&]() { return dist(); });
    return sendbuf;
}

//--------------------------------------------------------------------------------------//

template <typename Tp>
inline void
generate(const int64_t& nsize, std::vector<Tp>& sendbuf)
{
    sendbuf.resize(nsize, 0.0);
    for(auto& itr : sendbuf)
        itr = 0.0;
    std::mt19937 rng;
    rng.seed(54561434UL);
    auto dist = [&]() { return std::generate_canonical<Tp, 10>(rng); };
    std::generate(sendbuf.begin(), sendbuf.end(), [&]() { return dist(); });
}

//--------------------------------------------------------------------------------------//

template <typename Tp>
inline vector_t<Tp>
allreduce(const vector_t<Tp>& sendbuf)
{
    vector_t<Tp> recvbuf(sendbuf.size(), 0.0);
#if defined(TIMEMORY_USE_MPI)
    auto dtype = (std::is_same<Tp, float>::value) ? MPI_FLOAT : MPI_DOUBLE;
    MPI_Allreduce(sendbuf.data(), recvbuf.data(), sendbuf.size(), dtype, MPI_SUM,
                  MPI_COMM_WORLD);
#else
    std::copy(sendbuf.begin(), sendbuf.end(), recvbuf.begin());
#endif
    return recvbuf;
}

//--------------------------------------------------------------------------------------//

template <typename Tp>
inline void
allreduce(const vector_t<Tp>& sendbuf, vector_t<Tp>& recvbuf)
{
    recvbuf.resize(sendbuf.size(), 0.0);
    for(auto& itr : recvbuf)
        itr = 0.0;
#if defined(TIMEMORY_USE_MPI)
    auto dtype = (std::is_same<Tp, float>::value) ? MPI_FLOAT : MPI_DOUBLE;
    MPI_Allreduce(sendbuf.data(), recvbuf.data(), sendbuf.size(), dtype, MPI_SUM,
                  MPI_COMM_WORLD);
#else
    std::copy(sendbuf.begin(), sendbuf.end(), recvbuf.begin());
#endif
}

//--------------------------------------------------------------------------------------//

}  // namespace details

//======================================================================================//

class gotcha_tests : public ::testing::Test
{
protected:
    void SetUp() override
    {
        static bool configured = false;
        if(!configured)
        {
            configured                    = true;
            tim::settings::width()        = 16;
            tim::settings::precision()    = 6;
            tim::settings::timing_units() = "sec";
            tim::settings::memory_units() = "kB";
            tim::settings::verbose()      = 0;
            tim::settings::debug()        = false;
            tim::settings::json_output()  = true;
            tim::settings::mpi_thread()   = false;
            tim::mpi::initialize(_argc, _argv);
#if defined(TIMEMORY_USE_PAPI)
            cpu_roofline_sp_flops::ert_config_type<float>::configure(1, 64);
            cpu_roofline_dp_flops::ert_config_type<double>::configure(1, 64);
#endif
            tim::settings::dart_output() = true;
            tim::settings::dart_count()  = 1;
            tim::settings::banner()      = false;
        }
    }
};

//======================================================================================//

TEST_F(gotcha_tests, mpi_explicit)
{
    mpi_gotcha_t::get_initializer() = [=]() {
        PRINT_HERE("%s", details::get_test_name().c_str());
#if defined(TIMEMORY_USE_MPI)
        mpi_gotcha_t::configure<0, int, const void*, void*, int, MPI_Datatype, MPI_Op,
                                MPI_Comm>("MPI_Allreduce");
#endif
    };

    TIMEMORY_BLANK_POINTER(auto_hybrid_t, details::get_test_name());

    float  fsum = 0.0;
    double dsum = 0.0;
    for(int i = 0; i < nitr; ++i)
    {
        auto fsendbuf = details::generate<float>(1000);
        auto frecvbuf = details::allreduce(fsendbuf);
        fsum += std::accumulate(frecvbuf.begin(), frecvbuf.end(), 0.0);

        auto dsendbuf = details::generate<double>(1000);
        auto drecvbuf = details::allreduce(dsendbuf);
        dsum += std::accumulate(drecvbuf.begin(), drecvbuf.end(), 0.0);
    }

    auto rank = tim::mpi::rank();
    auto size = tim::mpi::size();
    for(int i = 0; i < size; ++i)
    {
        tim::mpi::barrier();
        if(i == rank)
        {
            printf("\n");
            printf("[%i]> single-precision sum = %8.2f\n", rank, fsum);
            printf("[%i]> double-precision sum = %8.2f\n", rank, dsum);
        }
        tim::mpi::barrier();
    }
    tim::mpi::barrier();
    if(rank == 0)
        printf("\n");

    ASSERT_NEAR(fsum, 49892284.00 * size, tolerance);
    ASSERT_NEAR(dsum, 49868704.48 * size, tolerance);
}

//======================================================================================//

TEST_F(gotcha_tests, mpi_macro)
{
    mpi_gotcha_t::get_initializer() = [=]() {
        PRINT_HERE("%s", details::get_test_name().c_str());
#if defined(TIMEMORY_USE_MPI)
        TIMEMORY_C_GOTCHA(mpi_gotcha_t, 0, MPI_Allreduce);
#endif
    };

    TIMEMORY_BLANK_POINTER(auto_hybrid_t, details::get_test_name());

    float  fsum = 0.0;
    double dsum = 0.0;
    for(int i = 0; i < nitr; ++i)
    {
        auto fsendbuf = details::generate<float>(1000);
        auto frecvbuf = details::allreduce(fsendbuf);
        fsum += std::accumulate(frecvbuf.begin(), frecvbuf.end(), 0.0);

        auto dsendbuf = details::generate<double>(1000);
        auto drecvbuf = details::allreduce(dsendbuf);
        dsum += std::accumulate(drecvbuf.begin(), drecvbuf.end(), 0.0);
    }

    auto rank = tim::mpi::rank();
    auto size = tim::mpi::size();
    for(int i = 0; i < size; ++i)
    {
        tim::mpi::barrier();
        if(i == rank)
        {
            printf("\n");
            printf("[%i]> single-precision sum = %8.2f\n", rank, fsum);
            printf("[%i]> double-precision sum = %8.2f\n", rank, dsum);
        }
        tim::mpi::barrier();
    }
    tim::mpi::barrier();
    if(rank == 0)
        printf("\n");

    ASSERT_NEAR(fsum, 49892284.00 * size, tolerance);
    ASSERT_NEAR(dsum, 49868704.48 * size, tolerance);
}

//======================================================================================//

TEST_F(gotcha_tests, work_explicit)
{
    using tuple_type = std::tuple<float, double>;
    using pair_type  = std::pair<float, double>;

    work_gotcha_t::get_initializer() = [=]() {
        PRINT_HERE("%s", details::get_test_name().c_str());
        auto mangled_do_work = tim::mangle<decltype(ext::do_work)>("ext::do_work");
        work_gotcha_t::configure<0, tuple_type, int64_t, const pair_type&>(
            mangled_do_work);
    };

    TIMEMORY_BLANK_POINTER(auto_hybrid_t, details::get_test_name());

    float  fsum = 0.0;
    double dsum = 0.0;
    for(int i = 0; i < nitr; ++i)
    {
        auto ret = ext::do_work(1000, pair_type(0.25, 0.125));
        fsum += std::get<0>(ret);
        dsum += std::get<1>(ret);
    }

    auto rank = tim::mpi::rank();
    auto size = tim::mpi::size();
    for(int i = 0; i < size; ++i)
    {
        tim::mpi::barrier();
        if(i == rank)
        {
            printf("\n");
            printf("[%i]> single-precision sum = %8.2f\n", rank, fsum);
            printf("[%i]> double-precision sum = %8.2f\n", rank, dsum);
        }
        tim::mpi::barrier();
    }
    tim::mpi::barrier();
    if(rank == 0)
        printf("\n");

    ASSERT_NEAR(fsum, -2416347.50, tolerance);
    ASSERT_NEAR(dsum, -1829370.79, tolerance);
}

//======================================================================================//

TEST_F(gotcha_tests, work_macro)
{
    using pair_type = std::pair<float, double>;

    work_gotcha_t::get_initializer() = [=]() {
        PRINT_HERE("%s", details::get_test_name().c_str());
        TIMEMORY_CXX_GOTCHA(work_gotcha_t, 0, ext::do_work);
    };

    auto _exec = [&]() {
        TIMEMORY_BLANK_POINTER(auto_hybrid_t, details::get_test_name());

        float  fsum = 0.0;
        double dsum = 0.0;
        for(int i = 0; i < nitr; ++i)
        {
            auto ret = ext::do_work(1000, pair_type(0.25, 0.125));
            fsum += std::get<0>(ret);
            dsum += std::get<1>(ret);
        }

        auto rank = tim::mpi::rank();
        auto size = tim::mpi::size();
        for(int i = 0; i < size; ++i)
        {
            tim::mpi::barrier();
            if(i == rank)
            {
                printf("\n");
                printf("[%i]> single-precision sum = %8.2f\n", rank, fsum);
                printf("[%i]> double-precision sum = %8.2f\n", rank, dsum);
            }
            tim::mpi::barrier();
        }
        tim::mpi::barrier();
        if(rank == 0)
            printf("\n");

        ASSERT_NEAR(fsum, -2416347.50, tolerance);
        ASSERT_NEAR(dsum, -1829370.79, tolerance);
    };

    for(auto i = 0; i < 4; ++i)
        _exec();
}

//======================================================================================//

template <typename func_t>
void
print_func_info(const std::string& fname)
{
    using ret_type = typename tim::function_traits<func_t>::result_type;
    using arg_type = typename tim::function_traits<func_t>::args_type;
    std::cout << std::endl;
    std::cout << "  func name = " << fname << std::endl;
    std::cout << "memfun type = " << tim::demangle(typeid(func_t).name()) << std::endl;
    std::cout << "result type = " << tim::demangle(typeid(ret_type).name()) << std::endl;
    std::cout << "  args type = " << tim::demangle(typeid(arg_type).name()) << std::endl;
    std::cout << std::endl;
}

//======================================================================================//

TEST_F(gotcha_tests, malloc_gotcha)
{
    using toolset_t = tim::auto_tuple_t<gotcha_tuple_t, malloc_gotcha_t, mpi_gotcha_t>;

    malloc_gotcha::configure<gotcha_tuple_t>();

    mpi_gotcha_t::get_initializer() = [=]() {
#if defined(TIMEMORY_USE_MPI)
        TIMEMORY_C_GOTCHA(mpi_gotcha_t, 0, MPI_Allreduce);
#endif
    };

    PRINT_HERE("%s", "starting");
    toolset_t tool(details::get_test_name());

    float  fsum = 0.0;
    double dsum = 0.0;
    {
        for(int i = 0; i < nitr / 10; ++i)
        {
            std::vector<float>  fsendbuf, frecvbuf;
            std::vector<double> dsendbuf, drecvbuf;

            details::generate<float>(1000, fsendbuf);
            details::allreduce(fsendbuf, frecvbuf);
            fsum += std::accumulate(frecvbuf.begin(), frecvbuf.end(), 0.0);

            details::generate<double>(1000, dsendbuf);
            details::allreduce(dsendbuf, drecvbuf);
            dsum += std::accumulate(drecvbuf.begin(), drecvbuf.end(), 0.0);
        }
    }

    tool.stop();
    PRINT_HERE("%s", "stopped");

    malloc_gotcha_t& mc = *tool.get<malloc_gotcha_t>();
    std::cout << mc << std::endl;

    auto rank = tim::mpi::rank();
    auto size = tim::mpi::size();
    for(int i = 0; i < size; ++i)
    {
        tim::mpi::barrier();
        if(i == rank)
        {
            printf("\n");
            printf("[%i]> single-precision sum = %8.2f\n", rank, fsum);
            printf("[%i]> double-precision sum = %8.2f\n", rank, dsum);
        }
        tim::mpi::barrier();
    }
    tim::mpi::barrier();
    if(rank == 0)
        printf("\n");

    ASSERT_NEAR(fsum, 4986708.50 * size, tolerance);
    ASSERT_NEAR(dsum, 4986870.45 * size, tolerance);
}

//======================================================================================//

TEST_F(gotcha_tests, member_functions)
{
    using pair_type     = std::pair<float, double>;
    auto real_storage   = tim::storage<wall_clock>::instance();
    auto real_init_size = real_storage->size();
    printf("[initial]> wall-clock storage size: %li\n", (long int) real_init_size);

    memfun_gotcha_t::get_default_ready() = true;
    memfun_gotcha_t::get_initializer()   = [=]() {
        PRINT_HERE("%s", details::get_test_name().c_str());

        {
            using func_t = decltype(&DoWork::get);
            print_func_info<func_t>(TIMEMORY_STRINGIZE(DoWork::get));

            TIMEMORY_CXX_GOTCHA(memfun_gotcha_t, 0, &DoWork::get);
        }
        {
            using func_t = decltype(&DoWork::execute_fp4);
            print_func_info<func_t>(TIMEMORY_STRINGIZE(DoWork::execute_fp4));

            TIMEMORY_CXX_GOTCHA_MEMFUN(memfun_gotcha_t, 1, DoWork::execute_fp4);
        }
        {
            using func_t = decltype(&DoWork::execute_fp8);
            print_func_info<func_t>(TIMEMORY_STRINGIZE(DoWork::execute_fp8));

            TIMEMORY_CXX_GOTCHA(memfun_gotcha_t, 2, &DoWork::execute_fp8);
        }
        {
            using func_t = decltype(&DoWork::execute_fp);
            print_func_info<func_t>(TIMEMORY_STRINGIZE(DoWork::execute_fp));

            TIMEMORY_CXX_GOTCHA(memfun_gotcha_t, 3, &DoWork::execute_fp);
        }
    };

    float  fsum = 0.0;
    double dsum = 0.0;
    {
        TIMEMORY_BLANK_POINTER(auto_hybrid_t, details::get_test_name());

        DoWork dw(pair_type(0.25, 0.5));

        auto    _nitr = nitr / 10;
        int64_t ntot  = 0;
        for(int i = 0; i < _nitr; i += 10)
        {
            ntot += 10;
            if(i >= (_nitr - 10))
            {
                for(int j = 0; j < 10; ++j)
                {
                    dw.execute_fp4(1000);
                    dw.execute_fp8(1000);
                    auto ret = dw.get();
                    fsum += std::get<0>(ret);
                    dsum += std::get<1>(ret);
                }
            }
            else
            {
                auto _fp4 = [&]() {
                    for(int j = 0; j < 10; ++j)
                    {
                        dw.execute_fp4(1000);
                        auto ret = dw.get();
                        fsum += std::get<0>(ret);
                    }
                };

                auto _fp8 = [&]() {
                    for(int j = 0; j < 10; ++j)
                    {
                        dw.execute_fp8(1000);
                        auto ret = dw.get();
                        dsum += std::get<1>(ret);
                    }
                };

                std::thread t4(_fp4);
                std::thread t8(_fp8);

                t4.join();
                t8.join();
            }
        }

        int rank = tim::dmp::rank();
        if(rank == 0)
        {
            printf("\n");
            printf("[%i]> single-precision sum = %8.2f\n", rank, fsum);
            printf("[%i]> double-precision sum = %8.2f\n", rank, dsum);
        }

        float  fsum2 = 0.0;
        double dsum2 = 0.0;
        for(int64_t i = 0; i < ntot; ++i)
        {
            dw.execute_fp(1000, { 0.25 }, { 0.5 });
            auto ret = dw.get();
            fsum2 += std::get<0>(ret);
            dsum2 += std::get<1>(ret);
        }

        if(rank == 0)
        {
            printf("\n");
            printf("[%i]> single-precision sum2 = %8.2f\n", rank, fsum2);
            printf("[%i]> double-precision sum2 = %8.2f\n", rank, dsum2);
        }

        ASSERT_NEAR(fsum2, fsum, tolerance);
        ASSERT_NEAR(dsum2, dsum, tolerance);
    }

    auto rank = tim::mpi::rank();
    auto size = tim::mpi::size();
    for(int i = 0; i < size; ++i)
    {
        tim::mpi::barrier();
        if(i == rank)
        {
            printf("\n");
            printf("[%i]> single-precision sum = %8.2f\n", rank, fsum);
            printf("[%i]> double-precision sum = %8.2f\n", rank, dsum);
        }
        tim::mpi::barrier();
    }
    tim::mpi::barrier();
    if(rank == 0)
        printf("\n");

    auto real_final_size = real_storage->get().size();
    printf("[final]> wall-clock storage size: %li\n", (long int) real_final_size);

    ASSERT_NEAR(fsum, -241718.61, tolerance);
    ASSERT_NEAR(dsum, +88155.09, tolerance);
    ASSERT_EQ(real_final_size, 5 + real_init_size);
}

//======================================================================================//

#if defined(TIMEMORY_USE_MPI) && defined(REALLY_LONG_COMPILE_TIME)

//======================================================================================//

TEST_F(gotcha_tests, mpip)
{
    using namespace tim::component;
    using mpi_toolset_t = tim::auto_tuple_t<wall_clock, cpu_clock>;
    using mpip_gotcha_t = tim::component::gotcha<337, mpi_toolset_t>;
    using mpip_tuple_t  = tim::auto_tuple_t<wall_clock, mpip_gotcha_t>;

    auto init_mpip = []() {
        mpip_gotcha_t::get_initializer() = []() {
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 0, MPI_Send);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 1, MPI_Recv);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 2, MPI_Get_count);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 3, MPI_Bsend);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 4, MPI_Ssend);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 5, MPI_Rsend);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 6, MPI_Buffer_attach);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 7, MPI_Buffer_detach);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 8, MPI_Isend);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 9, MPI_Ibsend);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 10, MPI_Issend);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 11, MPI_Irsend);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 12, MPI_Irecv);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 13, MPI_Wait);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 14, MPI_Test);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 15, MPI_Request_free);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 16, MPI_Waitany);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 17, MPI_Testany);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 18, MPI_Waitall);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 19, MPI_Testall);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 20, MPI_Waitsome);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 21, MPI_Testsome);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 22, MPI_Iprobe);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 23, MPI_Probe);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 24, MPI_Cancel);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 25, MPI_Test_cancelled);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 26, MPI_Send_init);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 27, MPI_Bsend_init);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 28, MPI_Ssend_init);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 29, MPI_Rsend_init);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 30, MPI_Recv_init);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 31, MPI_Start);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 32, MPI_Startall);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 33, MPI_Sendrecv);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 34, MPI_Sendrecv_replace);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 35, MPI_Type_contiguous);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 36, MPI_Type_vector);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 37, MPI_Type_hvector);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 38, MPI_Type_indexed);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 39, MPI_Type_hindexed);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 40, MPI_Type_struct);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 41, MPI_Address);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 42, MPI_Type_extent);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 43, MPI_Type_size);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 44, MPI_Type_lb);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 45, MPI_Type_ub);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 46, MPI_Type_commit);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 47, MPI_Type_free);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 48, MPI_Get_elements);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 49, MPI_Pack);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 50, MPI_Unpack);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 51, MPI_Pack_size);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 52, MPI_Barrier);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 53, MPI_Bcast);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 54, MPI_Gather);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 55, MPI_Gatherv);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 56, MPI_Scatter);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 57, MPI_Scatterv);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 58, MPI_Allgather);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 59, MPI_Allgatherv);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 60, MPI_Alltoall);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 61, MPI_Alltoallv);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 62, MPI_Alltoallw);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 63, MPI_Exscan);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 64, MPI_Reduce);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 65, MPI_Op_create);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 66, MPI_Op_free);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 67, MPI_Allreduce);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 68, MPI_Reduce_scatter);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 69, MPI_Scan);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 70, MPI_Group_size);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 71, MPI_Group_rank);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 72, MPI_Group_translate_ranks);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 73, MPI_Group_compare);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 74, MPI_Comm_group);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 75, MPI_Group_union);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 76, MPI_Group_intersection);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 77, MPI_Group_difference);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 78, MPI_Group_incl);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 79, MPI_Group_excl);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 80, MPI_Group_range_incl);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 81, MPI_Group_range_excl);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 82, MPI_Group_free);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 83, MPI_Comm_size);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 84, MPI_Comm_rank);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 85, MPI_Comm_compare);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 86, MPI_Comm_dup);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 87, MPI_Comm_dup_with_info);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 88, MPI_Comm_create);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 89, MPI_Comm_split);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 90, MPI_Comm_free);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 91, MPI_Comm_test_inter);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 92, MPI_Comm_remote_size);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 93, MPI_Comm_remote_group);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 94, MPI_Intercomm_create);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 95, MPI_Intercomm_merge);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 96, MPI_Keyval_create);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 97, MPI_Keyval_free);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 98, MPI_Attr_put);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 99, MPI_Attr_get);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 100, MPI_Attr_delete);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 101, MPI_Topo_test);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 102, MPI_Cart_create);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 103, MPI_Dims_create);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 104, MPI_Graph_create);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 105, MPI_Graphdims_get);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 106, MPI_Graph_get);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 107, MPI_Cartdim_get);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 108, MPI_Cart_get);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 109, MPI_Cart_rank);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 110, MPI_Cart_coords);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 111, MPI_Graph_neighbors_count);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 112, MPI_Graph_neighbors);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 113, MPI_Cart_shift);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 114, MPI_Cart_sub);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 115, MPI_Cart_map);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 116, MPI_Graph_map);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 117, MPI_Get_processor_name);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 118, MPI_Get_version);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 119, MPI_Get_library_version);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 120, MPI_Errhandler_create);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 121, MPI_Errhandler_set);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 122, MPI_Errhandler_get);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 123, MPI_Errhandler_free);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 124, MPI_Error_string);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 125, MPI_Error_class);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 126, MPI_Init);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 127, MPI_Finalize);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 128, MPI_Initialized);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 129, MPI_Abort);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 130, MPI_DUP_FN);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 131, MPI_Close_port);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 132, MPI_Comm_accept);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 133, MPI_Comm_connect);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 134, MPI_Comm_disconnect);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 135, MPI_Comm_get_parent);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 136, MPI_Comm_join);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 137, MPI_Comm_spawn);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 138, MPI_Comm_spawn_multiple);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 139, MPI_Lookup_name);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 140, MPI_Open_port);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 141, MPI_Publish_name);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 142, MPI_Unpublish_name);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 143, MPI_Comm_set_info);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 144, MPI_Comm_get_info);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 145, MPI_Accumulate);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 146, MPI_Get);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 147, MPI_Put);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 148, MPI_Win_complete);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 149, MPI_Win_create);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 150, MPI_Win_fence);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 151, MPI_Win_free);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 152, MPI_Win_get_group);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 153, MPI_Win_lock);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 154, MPI_Win_post);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 155, MPI_Win_start);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 156, MPI_Win_test);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 157, MPI_Win_unlock);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 158, MPI_Win_wait);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 159, MPI_Win_allocate);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 160, MPI_Win_allocate_shared);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 161, MPI_Win_shared_query);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 162, MPI_Win_create_dynamic);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 163, MPI_Win_attach);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 164, MPI_Win_detach);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 165, MPI_Win_get_info);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 166, MPI_Win_set_info);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 167, MPI_Get_accumulate);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 168, MPI_Fetch_and_op);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 169, MPI_Compare_and_swap);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 170, MPI_Rput);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 171, MPI_Rget);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 172, MPI_Raccumulate);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 173, MPI_Rget_accumulate);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 174, MPI_Win_lock_all);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 175, MPI_Win_unlock_all);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 176, MPI_Win_flush);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 177, MPI_Win_flush_all);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 178, MPI_Win_flush_local);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 179, MPI_Win_flush_local_all);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 180, MPI_Win_sync);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 181, MPI_Add_error_class);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 182, MPI_Add_error_code);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 183, MPI_Add_error_string);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 184, MPI_Comm_call_errhandler);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 185, MPI_Comm_create_keyval);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 186, MPI_Comm_delete_attr);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 187, MPI_Comm_free_keyval);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 188, MPI_Comm_get_attr);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 189, MPI_Comm_get_name);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 190, MPI_Comm_set_attr);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 191, MPI_Comm_set_name);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 192, MPI_File_call_errhandler);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 193, MPI_Grequest_complete);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 194, MPI_Grequest_start);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 195, MPI_Init_thread);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 196, MPI_Is_thread_main);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 197, MPI_Query_thread);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 198, MPI_Status_set_cancelled);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 199, MPI_Status_set_elements);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 200, MPI_Type_create_keyval);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 201, MPI_Type_delete_attr);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 202, MPI_Type_dup);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 203, MPI_Type_free_keyval);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 204, MPI_Type_get_attr);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 205, MPI_Type_get_contents);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 206, MPI_Type_get_envelope);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 207, MPI_Type_get_name);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 208, MPI_Type_set_attr);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 209, MPI_Type_set_name);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 210, MPI_Type_match_size);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 211, MPI_Win_call_errhandler);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 212, MPI_Win_create_keyval);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 213, MPI_Win_delete_attr);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 214, MPI_Win_free_keyval);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 215, MPI_Win_get_attr);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 216, MPI_Win_get_name);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 217, MPI_Win_set_attr);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 218, MPI_Win_set_name);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 219, MPI_Alloc_mem);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 220, MPI_Comm_create_errhandler);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 221, MPI_Comm_get_errhandler);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 222, MPI_Comm_set_errhandler);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 223, MPI_File_create_errhandler);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 224, MPI_File_get_errhandler);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 225, MPI_File_set_errhandler);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 226, MPI_Finalized);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 227, MPI_Free_mem);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 228, MPI_Get_address);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 229, MPI_Info_create);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 230, MPI_Info_delete);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 231, MPI_Info_dup);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 232, MPI_Info_free);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 233, MPI_Info_get);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 234, MPI_Info_get_nkeys);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 235, MPI_Info_get_nthkey);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 236, MPI_Info_get_valuelen);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 237, MPI_Info_set);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 238, MPI_Pack_external);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 239, MPI_Pack_external_size);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 240, MPI_Request_get_status);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 241, MPI_Status_c2f);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 242, MPI_Status_f2c);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 243, MPI_Type_create_darray);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 244, MPI_Type_create_hindexed);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 245, MPI_Type_create_hvector);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 246, MPI_Type_create_indexed_block);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 247, MPI_Type_create_hindexed_block);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 248, MPI_Type_create_resized);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 249, MPI_Type_create_struct);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 250, MPI_Type_create_subarray);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 251, MPI_Type_get_extent);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 252, MPI_Type_get_true_extent);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 253, MPI_Unpack_external);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 254, MPI_Win_create_errhandler);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 255, MPI_Win_get_errhandler);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 256, MPI_Win_set_errhandler);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 257, MPI_Type_create_f90_integer);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 258, MPI_Type_create_f90_real);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 259, MPI_Type_create_f90_complex);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 260, MPI_Reduce_local);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 261, MPI_Op_commutative);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 262, MPI_Reduce_scatter_block);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 263, MPI_Dist_graph_create_adjacent);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 264, MPI_Dist_graph_create);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 265, MPI_Dist_graph_neighbors_count);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 266, MPI_Dist_graph_neighbors);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 267, MPI_Improbe);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 268, MPI_Imrecv);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 269, MPI_Mprobe);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 270, MPI_Mrecv);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 271, MPI_Comm_idup);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 272, MPI_Ibarrier);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 273, MPI_Ibcast);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 274, MPI_Igather);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 275, MPI_Igatherv);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 276, MPI_Iscatter);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 277, MPI_Iscatterv);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 278, MPI_Iallgather);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 279, MPI_Iallgatherv);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 280, MPI_Ialltoall);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 281, MPI_Ialltoallv);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 282, MPI_Ialltoallw);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 283, MPI_Ireduce);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 284, MPI_Iallreduce);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 285, MPI_Ireduce_scatter);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 286, MPI_Ireduce_scatter_block);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 287, MPI_Iscan);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 288, MPI_Iexscan);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 289, MPI_Ineighbor_allgather);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 290, MPI_Ineighbor_allgatherv);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 291, MPI_Ineighbor_alltoall);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 292, MPI_Ineighbor_alltoallv);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 293, MPI_Ineighbor_alltoallw);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 294, MPI_Neighbor_allgather);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 295, MPI_Neighbor_allgatherv);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 296, MPI_Neighbor_alltoall);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 297, MPI_Neighbor_alltoallv);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 298, MPI_Neighbor_alltoallw);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 299, MPI_Comm_split_type);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 300, MPI_Get_elements_x);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 301, MPI_Status_set_elements_x);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 302, MPI_Type_get_extent_x);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 303, MPI_Type_get_true_extent_x);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 304, MPI_Type_size_x);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 305, MPI_Comm_create_group);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 306, MPI_T_init_thread);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 307, MPI_T_finalize);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 308, MPI_T_enum_get_info);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 309, MPI_T_enum_get_item);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 310, MPI_T_cvar_get_num);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 311, MPI_T_cvar_get_info);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 312, MPI_T_cvar_handle_alloc);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 313, MPI_T_cvar_handle_free);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 314, MPI_T_cvar_read);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 315, MPI_T_cvar_write);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 316, MPI_T_pvar_get_num);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 317, MPI_T_pvar_get_info);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 318, MPI_T_pvar_session_create);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 319, MPI_T_pvar_session_free);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 320, MPI_T_pvar_handle_alloc);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 321, MPI_T_pvar_handle_free);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 322, MPI_T_pvar_start);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 323, MPI_T_pvar_stop);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 324, MPI_T_pvar_read);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 325, MPI_T_pvar_write);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 326, MPI_T_pvar_reset);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 327, MPI_T_pvar_readreset);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 328, MPI_T_category_get_num);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 329, MPI_T_category_get_info);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 330, MPI_T_category_get_cvars);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 331, MPI_T_category_get_pvars);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 332, MPI_T_category_get_categories);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 333, MPI_T_category_changed);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 334, MPI_T_cvar_get_index);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 335, MPI_T_pvar_get_index);
            TIMEMORY_C_GOTCHA(mpip_gotcha_t, 336, MPI_T_category_get_index);
        };
    };

    if(tim::get_env<bool>("INIT_MPIP_TOOLS", true))
        init_mpip();

    TIMEMORY_BLANK_POINTER(mpip_tuple_t, details::get_test_name());

    float  fsum = 0.0;
    double dsum = 0.0;
    for(int i = 0; i < nitr; ++i)
    {
        auto fsendbuf = details::generate<float>(1000);
        auto frecvbuf = details::allreduce(fsendbuf);
        fsum += std::accumulate(frecvbuf.begin(), frecvbuf.end(), 0.0);

        auto dsendbuf = details::generate<double>(1000);
        auto drecvbuf = details::allreduce(dsendbuf);
        dsum += std::accumulate(drecvbuf.begin(), drecvbuf.end(), 0.0);
    }

    auto rank = tim::mpi::rank();
    auto size = tim::mpi::size();
    for(int i = 0; i < size; ++i)
    {
        tim::mpi::barrier();
        if(i == rank)
        {
            printf("\n");
            printf("[%i]> single-precision sum = %8.2f\n", rank, fsum);
            printf("[%i]> double-precision sum = %8.2f\n", rank, dsum);
        }
        tim::mpi::barrier();
    }
    tim::mpi::barrier();
    if(rank == 0)
        printf("\n");

    ASSERT_NEAR(fsum, 49892284.00 * size, tolerance);
    ASSERT_NEAR(dsum, 49868704.48 * size, tolerance);
}

//======================================================================================//

#endif

//======================================================================================//

int
main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    _argc = argc;
    _argv = argv;

    tim::settings::verbose()     = 0;
    tim::settings::debug()       = false;
    tim::settings::json_output() = true;
    tim::timemory_init(&argc, &argv);
    tim::settings::dart_output() = true;
    tim::settings::dart_count()  = 1;
    tim::settings::banner()      = false;

    tim::settings::dart_type() = "peak_rss";
    // TIMEMORY_VARIADIC_BLANK_AUTO_TUPLE("PEAK_RSS", ::tim::component::peak_rss);
    auto ret = RUN_ALL_TESTS();

    tim::timemory_finalize();
    return ret;
}

//======================================================================================//

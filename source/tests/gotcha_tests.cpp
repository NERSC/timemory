// MIT License
//
// Copyright (c) 2019, The Regents of the University of California,
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

#include <timemory/timemory.hpp>

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

using namespace tim::component;
using tim::component_tuple;

// create a hybrid for inside the gotcha
using gotcha_tuple_t = component_tuple<real_clock, cpu_clock, peak_rss>;
using gotcha_list_t =
    tim::component_list<papi_array_t, cpu_roofline_sp_flops, cpu_roofline_dp_flops>;
using gotcha_hybrid_t = tim::auto_hybrid<gotcha_tuple_t, gotcha_list_t>;

// create gotcha types for various bundles of functions
using mpi_gotcha_t    = tim::component::gotcha<1, gotcha_hybrid_t>;
using work_gotcha_t   = tim::component::gotcha<1, gotcha_hybrid_t, int>;
using memfun_gotcha_t = tim::component::gotcha<3, gotcha_hybrid_t>;

using comp_t  = component_tuple<real_clock, cpu_clock, peak_rss>;
using tuple_t = component_tuple<comp_t, mpi_gotcha_t, work_gotcha_t, memfun_gotcha_t>;
using list_t  = gotcha_list_t;
using auto_hybrid_t = tim::auto_hybrid<tuple_t, list_t>;

template <typename _Tp>
using vector_t = std::vector<_Tp>;

static constexpr int64_t nitr      = 100000;
static const double      tolerance = 1.0e-2;

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

template <typename _Tp>
inline vector_t<_Tp>
generate(const int64_t& nsize)
{
    std::vector<_Tp> sendbuf(nsize, 0.0);
    std::mt19937     rng;
    rng.seed(54561434UL);
    auto dist = [&]() { return std::generate_canonical<_Tp, 10>(rng); };
    std::generate(sendbuf.begin(), sendbuf.end(), [&]() { return dist(); });
    return sendbuf;
}

//--------------------------------------------------------------------------------------//

template <typename _Tp>
inline vector_t<_Tp>
allreduce(const vector_t<_Tp>& sendbuf)
{
    vector_t<_Tp> recvbuf(sendbuf.size(), 0.0);
#if defined(TIMEMORY_USE_MPI)
    auto dtype = (std::is_same<_Tp, float>::value) ? MPI_FLOAT : MPI_DOUBLE;
    MPI_Allreduce(sendbuf.data(), recvbuf.data(), sendbuf.size(), dtype, MPI_SUM,
                  MPI_COMM_WORLD);
#else
    std::copy(sendbuf.begin(), sendbuf.end(), recvbuf.begin());
#endif
    return recvbuf;
}

//--------------------------------------------------------------------------------------//

}  // namespace details

//======================================================================================//

class gotcha_tests : public ::testing::Test
{
protected:
    void SetUp() override {}
};

//======================================================================================//

TEST_F(gotcha_tests, mpi_explicit)
{
    mpi_gotcha_t::get_initializer() = [=]() {
        PRINT_HERE(details::get_test_name().c_str());
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
        PRINT_HERE(details::get_test_name().c_str());
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
    using tuple_t = std::tuple<float, double>;
    using pair_t  = std::pair<float, double>;

    work_gotcha_t::get_initializer() = [=]() {
        PRINT_HERE(details::get_test_name().c_str());
        auto mangled_do_work = tim::mangle<decltype(ext::do_work)>("ext::do_work");
        work_gotcha_t::configure<0, tuple_t, int64_t, const pair_t&>(mangled_do_work);
    };

    TIMEMORY_BLANK_POINTER(auto_hybrid_t, details::get_test_name());

    float  fsum = 0.0;
    double dsum = 0.0;
    for(int i = 0; i < nitr; ++i)
    {
        auto ret = ext::do_work(1000, pair_t(0.25, 0.125));
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
    using pair_t = std::pair<float, double>;

    work_gotcha_t::get_initializer() = [=]() {
        PRINT_HERE(details::get_test_name().c_str());
        TIMEMORY_CXX_GOTCHA(work_gotcha_t, 0, ext::do_work);
    };

    TIMEMORY_BLANK_POINTER(auto_hybrid_t, details::get_test_name());

    float  fsum = 0.0;
    double dsum = 0.0;
    for(int i = 0; i < nitr; ++i)
    {
        auto ret = ext::do_work(1000, pair_t(0.25, 0.125));
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

TEST_F(gotcha_tests, member_functions)
{
    using pair_t = std::pair<float, double>;

    memfun_gotcha_t::get_initializer() = [=]() {
        PRINT_HERE(details::get_test_name().c_str());

        {
            using func_t         = decltype(&DoWork::get);
            constexpr size_t idx = 0;
            print_func_info<func_t>(TIMEMORY_STRINGIZE(DoWork::get));

            // auto func_name = tim::mangle<func_t>("DoWork::get");
            // memfun_gotcha_t::configure<idx, std::tuple<float, double>>(func_name);
            TIMEMORY_CXX_GOTCHA(memfun_gotcha_t, idx, &DoWork::get);
        }
        {
            using func_t         = decltype(&DoWork::execute_fp4);
            constexpr size_t idx = 1;
            print_func_info<func_t>(TIMEMORY_STRINGIZE(DoWork::execute_fp4));

            TIMEMORY_CXX_MEMFUN_GOTCHA(memfun_gotcha_t, idx, DoWork::execute_fp4);
        }
        {
            using func_t         = decltype(&DoWork::execute_fp8);
            constexpr size_t idx = 2;
            print_func_info<func_t>(TIMEMORY_STRINGIZE(DoWork::execute_fp8));

            TIMEMORY_CXX_GOTCHA(memfun_gotcha_t, idx, &DoWork::execute_fp8);
        }
    };

    TIMEMORY_BLANK_POINTER(auto_hybrid_t, details::get_test_name());

    float  fsum = 0.0;
    double dsum = 0.0;
    DoWork dw(pair_t(0.25, 0.5));

    for(int i = 0; i < nitr; ++i)
    {
        auto        _fp4 = [&]() { dw.execute_fp4(1000); };
        auto        _fp8 = [&]() { dw.execute_fp8(1000); };
        std::thread t4(_fp4);
        std::thread t8(_fp8);

        t4.join();
        t8.join();

        auto ret = dw.get();
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
    ASSERT_NEAR(dsum, 881550.95, tolerance);
}

//======================================================================================//

int
main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    tim::settings::width()        = 12;
    tim::settings::precision()    = 6;
    tim::settings::timing_units() = "sec";
    tim::settings::memory_units() = "kB";
    tim::settings::verbose()      = 0;
    tim::settings::debug()        = false;
    tim::settings::json_output()  = true;
    tim::timemory_init(argc, argv);  // parses environment, sets output paths
    tim::mpi::initialize(argc, argv);
    cpu_roofline_sp_flops::ert_config_type<float>::configure(1, 64);
    cpu_roofline_dp_flops::ert_config_type<double>::configure(1, 64);
    return RUN_ALL_TESTS();
}

//======================================================================================//

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
using gotcha_tuple_t  = component_tuple_t<wall_clock, cpu_clock, peak_rss>;
using gotcha_list_t   = tim::component_list_t<cpu_roofline_dp_flops>;
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
    return std::string(::testing::UnitTest::GetInstance()->current_test_suite()->name()) +
           "." + ::testing::UnitTest::GetInstance()->current_test_info()->name();
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
    static void SetUpTestSuite()
    {
        tim::settings::banner()       = false;
        tim::settings::width()        = 16;
        tim::settings::precision()    = 6;
        tim::settings::timing_units() = "sec";
        tim::settings::memory_units() = "kB";
        tim::settings::verbose()      = 0;
        tim::settings::debug()        = false;
        tim::settings::json_output()  = true;
        tim::settings::mpi_thread()   = false;
        tim::dmp::initialize(_argc, _argv);
        tim::timemory_init(&_argc, &_argv);
        tim::settings::dart_output() = true;
        tim::settings::dart_count()  = 1;
        tim::settings::dart_type()   = "peak_rss";
#if defined(TIMEMORY_USE_PAPI)
        cpu_roofline_sp_flops::ert_config_type<float>::configure(1, 64);
        cpu_roofline_dp_flops::ert_config_type<double>::configure(1, 64);
#endif
    }

    static void TearDownTestSuite()
    {
        tim::timemory_finalize();
        tim::dmp::finalize();
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
    using toolset_t = tim::auto_tuple_t<gotcha_tuple_t, malloc_gotcha, mpi_gotcha_t>;

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

    malloc_gotcha& mc = *tool.get<malloc_gotcha>();
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

TEST_F(gotcha_tests, void_function)
{
    auto _dbg              = tim::settings::debug();
    tim::settings::debug() = true;

    using puts_bundle_t = tim::component_tuple<trip_count>;
    using puts_gotcha_t = tim::component::gotcha<1, puts_bundle_t>;
    using void_bundle_t = tim::lightweight_tuple<puts_gotcha_t>;

    puts_gotcha_t::get_initializer() = [=]() {
        TIMEMORY_CXX_GOTCHA(puts_gotcha_t, 0, ext::do_puts);
    };

    auto _beg_ct = tim::storage<trip_count>::instance()->size();

    void_bundle_t _bundle{ details::get_test_name() };
    _bundle.start();
    for(int i = 0; i < 10; ++i)
        ext::do_puts(details::get_test_name().c_str());
    _bundle.stop();

    auto _end_ct = tim::storage<trip_count>::instance()->size();
    EXPECT_EQ(_end_ct - _beg_ct, 1) << " begin: " << _beg_ct << ", end: " << _end_ct;

    tim::settings::debug() = _dbg;
}

//======================================================================================//

namespace tim
{
namespace component
{
struct exp_intercept : public base<exp_intercept, void>
{
    static auto& get_intercepts()
    {
        static uint64_t _instance{ 0 };
        return _instance;
    }

    static auto& puts_intercepts()
    {
        static uint64_t _instance{ 0 };
        return _instance;
    }

    void operator()(const char* msg)
    {
        ++puts_intercepts();
        puts(msg);
    }

    double operator()(double val)
    {
        std::stringstream ss;
        ss << "computing exp(" << val << ")";
        ext::do_puts(ss.str().c_str());
        ++get_intercepts();
        return exp(val);
    }
};
}  // namespace component
}  // namespace tim

TEST_F(gotcha_tests, replacement)
{
    auto _dbg              = tim::settings::debug();
    tim::settings::debug() = true;

    using exp_intercept_t = gotcha<2, std::tuple<>, exp_intercept>;
    using exp_bundle_t    = tim::component_bundle<TIMEMORY_API, exp_intercept_t>;

    static_assert(exp_intercept_t::components_size == 0,
                  "exp_intercept_t should have no components");
    static_assert(exp_intercept_t::differentiator_is_component,
                  "exp_intercept_t won't replace exp");

    //
    // configure the initializer for the gotcha component which replaces exp with expf
    //
    exp_intercept_t::get_initializer() = []() {
        puts("Generating exp intercept...");
        TIMEMORY_C_GOTCHA(exp_intercept_t, 0, exp);
        TIMEMORY_CXX_GOTCHA(exp_intercept_t, 1, ext::do_puts);
    };

    {
        TIMEMORY_BLANK_MARKER(exp_bundle_t, details::get_test_name());
        double ret = 10.0;
        for(int i = 0; i < 10; ++i)
            ret += exp(ret);
        printf("result: %f\n", ret);
    }

    EXPECT_EQ(exp_intercept::get_intercepts(), 10);
    EXPECT_EQ(exp_intercept::puts_intercepts(), 10);

    tim::settings::debug() = _dbg;
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

#if defined(TIMEMORY_USE_MPI) && defined(TIMEMORY_USE_GOTCHA)

#    include "timemory/components/gotcha/mpip.hpp"

TEST_F(gotcha_tests, mpip)
{
    using api_t         = TIMEMORY_API;
    using mpi_toolset_t = tim::lightweight_tuple<trip_count>;
    using mpip_tuple_t =
        tim::component_tuple<trip_count, mpip_handle<mpi_toolset_t, api_t>>;

    configure_mpip<mpi_toolset_t, api_t>();
    auto ret = activate_mpip<mpi_toolset_t, api_t>();

    {
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

    deactivate_mpip<mpi_toolset_t, api_t>(ret);
}

#endif

//======================================================================================//

int
main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    _argc = argc;
    _argv = argv;
    return RUN_ALL_TESTS();
}

//======================================================================================//

TIMEMORY_INITIALIZE_STORAGE(mpi_gotcha_t)
TIMEMORY_INITIALIZE_STORAGE(work_gotcha_t)
TIMEMORY_INITIALIZE_STORAGE(memfun_gotcha_t)

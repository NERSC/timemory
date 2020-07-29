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

#include "gtest/gtest.h"

#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

#include "timemory/timemory.hpp"

using namespace tim::component;

static int    _argc = 0;
static char** _argv = nullptr;

using mutex_t        = std::mutex;
using lock_t         = std::unique_lock<mutex_t>;
using string_t       = std::string;
using stringstream_t = std::stringstream;
using floating_t     = double;
using cache_ptr_t    = std::shared_ptr<tim::rusage_cache>;
using bundle_t = tim::component_bundle<TIMEMORY_API, wall_clock, peak_rss, num_io_in,
                                       num_io_out, num_major_page_faults,
                                       num_minor_page_faults, priority_context_switch,
                                       voluntary_context_switch, current_peak_rss>;

static const int64_t niter        = 20;
static const int64_t nelements    = 0.95 * (tim::units::get_page_size() * 500);
static const auto    memory_unit  = std::pair<int64_t, string_t>(tim::units::KiB, "KiB");
static auto          cache        = std::array<cache_ptr_t, 2>{};
static auto          bundle       = std::shared_ptr<bundle_t>{};
static auto          cache_bundle = std::shared_ptr<bundle_t>{};
static auto          tot_size     = nelements * sizeof(int64_t) / memory_unit.first;
static const double  peak_tolerance = 5 * tim::units::MiB;

//--------------------------------------------------------------------------------------//

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

// this function consumes approximately "n" milliseconds of real time
inline void
do_sleep(long n)
{
    std::this_thread::sleep_for(std::chrono::milliseconds(n));
}

// this function consumes an unknown number of cpu resources
inline long
fibonacci(long n)
{
    return (n < 2) ? n : (fibonacci(n - 1) + fibonacci(n - 2));
}

// this function consumes approximately "t" milliseconds of cpu time
void
consume(long n)
{
    // a mutex held by one lock
    mutex_t mutex;
    // acquire lock
    lock_t hold_lk(mutex);
    // associate but defer
    lock_t try_lk(mutex, std::defer_lock);
    // get current time
    auto now = std::chrono::steady_clock::now();
    // try until time point
    while(std::chrono::steady_clock::now() < (now + std::chrono::milliseconds(n)))
        try_lk.try_lock();
}

// get a random entry from vector
template <typename Tp>
size_t
random_entry(const std::vector<Tp>& v)
{
    std::mt19937 rng;
    rng.seed(std::random_device()());
    std::uniform_int_distribution<std::mt19937::result_type> dist(0, v.size() - 1);
    return v.at(dist(rng));
}

void
allocate()
{
    if(!cache.at(0))
        cache.at(0) = std::make_shared<tim::rusage_cache>();

    if(!bundle)
        bundle = std::make_shared<bundle_t>("allocate",
                                            tim::quirk::config<tim::quirk::no_store>{});

    if(!cache_bundle)
        cache_bundle = std::make_shared<bundle_t>(
            "allocate", tim::quirk::config<tim::quirk::no_store>{});

    bundle->reset();
    bundle->start();
    cache_bundle->reset();
    cache_bundle->start(*cache.at(0));

    std::vector<int64_t> v(nelements, 15);
    auto                 ret  = fibonacci(0);
    long                 nfib = details::random_entry(v);
    for(int64_t i = 0; i < niter; ++i)
    {
        nfib = details::random_entry(v);
        ret += details::fibonacci(nfib);
    }

    if(ret < 0)
        printf("fibonacci(%li) * %li = %li\n", (long) nfib, (long) niter, ret);

    bundle->stop();
    cache_bundle->stop(*cache.at(0));
    if(!cache.at(1))
        cache.at(1) = std::make_shared<tim::rusage_cache>();
}

}  // namespace details

//--------------------------------------------------------------------------------------//

class cache_tests : public ::testing::Test
{
protected:
    void SetUp() override
    {
        static bool configured = false;
        if(!configured)
        {
            configured                   = true;
            tim::settings::verbose()     = 0;
            tim::settings::debug()       = false;
            tim::settings::json_output() = true;
            tim::settings::mpi_thread()  = false;
            tim::mpi::initialize(_argc, _argv);
            tim::timemory_init(_argc, _argv);
            tim::settings::dart_output() = true;
            tim::settings::dart_count()  = 1;
            tim::settings::banner()      = false;

            // preform allocation only once here
            details::allocate();
        }
        puts("");
        printf("##### Executing %s ... #####\n", details::get_test_name().c_str());
        puts("");
    }

    void TearDown() override { puts(""); }
};

//--------------------------------------------------------------------------------------//

template <typename Tp>
auto
run1()
{
    using trait_type                = typename tim::trait::cache<Tp>::type;
    static constexpr bool is_null_v = tim::concepts::is_null_type<trait_type>::value;
    using type =
        tim::conditional_t<(is_null_v), tim::type_list<>, tim::type_list<trait_type>>;
    using ttype     = tim::convert_t<tim::type_concat_t<type>, std::tuple<>>;
    using uniq_type = tim::unique_t<type, std::tuple<>>;
    std::cout << "\ttype           : " << tim::demangle<Tp>() << std::endl;
    std::cout << "\ttrait type     : " << tim::demangle<trait_type>() << std::endl;
    std::cout << "\ttrait is null  : " << std::boolalpha << is_null_v << std::endl;
    std::cout << "\ttype list type : " << tim::demangle<type>() << std::endl;
    std::cout << "\ttuple type     : " << tim::demangle<ttype>() << std::endl;
    std::cout << "\tunique type    : " << tim::demangle<uniq_type>() << std::endl;
    return uniq_type{};
}

//--------------------------------------------------------------------------------------//

template <typename Tp>
auto
run2()
{
    using trait_type =
        typename tim::impl::get_trait_type_tuple<tim::trait::cache, Tp>::trait_type;
    static constexpr bool is_null_v =
        tim::impl::get_trait_type_tuple<tim::trait::cache, Tp>::is_null_v;
    using type      = tim::impl::get_trait_type_tuple_t<tim::trait::cache, Tp>;
    using ttype     = tim::convert_t<tim::type_concat_t<type>, std::tuple<>>;
    using uniq_type = tim::unique_t<type, std::tuple<>>;
    std::cout << "\ttype           : " << tim::demangle<Tp>() << std::endl;
    std::cout << "\ttrait type     : " << tim::demangle<trait_type>() << std::endl;
    std::cout << "\ttrait is null  : " << std::boolalpha << is_null_v << std::endl;
    std::cout << "\ttype list type : " << tim::demangle<type>() << std::endl;
    std::cout << "\ttuple type     : " << tim::demangle<ttype>() << std::endl;
    std::cout << "\tunique type    : " << tim::demangle<uniq_type>() << std::endl;
    return uniq_type{};
}

//--------------------------------------------------------------------------------------//

template <typename... Tp>
auto
run3()
{
    using namespace tim::stl;
    using namespace tim::stl::ostream;

    using trait_type =
        typename tim::impl::get_trait_type<tim::trait::cache, Tp...>::trait_type;
    using type      = tim::impl::get_trait_type_t<tim::trait::cache, Tp...>;
    using ttype     = tim::convert_t<type, std::tuple<>>;
    using uniq_type = tim::unique_t<type, std::tuple<>>;
    std::cout << "\ttype           : " << tim::demangle<std::tuple<Tp...>>() << std::endl;
    std::cout << "\ttrait type     : " << tim::demangle<trait_type>() << std::endl;
    std::cout << "\ttype list type : " << tim::demangle<type>() << std::endl;
    std::cout << "\ttuple type     : " << tim::demangle<ttype>() << std::endl;
    std::cout << "\tunique type    : " << tim::demangle<uniq_type>() << std::endl;
    return uniq_type{};
}

//--------------------------------------------------------------------------------------//

void
print_rusage_cache(const tim::rusage_cache& _rusage)
{
    std::cout << "\tpeak_rss          : " << _rusage.get_peak_rss() << '\n';
    std::cout << "\tkernel mode time  : " << _rusage.get_kernel_mode_time() << '\n';
    std::cout << "\tuser mode time    : " << _rusage.get_user_mode_time() << '\n';
    std::cout << "\tio in             : " << _rusage.get_num_io_in() << '\n';
    std::cout << "\tio out            : " << _rusage.get_num_io_out() << '\n';
    std::cout << "\tmajor page faults : " << _rusage.get_num_major_page_faults() << '\n';
    std::cout << "\tminor page faults : " << _rusage.get_num_minor_page_faults() << '\n';
    std::cout << "\tprio ctx switch   : " << _rusage.get_num_priority_context_switch()
              << '\n';
    std::cout << "\tvol ctx switch    : " << _rusage.get_num_voluntary_context_switch()
              << '\n';
}

//--------------------------------------------------------------------------------------//

TEST_F(cache_tests, peak_rss)
{
    auto ret1 = run1<peak_rss>();
    puts("");
    auto ret2 = run2<peak_rss>();

    auto check1 = std::is_same<decltype(ret1), std::tuple<tim::rusage_cache>>::value;
    auto check2 = std::is_same<decltype(ret2), std::tuple<tim::rusage_cache>>::value;
    EXPECT_TRUE(check1);
    EXPECT_TRUE(check2);
}

//--------------------------------------------------------------------------------------//

TEST_F(cache_tests, wall_clock)
{
    auto ret1 = run1<wall_clock>();
    puts("");
    auto ret2   = run2<wall_clock>();
    auto check1 = std::is_same<decltype(ret1), std::tuple<>>::value;
    auto check2 = std::is_same<decltype(ret2), std::tuple<>>::value;
    EXPECT_TRUE(check1);
    EXPECT_TRUE(check2);
}

//--------------------------------------------------------------------------------------//

TEST_F(cache_tests, peak_rss_wall_clock)
{
    auto ret   = run3<peak_rss, wall_clock>();
    auto check = std::is_same<decltype(ret), std::tuple<tim::rusage_cache>>::value;
    EXPECT_TRUE(check);
}

//--------------------------------------------------------------------------------------//

TEST_F(cache_tests, rusage)
{
    using data_type =
        std::tuple<peak_rss, current_peak_rss, num_io_in, num_io_out, wall_clock>;
    using trait_type    = tim::get_trait_type_t<tim::trait::cache, data_type>;
    using cache_type    = typename tim::operation::construct_cache<data_type>::type;
    auto        _cache  = tim::invoke::get_cache<data_type>();
    const auto& _rusage = std::get<0>(_cache);

    std::cout << "trait type        : " << tim::demangle<trait_type>() << std::endl;
    std::cout << "cache type        : " << tim::demangle<cache_type>() << std::endl;
    std::cout << "cache value       : " << tim::demangle<decltype(_cache)>() << std::endl;

    print_rusage_cache(_rusage);
    auto check = std::is_same<decltype(_cache), std::tuple<tim::rusage_cache>>::value;
    EXPECT_TRUE(check);
}

//--------------------------------------------------------------------------------------//

TEST_F(cache_tests, validation)
{
    puts("\n>>> INITIAL CACHE <<<\n");
    print_rusage_cache(*cache.at(0));
    puts("\n>>> BUNDLE <<<\n");
    std::cout << *bundle << std::endl;
    puts("\n>>> CACHE BUNDLE <<<\n");
    std::cout << *cache_bundle << std::endl;
    puts("\n>>> FINAL CACHE <<<\n");
    print_rusage_cache(*cache.at(1));

    EXPECT_NEAR(tot_size, bundle->get<peak_rss>()->get(), peak_tolerance);
    EXPECT_NEAR(std::get<0>(bundle->get<current_peak_rss>()->get()),
                cache.at(0)->get_peak_rss() / memory_unit.first, peak_tolerance);
    EXPECT_NEAR(std::get<1>(bundle->get<current_peak_rss>()->get()),
                cache.at(1)->get_peak_rss() / memory_unit.first, peak_tolerance);
}

//--------------------------------------------------------------------------------------//

int
main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    _argc = argc;
    _argv = argv;

    auto ret = RUN_ALL_TESTS();

    tim::timemory_finalize();
    tim::dmp::finalize();
    return ret;
}

//--------------------------------------------------------------------------------------//

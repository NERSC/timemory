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

#include "test_macros.hpp"

TIMEMORY_TEST_DEFAULT_MAIN

#include "timemory/timemory.hpp"

//--------------------------------------------------------------------------------------//

namespace details
{
//  Get the current tests name
inline std::string
get_test_name()
{
    return std::string(::testing::UnitTest::GetInstance()->current_test_suite()->name()) +
           "." + ::testing::UnitTest::GetInstance()->current_test_info()->name();
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
void
random_fill(std::vector<Tp>& v, Tp scale, Tp offset = 1.0)
{
    std::mt19937 rng;
    rng.seed(std::random_device()());
    std::generate(v.begin(), v.end(), [&]() {
        return (scale * std::generate_canonical<Tp, 10>(rng)) + offset;
    });
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

}  // namespace details

//--------------------------------------------------------------------------------------//

class api_tests : public ::testing::Test
{
protected:
    TIMEMORY_TEST_DEFAULT_SUITE_BODY
};

namespace os       = ::tim::os;
namespace api      = ::tim::api;
namespace project  = ::tim::project;
namespace category = ::tim::category;
namespace tpls     = ::tim::tpls;
namespace trait    = ::tim::trait;
namespace concepts = ::tim::concepts;

using namespace tim::component;

using tim_bundle_t  = tim::component_bundle<project::timemory, wall_clock, cpu_clock>;
using cali_bundle_t = tim::component_bundle<tpls::caliper, caliper_marker, wall_clock>;
using time_bundle_t = tim::component_tuple<wall_clock, cpu_clock, num_minor_page_faults>;
using os_bundle_t   = tim::component_bundle<os::agnostic, wall_clock, cpu_clock>;

//--------------------------------------------------------------------------------------//

TEST_F(api_tests, enum_vs_macro)
{
    namespace component = tim::component;
    auto macro_sz = std::tuple_size<tim::type_list<TIMEMORY_COMPONENT_TYPES>>::value;
    auto enum_sz =
        TIMEMORY_NATIVE_COMPONENTS_END - TIMEMORY_NATIVE_COMPONENT_INTERNAL_SIZE;

    EXPECT_EQ(macro_sz, enum_sz);
}

//--------------------------------------------------------------------------------------//

TEST_F(api_tests, placeholder)
{
    using nothing_placeholder = placeholder<nothing>;

    auto in_complete = tim::is_one_of<nothing_placeholder, tim::complete_types_t>::value;
    auto in_avail    = tim::is_one_of<nothing_placeholder, tim::available_types_t>::value;

    EXPECT_FALSE(in_complete) << "complete_types_t: "
                              << tim::demangle<tim::complete_types_t>() << std::endl;

    EXPECT_FALSE(in_avail) << "available_types_t: "
                           << tim::demangle<tim::available_types_t>() << std::endl;
}

//--------------------------------------------------------------------------------------//

TEST_F(api_tests, timemory)
{
    using test_t  = project::timemory;
    auto incoming = trait::runtime_enabled<test_t>::get();
    trait::runtime_enabled<test_t>::set(false);

    tim_bundle_t _obj(details::get_test_name());
    _obj.start();
    details::consume(1000);
    _obj.stop();
    EXPECT_NEAR(_obj.get<wall_clock>()->get(), 0.0, 1.0e-6);
    EXPECT_NEAR(_obj.get<cpu_clock>()->get(), 0.0, 1.0e-6);

    trait::runtime_enabled<test_t>::set(incoming);
}

//--------------------------------------------------------------------------------------//

TEST_F(api_tests, tpls)
{
    using test_t  = tpls::caliper;
    auto incoming = trait::runtime_enabled<test_t>::get();
    trait::runtime_enabled<test_t>::set(false);

    cali_bundle_t _obj(details::get_test_name());
    _obj.start();
    details::consume(1000);
    _obj.stop();
    EXPECT_EQ(_obj.get<wall_clock>(), nullptr);

    trait::runtime_enabled<test_t>::set(incoming);
}

//--------------------------------------------------------------------------------------//

TEST_F(api_tests, category)
{
    using test_t  = category::timing;
    auto incoming = trait::runtime_enabled<test_t>::get();
    trait::runtime_enabled<test_t>::set(false);

    using wc_api_t   = trait::component_apis_t<wall_clock>;
    using wc_true_t  = tim::get_true_types_t<trait::runtime_configurable, wc_api_t>;
    using wc_false_t = tim::get_false_types_t<trait::runtime_configurable, wc_api_t>;
    using apitypes_t = typename trait::runtime_enabled<wall_clock>::api_type_list;

    puts("");
    PRINT_HERE("component-apis : %s", tim::demangle<wc_api_t>().c_str());
    PRINT_HERE("true-types     : %s", tim::demangle<wc_false_t>().c_str());
    PRINT_HERE("false-types    : %s", tim::demangle<wc_true_t>().c_str());
    PRINT_HERE("type-traits    : %s", tim::demangle<apitypes_t>().c_str());
    puts("");

    EXPECT_FALSE(trait::runtime_enabled<wall_clock>::get());
    EXPECT_FALSE(trait::runtime_enabled<cpu_clock>::get());
    EXPECT_TRUE(trait::runtime_enabled<num_minor_page_faults>::get());

    time_bundle_t _obj(details::get_test_name());
    _obj.start();
    details::consume(500);
    for(size_t i = 0; i < 10; ++i)
    {
        std::vector<double> _data(10000, 0.0);
        details::random_fill(_data, 10.0, 1.0);
        auto val = details::random_entry(_data);
        EXPECT_GE(val, 1.0);
        EXPECT_LE(val, 11.0);
    }
    _obj.stop();
    EXPECT_NEAR(_obj.get<wall_clock>()->get(), 0.0, 1.0e-6);
    EXPECT_NEAR(_obj.get<cpu_clock>()->get(), 0.0, 1.0e-6);
    EXPECT_GT(_obj.get<num_minor_page_faults>()->get(), 0);

    trait::runtime_enabled<test_t>::set(incoming);
}

//--------------------------------------------------------------------------------------//

TEST_F(api_tests, os)
{
    trait::apply<trait::runtime_enabled>::set<os::supports_unix, os::supports_windows,
                                              wall_clock>(false);

    os_bundle_t _obj(details::get_test_name());
    _obj.start();
    details::consume(1000);
    _obj.stop();
    EXPECT_NEAR(_obj.get<wall_clock>()->get(), 0.0, 1.0e-6);
    EXPECT_NEAR(_obj.get<cpu_clock>()->get(), 1.0, 0.1);

    trait::apply<trait::runtime_enabled>::set<os::supports_unix, os::supports_windows,
                                              wall_clock>(true);
}

//--------------------------------------------------------------------------------------//

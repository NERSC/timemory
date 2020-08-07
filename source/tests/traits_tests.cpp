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

static int    _argc = 0;
static char** _argv = nullptr;

using mutex_t = std::mutex;
using lock_t  = std::unique_lock<mutex_t>;

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

}  // namespace details

//--------------------------------------------------------------------------------------//

class traits_tests : public ::testing::Test
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
            tim::settings::file_output() = false;
            tim::settings::mpi_thread()  = false;
            tim::mpi::initialize(_argc, _argv);
            tim::timemory_init(_argc, _argv);
            tim::settings::dart_output() = true;
            tim::settings::dart_count()  = 1;
            tim::settings::banner()      = false;
        }
    }
};

//--------------------------------------------------------------------------------------//

TIMEMORY_DECLARE_COMPONENT(void_component)
TIMEMORY_DECLARE_COMPONENT(int64_component)
TIMEMORY_DECLARE_COMPONENT(array_component)
TIMEMORY_TEMPLATE_COMPONENT(template_component, typename T, T)
TIMEMORY_TEMPLATE_COMPONENT(variadic_component, TIMEMORY_ESC(typename T, size_t... N), T,
                            N...)

TIMEMORY_DEFINE_CONCRETE_TRAIT(base_has_accum, component::array_component, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(base_has_last, component::int64_component, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::template_component<int64_t>,
                               false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::template_component<int32_t>,
                               false_type)

TIMEMORY_TRAIT_TYPE(data, component::void_component, void)
TIMEMORY_TRAIT_TYPE(data, component::array_component, std::array<double, 4>)
TIMEMORY_TEMPLATE_TRAIT_TYPE(data, component::template_component, typename T, T,
                             std::pair<int, T>)
TIMEMORY_TEMPLATE_TRAIT_TYPE(data, component::variadic_component,
                             TIMEMORY_ESC(typename T, size_t... N), TIMEMORY_ESC(T, N...),
                             std::tuple<std::array<T, N>...>)

namespace tim
{
namespace component
{
struct void_component : base<void_component>
{};
struct int64_component : base<int64_component, int64_t>
{};
struct array_component : base<array_component>
{};
template <typename T>
struct template_component : base<template_component<T>>
{};
template <typename T, size_t... N>
struct variadic_component : base<variadic_component<T, N...>>
{};
}  // namespace component
}  // namespace tim

using namespace tim::component;
using variadic_type_t = std::tuple<std::array<double, 1>, std::array<double, 3>>;

//--------------------------------------------------------------------------------------//

TEST_F(traits_tests, data)
{
#define TYPE_TEST(DATA_TYPE, ...)                                                        \
    std::is_same<TIMEMORY_ESC(DATA_TYPE), tim::trait::data_t<__VA_ARGS__>>::value;       \
    printf("[%s]> data type for %-38s :: %s\n", details::get_test_name().c_str(),        \
           tim::demangle<__VA_ARGS__>().substr(16).c_str(),                              \
           tim::demangle<tim::trait::data_t<__VA_ARGS__>>().c_str())

    auto void_test  = TYPE_TEST(void, void_component);
    auto int_test   = TYPE_TEST(tim::type_list<>, int64_component);
    auto array_test = TYPE_TEST(TIMEMORY_ESC(std::array<double, 4>), array_component);
    auto temp_i64_test =
        TYPE_TEST(TIMEMORY_ESC(std::pair<int, int64_t>), template_component<int64_t>);
    auto temp_i32_test =
        TYPE_TEST(TIMEMORY_ESC(std::pair<int, int32_t>), template_component<int32_t>);
    auto temp_u64_test =
        TYPE_TEST(TIMEMORY_ESC(std::pair<int, uint64_t>), template_component<uint64_t>);
    auto var_test = TYPE_TEST(variadic_type_t, variadic_component<double, 1, 3>);

    EXPECT_TRUE(void_test);
    EXPECT_TRUE(int_test);
    EXPECT_TRUE(array_test);
    EXPECT_TRUE(temp_i64_test);
    EXPECT_TRUE(temp_i32_test);
    EXPECT_TRUE(temp_u64_test);
    EXPECT_TRUE(var_test);
#undef TYPE_TEST
}

//--------------------------------------------------------------------------------------//

TEST_F(traits_tests, component_value_type)
{
#define TYPE_TEST(DATA_TYPE, ...)                                                        \
    std::is_same<TIMEMORY_ESC(DATA_TYPE),                                                \
                 tim::trait::component_value_type_t<__VA_ARGS__>>::value;                \
    printf("[%s]> component value type for %-38s :: %s\n",                               \
           details::get_test_name().c_str(),                                             \
           tim::demangle<__VA_ARGS__>().substr(16).c_str(),                              \
           tim::demangle<tim::trait::component_value_type_t<__VA_ARGS__>>().c_str())

    auto void_test     = TYPE_TEST(void, void_component);
    auto int_test      = TYPE_TEST(int64_t, int64_component);
    auto array_test    = TYPE_TEST(TIMEMORY_ESC(std::array<double, 4>), array_component);
    auto temp_i64_test = TYPE_TEST(void, template_component<int64_t>);
    auto temp_i32_test = TYPE_TEST(void, template_component<int32_t>);
    auto temp_u64_test =
        TYPE_TEST(TIMEMORY_ESC(std::pair<int, uint64_t>), template_component<uint64_t>);
    auto var_test = TYPE_TEST(variadic_type_t, variadic_component<double, 1, 3>);

    EXPECT_TRUE(void_test);
    EXPECT_TRUE(int_test);
    EXPECT_TRUE(array_test);
    EXPECT_TRUE(temp_i64_test);
    EXPECT_TRUE(temp_i32_test);
    EXPECT_TRUE(temp_u64_test);
    EXPECT_TRUE(var_test);
#undef TYPE_TEST
}

//--------------------------------------------------------------------------------------//

TEST_F(traits_tests, value_type)
{
#define TYPE_TEST(DATA_TYPE, ...)                                                        \
    std::is_same<TIMEMORY_ESC(DATA_TYPE), typename __VA_ARGS__::value_type>::value;      \
    printf("[%s]> value type for %-38s :: %s\n", details::get_test_name().c_str(),       \
           tim::demangle<__VA_ARGS__>().substr(16).c_str(),                              \
           tim::demangle<typename __VA_ARGS__::value_type>().c_str())

    auto void_test  = TYPE_TEST(void, void_component);
    auto int_test   = TYPE_TEST(int64_t, int64_component);
    auto array_test = TYPE_TEST(TIMEMORY_ESC(std::array<double, 4>), array_component);
    auto temp_i64_test =
        TYPE_TEST(TIMEMORY_ESC(std::pair<int, int64_t>), template_component<int64_t>);
    auto temp_i32_test =
        TYPE_TEST(TIMEMORY_ESC(std::pair<int, int32_t>), template_component<int32_t>);
    auto temp_u64_test =
        TYPE_TEST(TIMEMORY_ESC(std::pair<int, uint64_t>), template_component<uint64_t>);
    auto var_test = TYPE_TEST(variadic_type_t, variadic_component<double, 1, 3>);

    EXPECT_TRUE(void_test);
    EXPECT_TRUE(int_test);
    EXPECT_TRUE(array_test);
    EXPECT_TRUE(temp_i64_test);
    EXPECT_TRUE(temp_i32_test);
    EXPECT_TRUE(temp_u64_test);
    EXPECT_TRUE(var_test);
#undef TYPE_TEST
}

//--------------------------------------------------------------------------------------//

TEST_F(traits_tests, accum_type)
{
#define TYPE_TEST(DATA_TYPE, ...)                                                        \
    std::is_same<TIMEMORY_ESC(DATA_TYPE), typename __VA_ARGS__::accum_type>::value;      \
    printf("[%s]> accum type for %-38s :: %s\n", details::get_test_name().c_str(),       \
           tim::demangle<__VA_ARGS__>().substr(16).c_str(),                              \
           tim::demangle<typename __VA_ARGS__::accum_type>().c_str())

    auto void_test  = TYPE_TEST(void, void_component);
    auto int_test   = TYPE_TEST(int64_t, int64_component);
    auto array_test = TYPE_TEST(std::tuple<>, array_component);
    auto temp_i64_test =
        TYPE_TEST(TIMEMORY_ESC(std::pair<int, int64_t>), template_component<int64_t>);
    auto temp_i32_test =
        TYPE_TEST(TIMEMORY_ESC(std::pair<int, int32_t>), template_component<int32_t>);
    auto temp_u64_test =
        TYPE_TEST(TIMEMORY_ESC(std::pair<int, uint64_t>), template_component<uint64_t>);
    auto var_test = TYPE_TEST(variadic_type_t, variadic_component<double, 1, 3>);

    EXPECT_TRUE(void_test);
    EXPECT_TRUE(int_test);
    EXPECT_TRUE(array_test);
    EXPECT_TRUE(temp_i64_test);
    EXPECT_TRUE(temp_i32_test);
    EXPECT_TRUE(temp_u64_test);
    EXPECT_TRUE(var_test);
#undef TYPE_TEST
}

//--------------------------------------------------------------------------------------//

TEST_F(traits_tests, last_type)
{
#define TYPE_TEST(DATA_TYPE, ...)                                                        \
    std::is_same<TIMEMORY_ESC(DATA_TYPE), typename __VA_ARGS__::last_type>::value;       \
    printf("[%s]> last type for %-38s :: %s\n", details::get_test_name().c_str(),        \
           tim::demangle<__VA_ARGS__>().substr(16).c_str(),                              \
           tim::demangle<typename __VA_ARGS__::last_type>().c_str())

    auto void_test     = TYPE_TEST(void, void_component);
    auto int_test      = TYPE_TEST(int64_t, int64_component);
    auto array_test    = TYPE_TEST(std::tuple<>, array_component);
    auto temp_i64_test = TYPE_TEST(std::tuple<>, template_component<int64_t>);
    auto temp_i32_test = TYPE_TEST(std::tuple<>, template_component<int32_t>);
    auto temp_u64_test = TYPE_TEST(std::tuple<>, template_component<uint64_t>);
    auto var_test      = TYPE_TEST(std::tuple<>, variadic_component<double, 1, 3>);

    EXPECT_TRUE(void_test);
    EXPECT_TRUE(int_test);
    EXPECT_TRUE(array_test);
    EXPECT_TRUE(temp_i64_test);
    EXPECT_TRUE(temp_i32_test);
    EXPECT_TRUE(temp_u64_test);
    EXPECT_TRUE(var_test);
#undef TYPE_TEST
}

//--------------------------------------------------------------------------------------//

TEST_F(traits_tests, generates_output)
{
#define TYPE_TEST(...)                                                                   \
    tim::trait::generates_output<__VA_ARGS__>::value;                                    \
    printf("[%s]> output type for %-38s :: %s\n", details::get_test_name().c_str(),      \
           tim::demangle<__VA_ARGS__>().substr(16).c_str(),                              \
           tim::demangle<tim::trait::generates_output<__VA_ARGS__>::type>().c_str())

    auto void_test     = TYPE_TEST(void_component);
    auto int_test      = TYPE_TEST(int64_component);
    auto array_test    = TYPE_TEST(array_component);
    auto temp_i64_test = TYPE_TEST(template_component<int64_t>);
    auto temp_i32_test = TYPE_TEST(template_component<int32_t>);
    auto temp_u64_test = TYPE_TEST(template_component<uint64_t>);
    auto var_test      = TYPE_TEST(variadic_component<double, 1, 3>);

    EXPECT_FALSE(void_test);
    EXPECT_TRUE(int_test);
    EXPECT_TRUE(array_test);
    EXPECT_TRUE(temp_i64_test);
    EXPECT_TRUE(temp_i32_test);
    EXPECT_TRUE(temp_u64_test);
    EXPECT_TRUE(var_test);
#undef TYPE_TEST
}

//--------------------------------------------------------------------------------------//

TEST_F(traits_tests, collects_data)
{
#define TYPE_TEST(...)                                                                   \
    tim::trait::collects_data<__VA_ARGS__>::value;                                       \
    printf("[%s]> collects_data type for %-38s :: %s\n",                                 \
           details::get_test_name().c_str(),                                             \
           tim::demangle<__VA_ARGS__>().substr(16).c_str(),                              \
           (tim::trait::collects_data<__VA_ARGS__>::value) ? "true" : "false")

    auto void_test     = TYPE_TEST(void_component);
    auto int_test      = TYPE_TEST(int64_component);
    auto array_test    = TYPE_TEST(array_component);
    auto temp_i64_test = TYPE_TEST(template_component<int64_t>);
    auto temp_i32_test = TYPE_TEST(template_component<int32_t>);
    auto temp_u64_test = TYPE_TEST(template_component<uint64_t>);
    auto var_test      = TYPE_TEST(variadic_component<double, 1, 3>);

    EXPECT_FALSE(void_test);
    EXPECT_TRUE(int_test);
    EXPECT_TRUE(array_test);
    EXPECT_FALSE(temp_i64_test);
    EXPECT_FALSE(temp_i32_test);
    EXPECT_TRUE(temp_u64_test);
    EXPECT_TRUE(var_test);
#undef TYPE_TEST
}

//--------------------------------------------------------------------------------------//

TEST_F(traits_tests, sizeof)
{
#define TYPE_TEST(...)                                                                   \
    sizeof(__VA_ARGS__);                                                                 \
    printf("[%s]> sizeof(%-38s) = %lu\n", details::get_test_name().c_str(),              \
           tim::demangle<__VA_ARGS__>().substr(16).c_str(),                              \
           (unsigned long) sizeof(__VA_ARGS__))

    auto void_test     = TYPE_TEST(void_component);
    auto int_test      = TYPE_TEST(int64_component);
    auto array_test    = TYPE_TEST(array_component);
    auto temp_i64_test = TYPE_TEST(template_component<int64_t>);
    auto temp_i32_test = TYPE_TEST(template_component<int32_t>);
    auto temp_u64_test = TYPE_TEST(template_component<uint64_t>);
    auto var_test      = TYPE_TEST(variadic_component<double, 1, 3>);

    EXPECT_EQ(void_test, 16);
    EXPECT_EQ(int_test, 64);
    EXPECT_EQ(temp_i32_test, 56);
    EXPECT_EQ(temp_i64_test, 72);
    EXPECT_EQ(temp_u64_test, 80);
    EXPECT_EQ(array_test, 80);
    EXPECT_EQ(var_test, 112);
#undef TYPE_TEST
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

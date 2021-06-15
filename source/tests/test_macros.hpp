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

#if defined(TIMEMORY_USE_MPI)
#    include <mpi.h>
#endif

#if !defined(TIMEMORY_TEST_NO_METRIC)
#    include "timemory/timemory.hpp"
#endif

#include "gtest/gtest.h"

#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using mutex_t        = std::mutex;
using lock_t         = std::unique_lock<mutex_t>;
using string_t       = std::string;
using stringstream_t = std::stringstream;

namespace
{
#if !defined(DISABLE_TIMEMORY)
template <typename Tp>
inline std::string
as_string(const tim::node::result<Tp>& _obj)
{
    std::ostringstream _oss{};
    _oss << "tid=" << std::setw(2) << _obj.tid() << ", ";
    _oss << "pid=" << std::setw(6) << _obj.pid() << ", ";
    _oss << "depth=" << std::setw(2) << _obj.depth() << ", ";
    _oss << "hash=" << std::setw(21) << _obj.hash() << ", ";
    _oss << "prefix=" << _obj.prefix() << ", ";
    _oss << "data=" << _obj.data();
    return _oss.str();
}
#else
template <typename Tp>
inline std::string
as_string(const Tp&)
{
    return std::string{};
}
#endif
//
#if !defined(TIMEMORY_TEST_NO_METRIC) && !defined(DISABLE_TIMEMORY)
inline auto&
metric()
{
    static tim::lightweight_tuple<tim::component::wall_clock, tim::component::peak_rss>
        _instance{ ::testing::UnitTest::GetInstance()->current_test_suite()->name() };
    return _instance;
}
inline void
print_dart(
    tim::lightweight_tuple<tim::component::wall_clock, tim::component::peak_rss>& ct)
{
    if(tim::dmp::rank() > 0)
        return;
    using namespace tim::component;
    auto* wc = ct.get<wall_clock>();
    if(wc)
        tim::operation::echo_measurement<wall_clock, true>{ *wc, { "wall_clock" } };
    auto* pr = ct.get<peak_rss>();
    if(pr)
        tim::operation::echo_measurement<peak_rss, true>{ *pr, { "peak_rss" } };
}
#else
struct dummy
{
    void start() {}
    void stop() {}
};
inline auto&
metric()
{
    static dummy _instance{};
    return _instance;
}
inline void
print_dart(dummy&)
{}
#endif
}  // namespace

#if !defined(DISABLE_TIMEMORY)
#    define TIMEMORY_TEST_ARGS                                                           \
        static int    _argc = 0;                                                         \
        static char** _argv = nullptr;

#    define TIMEMORY_TEST_MAIN                                                           \
        int main(int argc, char** argv)                                                  \
        {                                                                                \
            ::testing::InitGoogleTest(&argc, argv);                                      \
            _argc = argc;                                                                \
            _argv = argv;                                                                \
            return RUN_ALL_TESTS();                                                      \
        }

#    define TIMEMORY_TEST_DEFAULT_MAIN                                                   \
        TIMEMORY_TEST_ARGS                                                               \
        TIMEMORY_TEST_MAIN

#    define TIMEMORY_TEST_SUITE_SETUP(...)                                               \
    protected:                                                                           \
        static void SetUpTestSuite()                                                     \
        {                                                                                \
            puts("[SetupTestSuite] setup starting");                                     \
            tim::settings::verbose()     = 0;                                            \
            tim::settings::debug()       = false;                                        \
            tim::settings::json_output() = true;                                         \
            puts("[SetupTestSuite] initializing dmp");                                   \
            tim::dmp::initialize(_argc, _argv);                                          \
            puts("[SetupTestSuite] initializing timemory");                              \
            tim::timemory_init(_argc, _argv);                                            \
            puts("[SetupTestSuite] timemory initialized");                               \
            tim::settings::dart_output() = false;                                        \
            tim::settings::dart_count()  = 1;                                            \
            tim::settings::banner()      = false;                                        \
            __VA_ARGS__;                                                                 \
            puts("[SetupTestSuite] setup completed");                                    \
            metric().start();                                                            \
        }

#    define TIMEMORY_TEST_DEFAULT_SUITE_SETUP TIMEMORY_TEST_SUITE_SETUP({})

#    define TIMEMORY_TEST_SUITE_TEARDOWN(...)                                            \
    protected:                                                                           \
        static void TearDownTestSuite()                                                  \
        {                                                                                \
            metric().stop();                                                             \
            print_dart(metric());                                                        \
            __VA_ARGS__;                                                                 \
            tim::timemory_finalize();                                                    \
            if(tim::dmp::rank() == 0)                                                    \
                tim::enable_signal_detection(tim::signal_settings::get_default());       \
            tim::dmp::finalize();                                                        \
        }

#else
#    define TIMEMORY_TEST_ARGS
#    if defined(TIMEMORY_USE_MPI)
#        define TIMEMORY_TEST_MAIN                                                       \
            int main(int argc, char** argv)                                              \
            {                                                                            \
                MPI_Init(&argc, &argv);                                                  \
                ::testing::InitGoogleTest(&argc, argv);                                  \
                auto ret = RUN_ALL_TESTS();                                              \
                MPI_Finalize();                                                          \
                return ret;                                                              \
            }
#    else
#        define TIMEMORY_TEST_MAIN                                                       \
            int main(int argc, char** argv)                                              \
            {                                                                            \
                ::testing::InitGoogleTest(&argc, argv);                                  \
                return RUN_ALL_TESTS();                                                  \
            }
#    endif
#    define TIMEMORY_TEST_DEFAULT_MAIN TIMEMORY_TEST_MAIN
#    define TIMEMORY_TEST_SUITE_SETUP(...)
#    define TIMEMORY_TEST_DEFAULT_SUITE_SETUP
#    define TIMEMORY_TEST_SUITE_TEARDOWN(...)
#endif

#define TIMEMORY_TEST_DEFAULT_SUITE_TEARDOWN TIMEMORY_TEST_SUITE_TEARDOWN({})

#define TIMEMORY_TEST_SETUP(...)                                                         \
protected:                                                                               \
    void SetUp() override                                                                \
    {                                                                                    \
        puts("");                                                                        \
        printf("##### Executing %s ... #####\n", details::get_test_name().c_str());      \
        puts("");                                                                        \
        __VA_ARGS__;                                                                     \
    }

#define TIMEMORY_TEST_DEFAULT_SETUP TIMEMORY_TEST_SETUP({})

#define TIMEMORY_TEST_TEARDOWN(...)                                                      \
protected:                                                                               \
    void TearDown() override                                                             \
    {                                                                                    \
        __VA_ARGS__;                                                                     \
        puts("");                                                                        \
    }

#define TIMEMORY_TEST_DEFAULT_TEARDOWN TIMEMORY_TEST_TEARDOWN({})

#define TIMEMORY_TEST_DEFAULT_SUITE_BODY                                                 \
    TIMEMORY_TEST_DEFAULT_SUITE_SETUP                                                    \
    TIMEMORY_TEST_DEFAULT_SUITE_TEARDOWN                                                 \
    TIMEMORY_TEST_DEFAULT_SETUP                                                          \
    TIMEMORY_TEST_DEFAULT_TEARDOWN

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
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using mutex_t        = std::mutex;
using lock_t         = std::unique_lock<mutex_t>;
using string_t       = std::string;
using stringstream_t = std::stringstream;

#define TIMEMORY_TEST_ARGS                                                               \
    static int    _argc = 0;                                                             \
    static char** _argv = nullptr;

#define TIMEMORY_TEST_MAIN                                                               \
    int main(int argc, char** argv)                                                      \
    {                                                                                    \
        ::testing::InitGoogleTest(&argc, argv);                                          \
        _argc = argc;                                                                    \
        _argv = argv;                                                                    \
        return RUN_ALL_TESTS();                                                          \
    }

#define TIMEMORY_TEST_DEFAULT_MAIN                                                       \
    TIMEMORY_TEST_ARGS                                                                   \
    TIMEMORY_TEST_MAIN

#define TIMEMORY_TEST_SUITE_SETUP(...)                                                   \
protected:                                                                               \
    static void SetUpTestSuite()                                                         \
    {                                                                                    \
        tim::settings::verbose()     = 0;                                                \
        tim::settings::debug()       = false;                                            \
        tim::settings::json_output() = true;                                             \
        tim::settings::mpi_thread()  = false;                                            \
        tim::dmp::initialize(_argc, _argv);                                              \
        tim::timemory_init(_argc, _argv);                                                \
        tim::settings::dart_output() = true;                                             \
        tim::settings::dart_count()  = 1;                                                \
        tim::settings::banner()      = false;                                            \
        __VA_ARGS__;                                                                     \
    }

#define TIMEMORY_TEST_DEFAULT_SUITE_SETUP TIMEMORY_TEST_SUITE_SETUP({})

#define TIMEMORY_TEST_SUITE_TEARDOWN(...)                                                \
protected:                                                                               \
    static void TearDownTestSuite()                                                      \
    {                                                                                    \
        __VA_ARGS__;                                                                     \
        tim::timemory_finalize();                                                        \
        tim::dmp::finalize();                                                            \
    }

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

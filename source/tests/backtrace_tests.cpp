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

TIMEMORY_NOINLINE auto
foo()
{
    return tim::get_backtrace<4>();
}

TIMEMORY_NOINLINE auto
bar()
{
    return foo();
}

TIMEMORY_NOINLINE auto
spam()
{
    return bar();
}

TIMEMORY_NOINLINE auto
foo_d()
{
    return tim::get_demangled_backtrace<4>();
}

TIMEMORY_NOINLINE auto
bar_d()
{
    return foo_d();
}

TIMEMORY_NOINLINE auto
spam_d()
{
    return bar_d();
}

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

class backtrace_tests : public ::testing::Test
{
protected:
    TIMEMORY_TEST_DEFAULT_SUITE_SETUP
    TIMEMORY_TEST_DEFAULT_SUITE_TEARDOWN

    TIMEMORY_TEST_DEFAULT_SETUP
    TIMEMORY_TEST_DEFAULT_TEARDOWN
};

//--------------------------------------------------------------------------------------//

TEST_F(backtrace_tests, backtrace)
{
    auto ret = spam();
    int  cnt = 0;
    for(const auto& itr : ret)
    {
        if(itr)
        {
            std::cerr << itr << std::endl;
            ++cnt;
        }
    }

    ASSERT_EQ(cnt, 4);

    for(const auto& itr : ret)
        EXPECT_TRUE(std::string(itr).find("backtrace_tests") != std::string::npos) << itr;

    EXPECT_TRUE(std::string(ret.at(0)).find("_Z3foo") != std::string::npos) << ret.at(0);
    EXPECT_TRUE(std::string(ret.at(1)).find("_Z3bar") != std::string::npos) << ret.at(1);
    EXPECT_TRUE(std::string(ret.at(2)).find("_Z4spam") != std::string::npos) << ret.at(2);
    EXPECT_TRUE(std::string(ret.at(3)).find("_ZN30backtrace_tests_backtrace_Test") !=
                std::string::npos)
        << ret.at(3);
}

//--------------------------------------------------------------------------------------//

TEST_F(backtrace_tests, demangled_backtrace)
{
    auto ret = spam_d();
    int  cnt = 0;
    for(const auto& itr : ret)
    {
        if(!itr.empty())
        {
            std::cerr << itr << std::endl;
            ++cnt;
        }
    }

    ASSERT_EQ(cnt, 4);

    for(const auto& itr : ret)
        EXPECT_TRUE(itr.find("backtrace_tests") != std::string::npos) << itr;

    EXPECT_TRUE(ret.at(0).find("foo_d()") != std::string::npos) << ret.at(0);
    EXPECT_TRUE(ret.at(1).find("bar_d()") != std::string::npos) << ret.at(1);
    EXPECT_TRUE(ret.at(2).find("spam_d()") != std::string::npos) << ret.at(2);
    EXPECT_TRUE(ret.at(3).find("backtrace_tests_demangled_backtrace_Test::TestBody()") !=
                std::string::npos)
        << ret.at(3);
}

//--------------------------------------------------------------------------------------//

TEST_F(backtrace_tests, decode)
{
    auto ret_m = spam();
    auto ret_d = spam_d();

    int cnt_m = 0;
    int cnt_d = 0;
    for(const auto& itr : ret_m)
    {
        if(itr)
            ++cnt_m;
    }

    for(const auto& itr : ret_d)
    {
        if(!itr.empty())
            ++cnt_d;
    }

    ASSERT_EQ(cnt_m, cnt_d);

    for(int i = 0; i < cnt_m; ++i)
    {
        std::string _d = tim::operation::decode<TIMEMORY_API>{}(ret_d.at(i));
        EXPECT_EQ(ret_d.at(i), _d);
        auto _pos = _d.find("_d()");
        while(_pos != std::string::npos)
        {
            _d   = _d.erase(_pos, 2);
            _pos = _d.find("_d()");
        }
        std::string _m = tim::operation::decode<TIMEMORY_API>{}(ret_m.at(i));
        std::regex  _re(
            "([0-9])[ ]+([a-zA-Z:_]+)[ ]+0x[0-9a-fA-F]+[ ]+([A-Za-z_:\\(\\)]+).*");
        _m = std::regex_replace(_m, _re, "$1 $2 $3");
        _d = std::regex_replace(_d, _re, "$1 $2 $3");
        if(i == 0)
        {
            EXPECT_EQ(_m, std::string("0 backtrace_tests foo()"));
            EXPECT_EQ(_d, std::string("0 backtrace_tests foo()"));
        }
        EXPECT_EQ(_m, _d) << "\nmangle: " << ret_m.at(i) << "\ndemangle: " << ret_d.at(i);
    }
}

//--------------------------------------------------------------------------------------//

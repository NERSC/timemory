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
    return tim::get_backtrace<4, 1>();
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

static inline auto&
replace_seq(std::string& inp, const std::string& seq, const std::string& repl = {})
{
    auto pos = std::string::npos;
    while((pos = inp.find(seq)) != std::string::npos)
    {
        if(repl.empty())
            inp = inp.erase(pos, seq.length());
        else
            inp = inp.replace(pos, seq.length(), repl);
    }
    return inp;
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
    for(auto itr : ret)
    {
        if(strlen(itr) > 0)
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

    for(auto& itr : ret)
        details::replace_seq(itr, "[abi:cxx11]");

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
    for(auto itr : ret_m)
    {
        if(strlen(itr))
            ++cnt_m;
    }

    for(const auto& itr : ret_d)
    {
        if(!itr.empty())
            ++cnt_d;
    }

    ASSERT_EQ(cnt_m, cnt_d);

    auto _basename = [](std::string& inp) -> std::string& {
        auto _b = inp.find_first_of('/');
        while(_b > 0 && _b != std::string::npos)
        {
            if(inp.at(_b - 1) == '.')
                --_b;
            else
                break;
        }
        auto _e = inp.find_last_of('/');
        if(_e > _b && _e != std::string::npos)
            inp = inp.erase(_b, _e - _b + 1);
        return inp;
    };

    for(int i = 0; i < cnt_m; ++i)
    {
        std::string _d = tim::operation::decode<TIMEMORY_API>{}(ret_d.at(i));
        EXPECT_EQ(ret_d.at(i), _d);
        std::string _m = tim::operation::decode<TIMEMORY_API>{}(ret_m.at(i));

        std::stringstream _errmsg;
        bool              _apply = true;
        auto _storemsg = [&_apply, &_d, &_m, &_errmsg](const std::string& _label) {
            if(_apply || tim::settings::debug() || tim::settings::verbose() > 2)
                _errmsg << '\n'
                        << _label << "\n\t_d :: " << _d << "\n\t_m :: " << _m << '\n';
        };

        // _storemsg("Stripping [abi:cxx11]");
        if(_apply)
        {
            details::replace_seq(_d, "[abi:cxx11]");
            details::replace_seq(_m, "[abi:cxx11]");
        }

        // _storemsg("Replacing _d() with ()");
        if(_apply)
        {
            details::replace_seq(_d, "_d()", "()");
            details::replace_seq(_m, "_d()", "()");
        }

        _storemsg("Generic replacement (macOS)");
        if(_apply)
        {
            std::regex _re(
                "[0-9 ]+([a-zA-Z:_]+)[ ]+0x[[:xdigit:]]+[ ]+([A-Za-z_:\\(\\)]+).*");
            if(std::regex_search(_m, _re) && std::regex_search(_d, _re))
                _apply = false;
            _m = std::regex_replace(_m, _re, "$1 $2");
            _d = std::regex_replace(_d, _re, "$1 $2");
        }

        _storemsg("Generic replacement (linux)");
        if(_apply)
        {
            std::regex _re("([^\\(]+)[\\(]([^\\+]+)(\\+0x[[:xdigit:]]+)(.*)");
            if(std::regex_search(_m, _re) && std::regex_search(_d, _re))
                _apply = false;
            _m = std::regex_replace(_m, _re, "$1 $2");
            _d = std::regex_replace(_d, _re, "$1 $2");
        }

#if defined(ONLY_HERE_FOR_REFERENCE_DONT_DEFINE_ME)
        _storemsg("Inserting spaces b/t address start");
        if(_apply)
        {
            std::regex _re("([A-Za-z0-9_])\\(([A-Za-z0-9_])");
            _m = std::regex_replace(_m, _re, "$1 $2");
            _d = std::regex_replace(_d, _re, "$1 $2");
        }

        _storemsg("Inserting spaces b/t address end");
        if(_apply)
        {
            std::regex _re("\\)(\\+0x[[:xdigit:]]+)\\)");
            _m = std::regex_replace(_m, _re, ") [$1]");
            _d = std::regex_replace(_d, _re, ") [$1]");
        }

        _storemsg("Stripping addresses");
        if(_apply)
        {
            std::regex _re("([ ]*\\[[+]?0x[[:xdigit:]]+\\])");
            _m = std::regex_replace(_m, _re, "");
            _d = std::regex_replace(_d, _re, "");
        }
#endif

        _apply = true;
        _storemsg("Basename");
        _m = _basename(_m);
        _d = _basename(_d);

        _storemsg("Final result");
        switch(i)
        {
            case 0:
            {
                auto _e = std::string("backtrace_tests foo()");
                EXPECT_EQ(_m, _e) << _errmsg.str();
                EXPECT_EQ(_d, _e) << _errmsg.str();
                break;
            }
            case 1:
            {
                auto _e = std::string("backtrace_tests bar()");
                EXPECT_EQ(_m, _e) << _errmsg.str();
                EXPECT_EQ(_d, _e) << _errmsg.str();
                break;
            }
            case 2:
            {
                auto _e = std::string("backtrace_tests spam()");
                EXPECT_EQ(_m, _e) << _errmsg.str();
                EXPECT_EQ(_d, _e) << _errmsg.str();
                break;
            }
            case 3:
            {
                auto _e = std::string(
                    "backtrace_tests backtrace_tests_decode_Test::TestBody()");
                EXPECT_EQ(_m, _e) << _errmsg.str();
                EXPECT_EQ(_d, _e) << _errmsg.str();
                break;
            }
            default: break;
        }

        EXPECT_EQ(_m, _d) << "\nmangle:   " << ret_m.at(i)
                          << "\ndemangle: " << ret_d.at(i);

        std::cerr << '[' << details::get_test_name() << " @ " << i << "]:\n  _d :: " << _d
                  << "\n  _m :: " << _m << '\n';
    }
}

//--------------------------------------------------------------------------------------//

TEST_F(backtrace_tests, print_backtrace)
{
    std::stringstream ss;
    tim::print_backtrace<3, 2>(ss, TIMEMORY_PID_TID_STRING,
                               TIMEMORY_FILE_LINE_FUNC_STRING);
    std::cerr << ss.str();
    auto btvec = tim::delimit(ss.str(), "\n");
    EXPECT_GE(btvec.size(), 3) << ss.str();
}

//--------------------------------------------------------------------------------------//

TEST_F(backtrace_tests, print_demangled_backtrace)
{
    std::stringstream ss;
    tim::print_demangled_backtrace<3, 2>(ss, TIMEMORY_PID_TID_STRING,
                                         TIMEMORY_FILE_LINE_FUNC_STRING);
    std::cerr << ss.str();
    auto btvec = tim::delimit(ss.str(), "\n");
    EXPECT_GE(btvec.size(), 3) << ss.str();
}

//--------------------------------------------------------------------------------------//
#if defined(TIMEMORY_USE_LIBUNWIND)

TEST_F(backtrace_tests, print_unw_backtrace)
{
    std::stringstream ss;
    tim::print_unw_backtrace<3, 2>(ss, TIMEMORY_PID_TID_STRING,
                                   TIMEMORY_FILE_LINE_FUNC_STRING);
    std::cerr << ss.str();
    auto btvec = tim::delimit(ss.str(), "\n");
    EXPECT_GE(btvec.size(), 3) << ss.str();
}

//--------------------------------------------------------------------------------------//

TEST_F(backtrace_tests, print_demangled_unw_backtrace)
{
    std::stringstream ss;
    tim::print_demangled_unw_backtrace<3, 2>(ss, TIMEMORY_PID_TID_STRING,
                                             TIMEMORY_FILE_LINE_FUNC_STRING);
    std::cerr << ss.str();
    auto btvec = tim::delimit(ss.str(), "\n");
    EXPECT_GE(btvec.size(), 3) << ss.str();
}

#endif

//--------------------------------------------------------------------------------------//

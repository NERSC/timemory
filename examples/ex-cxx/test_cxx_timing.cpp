// MIT License
//
// Copyright (c) 2018, The Regents of the University of California, 
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

#include <cmath>
#include <chrono>
#include <thread>
#include <fstream>
#include <unordered_map>
#include <vector>
#include <iterator>
#include <future>

#include <cassert>

#include <timemory/manager.hpp>
#include <timemory/auto_timer.hpp>
#include <timemory/signal_detection.hpp>
#include <timemory/mpi.hpp>
#include <timemory/rss.hpp>

typedef tim::timer          tim_timer_t;
typedef tim::manager manager_t;

// ASSERT_NEAR
// EXPECT_EQ
// EXPECT_FLOAT_EQ
// EXPECT_DOUBLE_EQ

#define EXPECT_EQ(lhs, rhs) if(lhs != rhs) { \
    std::stringstream ss; \
    ss << #lhs << " != " << #rhs << " @ line " \
       << __LINE__ << " of " << __FILE__; \
    std::cerr << ss.str() << std::endl; \
    throw std::runtime_error(ss.str()); }

#define ASSERT_FALSE(expr) if( expr ) { \
    std::stringstream ss; \
    ss << "Expression: ( " << #expr << " ) "\
       << "failed @ line " \
       << __LINE__ << " of " << __FILE__; \
    std::cerr << ss.str() << std::endl; \
    throw std::runtime_error(ss.str()); }

#define ASSERT_TRUE(expr) if(!( expr )) { \
    std::stringstream ss; \
    ss << "Expression: !( " << #expr << " ) "\
       << "failed @ line " \
       << __LINE__ << " of " << __FILE__; \
    std::cerr << ss.str() << std::endl; \
    throw std::runtime_error(ss.str()); }

#define PRINT_HERE std::cout << "HERE: " << " [ " << __FUNCTION__ \
    << ":" << __LINE__ << " ] " << std::endl;

//----------------------------------------------------------------------------//
// fibonacci calculation
int64_t fibonacci(int32_t n)
{
    if (n < 2) return n;
    if(n > 34)
    {
        TIMEMORY_AUTO_TIMER();
        return fibonacci(n-1) + fibonacci(n-2);
    }
    else
        return fibonacci(n-1) + fibonacci(n-2);
}
//----------------------------------------------------------------------------//
// time fibonacci with return type and arguments
// e.g. std::function < int32_t ( int32_t ) >
int64_t time_fibonacci(int32_t n)
{
    std::stringstream ss;
    ss << "(" << n << ")";
    TIMEMORY_AUTO_TIMER(ss.str());
    return fibonacci(n);
}
//----------------------------------------------------------------------------//

void print_info(const std::string&);
void print_size(const std::string&, int64_t, bool = true);
void print_depth(const std::string&, int64_t, bool = true);
void test_rss_usage();
void test_timing_pointer();
void test_manager();
void test_timing_toggle();
void test_timing_depth();
void test_timing_thread();
void test_serialize();
void test_format();

//============================================================================//

int main(int /*argc*/, char** argv)
{
    tim::enable_signal_detection({
                                   tim::sys_signal::sHangup,
                                   tim::sys_signal::sInterrupt,
                                   tim::sys_signal::sIllegal,
                                   tim::sys_signal::sSegFault,
                                   tim::sys_signal::sFPE
                               });

    tim_timer_t t = tim_timer_t("Total time");
    t.start();

    tim::format::timer::push();

    int num_fail = 0;
    int num_test = 0;

#define RUN_TEST(func) { try { num_test += 1; func (); } catch(std::exception& e) \
    { std::cerr << e.what() << std::endl; num_fail += 1; } }

    try
    {
        RUN_TEST(test_serialize);
        RUN_TEST(test_rss_usage);
        RUN_TEST(test_timing_pointer);
        RUN_TEST(test_manager);
        RUN_TEST(test_timing_toggle);
        RUN_TEST(test_timing_depth);
        RUN_TEST(test_timing_thread);
        RUN_TEST(test_format);
    }
    catch(std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }

    std::cout << "\nDone.\n" << std::endl;

    std::cout << "[" << argv[0] << "] ";

    if(num_fail > 0)
        std::cout << "Tests failed: " << num_fail << "/" << num_test << std::endl;
    else
        std::cout << "Tests passed: " << (num_test - num_fail) << "/" << num_test
                  << std::endl;

    t.stop();
    std::cout << std::endl;
    t.report();
    std::cout << std::endl;
    tim::format::timer::pop();

    manager_t::instance()->write_missing();
    tim::disable_signal_detection();

    exit(num_fail);
}

//============================================================================//

void print_info(const std::string& func)
{
    if(tim::mpi_rank() == 0)
        std::cout << "\n[" << tim::mpi_rank() << "]\e[1;31m TESTING \e[0m["
                  << "\e[1;36m" << func << "\e[0m"
                  << "]...\n" << std::endl;
}

//============================================================================//

void print_size(const std::string& func, int64_t line, bool extra_endl)
{
    if(tim::mpi_rank() == 0)
    {
        std::cout << "[" << tim::mpi_rank() << "] "
                  << func << "@" << line
                  << " : Timing manager size: "
                  << manager_t::instance()->size()
                  << std::endl;

        if(extra_endl)
            std::cout << std::endl;
    }
}

//============================================================================//

void print_string(const std::string& str)
{
    std::stringstream _ss;
    _ss << "[" << tim::mpi_rank() << "] "
        << str
        << std::endl;
    std::cout << _ss.str();
}

//============================================================================//

void print_depth(const std::string& func, int64_t line, bool extra_endl)
{
    if(tim::mpi_rank() == 0)
    {
        std::cout << "[" << tim::mpi_rank() << "] "
                  << func << "@" << line
                  << " : Timing manager size: "
                  << manager_t::instance()->get_max_depth()
                  << std::endl;

        if(extra_endl)
            std::cout << std::endl;
    }
}

//============================================================================//

void test_serialize()
{
    print_info(__FUNCTION__);
}

//============================================================================//

void test_rss_usage()
{
    print_info(__FUNCTION__);

    typedef std::vector<uint64_t> vector_t;

    tim::format::rss _format("", ": RSS [current = %c %A] [peak = %m %A]",
                             tim::units::kilobyte, false);

    tim::rss::usage _rss_init(_format);
    tim::rss::usage _rss_calc(_format);
    _rss_init.format()->prefix("initial");
    _rss_calc.format()->prefix("allocated");

    tim::format::rss _rformat("", "%C %A, %M %A, %c %A, %m %A",
                              tim::units::kilobyte, false);
    tim::format::timer _tformat(__FUNCTION__,
                                ": %w %T, %u %T, %s %T, %t %T, %p%, %R, x%l",
                                tim::units::msec, _rformat, false);

    auto rt = tim::timer(__FUNCTION__);
    auto ct = tim::timer(_tformat);
    rt.start();
    ct.start();

    uint64_t nsize = 1048576;
    vector_t* v = new vector_t();

    _rss_init.record();
    v->reserve(nsize);
    for(uint64_t i = 0; i < nsize; ++i)
        v->push_back(i);
    _rss_calc.record();

    v->clear();
    delete v;

    // real usage
    int64_t _r_usage = _rss_calc.current<int64_t>(tim::units::kilobyte) -
                       _rss_init.current<int64_t>(tim::units::kilobyte);

    // expected usage
    int64_t _e_usage = 8192;
    // actual difference
    int64_t _a_diff = std::abs(_r_usage - _e_usage);

    std::cout << _rss_init << std::endl;
    std::cout << _rss_calc << std::endl;
    std::cout << "    real usage diff : " << _r_usage << std::endl;
    std::cout << "expected usage diff : " << _e_usage << std::endl;
    std::cout << "  actual difference : " << _a_diff << std::endl;

    ASSERT_TRUE(_a_diff < 250);

    fibonacci(36);

    rt.stop();
    ct.stop();

    print_string(rt.as_string());
    print_string(ct.as_string());
}

//============================================================================//

void test_timing_pointer()
{
    print_info(__FUNCTION__);

    uint16_t set_depth = 5;
    uint16_t get_depth = 1;

    print_depth(__FUNCTION__, __LINE__, false);
    {
        manager_t::instance()->set_max_depth(set_depth);
    }

    print_depth(__FUNCTION__, __LINE__, false);
    {
        get_depth = manager_t::instance()->get_max_depth();
    }

    print_depth(__FUNCTION__, __LINE__, false);
    EXPECT_EQ(set_depth, get_depth);
    manager_t::instance()->set_max_depth(std::numeric_limits<uint16_t>::max());
}

//============================================================================//

void test_manager()
{
    print_info(__FUNCTION__);

    auto tman = manager_t::instance();
    tman->clear();

    bool _is_enabled = tman->is_enabled();
    tman->enable(true);

    tim_timer_t& t = tman->timer("manager_test");
    t.start();

    for(auto itr : { 34, 36, 39, 40, 42, 38, 34, 42 })
        time_fibonacci(itr);

    t.stop();

    print_size(__FUNCTION__, __LINE__);
    tman->report();
    tman->set_output_stream("test_output/cxx_timing_report.out");
    tman->report();
    tman->write_json("test_output/cxx_timing_report.json");
    tman->write_missing();
    EXPECT_EQ(manager_t::instance()->size(), 33);

    for(const auto& itr : *tman)
    {
        ASSERT_FALSE(itr.timer().real_elapsed() < 0.0);
        ASSERT_FALSE(itr.timer().user_elapsed() < 0.0);
    }

    tman->enable(_is_enabled);
}

//============================================================================//

void test_timing_toggle()
{
    print_info(__FUNCTION__);

    auto tman = manager_t::instance();
    tman->clear();

    bool _is_enabled = tman->is_enabled();
    tman->enable(true);
    tman->set_output_stream(std::cout);

    tman->enable(true);
    {
        TIMEMORY_AUTO_TIMER("[toggle_on]");
        time_fibonacci(42);
    }
    print_size(__FUNCTION__, __LINE__);
    tman->report();
    EXPECT_EQ(manager_t::instance()->size(), 11);

    tman->clear();
    tman->enable(false);
    std::cout << std::endl;
    {
        TIMEMORY_AUTO_TIMER("[toggle_off]");
        time_fibonacci(42);
        //std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    print_size(__FUNCTION__, __LINE__);
    tman->report();
    EXPECT_EQ(manager_t::instance()->size(), 1);

    tman->clear();
    tman->enable(true);
    {
        TIMEMORY_AUTO_TIMER("[toggle_on]");
        time_fibonacci(42);
        tman->enable(false);
        TIMEMORY_AUTO_TIMER("[toggle_off]");
        time_fibonacci(40);
    }
    print_size(__FUNCTION__, __LINE__);
    tman->report();
    EXPECT_EQ(manager_t::instance()->size(), 11);

    tman->write_serialization("test_output/cxx_timing_toggle.json");
    tman->write_missing();
    tman->enable(_is_enabled);
}

//============================================================================//

void test_timing_depth()
{
    print_info(__FUNCTION__);

    auto tman = manager_t::instance();
    tman->clear();

    bool _is_enabled = tman->is_enabled();
    tman->enable(true);
    tman->set_output_stream(std::cout);

    print_depth(__FUNCTION__, __LINE__, false);
    int32_t _max_depth = tman->get_max_depth();
    tman->set_max_depth(4);
    print_depth(__FUNCTION__, __LINE__, false);
    {
        TIMEMORY_AUTO_TIMER();
        for(auto itr : { 38, 39, 40 })
            time_fibonacci(itr);
    }

    bool ign_cutoff;
    print_depth(__FUNCTION__, __LINE__, false);
    print_size(__FUNCTION__, __LINE__);
    tman->report(ign_cutoff = true);
    EXPECT_EQ(manager_t::instance()->size(), 8);

    tman->write_serialization("test_output/cxx_timing_depth.json");
    tman->write_missing();
    tman->enable(_is_enabled);
    tman->set_max_depth(_max_depth);
}

//============================================================================//

typedef std::vector<std::thread*> thread_list_t;

//============================================================================//

void thread_func(int32_t nfib, std::shared_future<void> fut)
{
    //std::this_thread::sleep_for(std::chrono::milliseconds(500));

    //int32_t nsize = manager_t::instance()->size();
    //if(nsize > 0)
    //    std::cerr << "thread-local manager size: " << nsize << std::endl;

    //std::stringstream ss;
    //ss << "--> " << std::this_thread::get_id() << " -- waiting ... " << std::endl;
    //std::cout << ss.str();

    fut.get();
    time_fibonacci(nfib);
}

//============================================================================//

std::thread* create_thread(int32_t nfib, std::shared_future<void> fut)
{
    TIMEMORY_AUTO_TIMER();
    static int32_t n = 0;
    return new std::thread(thread_func, nfib + (n++)%2, fut);
}

//============================================================================//

void join_thread(thread_list_t::iterator titr, thread_list_t& tlist)
{
    if(titr == tlist.end())
        return;

    TIMEMORY_AUTO_TIMER();

    (*titr)->join();
    join_thread(++titr, tlist);
}

//============================================================================//

void test_timing_thread(int num_threads)
{
    std::stringstream ss;
    ss << "[" << num_threads << "_threads]";

    TIMEMORY_AUTO_TIMER(ss.str());

    thread_list_t threads(num_threads, nullptr);

    std::promise<void> prom;
    std::shared_future<void> fut = prom.get_future().share();

    for(auto& itr : threads)
        itr = create_thread(43, fut);

    std::this_thread::sleep_for(std::chrono::milliseconds(2000));

    prom.set_value();

    join_thread(threads.begin(), threads);

    for(auto& itr : threads)
        delete itr;

    threads.clear();
}

//============================================================================//

void test_timing_thread()
{
    print_info(__FUNCTION__);

    auto tman = manager_t::instance();
    tman->clear();

    bool _is_enabled = tman->is_enabled();
    tman->enable(true);
    tman->set_output_stream(std::cout);

    test_timing_thread(12);

    // divide the threaded clocks that are merge
    tman->merge(true);

    bool ign_cutoff;
    print_depth(__FUNCTION__, __LINE__, false);
    print_size(__FUNCTION__, __LINE__);
    tman->report(ign_cutoff = true);
    ASSERT_TRUE(manager_t::instance()->size() >= 36);

    tman->write_serialization("test_output/cxx_timing_thread.json");
    tman->write_missing();
    tman->enable(_is_enabled);
}

//============================================================================//

void test_format()
{
    print_info(__FUNCTION__);

    tim::format::timer::default_format("[%T - %A] : %w, %u, %s, %t, %p%, x%l, %C, %M, %c, %m");
    tim::format::timer::default_unit(tim::units::msec);
    tim::format::timer::default_precision(1);
    tim::format::rss::default_format("[ c, p %A ] : %C, %M");
    tim::format::rss::default_unit(tim::units::kilobyte);
    tim::format::rss::default_precision(0);

    auto tman = manager_t::instance();
    tman->clear();

    bool _is_enabled = tman->is_enabled();
    tman->enable(true);

    tim_timer_t& t = tman->timer("test_format");
    t.start();

    for(auto itr : { 34, 36, 39, 40 })
        time_fibonacci(itr);

    t.stop();

    print_size(__FUNCTION__, __LINE__);
    // reports to stdout
    tman->report();
    tman->set_output_stream("test_output/cxx_timing_format.out");
    // reports to file
    tman->report();
    tman->write_json("test_output/cxx_timing_format.json");
    tman->write_missing();

    EXPECT_EQ(manager_t::instance()->size(), 19);

    for(const auto& itr : *tman)
    {
        ASSERT_FALSE(itr.timer().real_elapsed() < 0.0);
        ASSERT_FALSE(itr.timer().user_elapsed() < 0.0);
    }
    tman->enable(_is_enabled);

    tman->clear();

    tim::rss::usage usage;
    usage.record();

    std::cout << "\nUsage " << usage << std::endl;
}

//============================================================================//

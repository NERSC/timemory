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

#include <timemory/timing_manager.hpp>
#include <timemory/auto_timer.hpp>
#include <timemory/signal_detection.hpp>
#include <timemory/mpi.hpp>

typedef NAME_TIM::timer          tim_timer_t;
typedef NAME_TIM::timing_manager timing_manager_t;

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
void test_timing_pointer();
void test_timing_manager();
void test_timing_toggle();
void test_timing_depth();
void test_timing_thread();

//============================================================================//

int main(int /*argc*/, char** argv)
{
    NAME_TIM::EnableSignalDetection({
                                        NAME_TIM::sys_signal::sHangup,
                                        NAME_TIM::sys_signal::sInterrupt,
                                        NAME_TIM::sys_signal::sIllegal,
                                        NAME_TIM::sys_signal::sSegFault,
                                        NAME_TIM::sys_signal::sFPE
                                    });

    tim_timer_t t = tim_timer_t("Total time");
    t.start();

    int num_fail = 0;
    int num_test = 0;

#define RUN_TEST(func) { try { num_test += 1; func (); } catch(std::exception& e) \
    { std::cerr << e.what() << std::endl; num_fail += 1; } }

    try
    {
        RUN_TEST(test_timing_pointer);
        RUN_TEST(test_timing_manager);
        RUN_TEST(test_timing_toggle);
        RUN_TEST(test_timing_depth);
        RUN_TEST(test_timing_thread);
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

    delete timing_manager_t::instance();

    exit(num_fail);
}

//============================================================================//

void print_info(const std::string& func)
{
    if(NAME_TIM::mpi_rank() == 0)
        std::cout << "\n[" << NAME_TIM::mpi_rank() << "]\e[1;31m TESTING \e[0m["
                  << "\e[1;36m" << func << "\e[0m"
                  << "]...\n" << std::endl;
}

//============================================================================//

void print_size(const std::string& func, int64_t line, bool extra_endl)
{
    if(NAME_TIM::mpi_rank() == 0)
    {
        std::cout << "[" << NAME_TIM::mpi_rank() << "] "
                  << func << "@" << line
                  << " : Timing manager size: "
                  << timing_manager_t::instance()->size()
                  << std::endl;

        if(extra_endl)
            std::cout << std::endl;
    }
}

//============================================================================//

void print_depth(const std::string& func, int64_t line, bool extra_endl)
{
    if(NAME_TIM::mpi_rank() == 0)
    {
        std::cout << "[" << NAME_TIM::mpi_rank() << "] "
                  << func << "@" << line
                  << " : Timing manager size: "
                  << timing_manager_t::instance()->get_max_depth()
                  << std::endl;

        if(extra_endl)
            std::cout << std::endl;
    }
}

//============================================================================//

void test_timing_pointer()
{
    print_info(__FUNCTION__);

    uint16_t set_depth = 4;
    uint16_t get_depth = 0;

    print_depth(__FUNCTION__, __LINE__, false);
    {
        timing_manager_t::instance()->set_max_depth(4);
    }

    print_depth(__FUNCTION__, __LINE__, false);
    {
        get_depth = timing_manager_t::instance()->get_max_depth();
    }

    print_depth(__FUNCTION__, __LINE__, false);
    EXPECT_EQ(set_depth, get_depth);
    timing_manager_t::instance()->set_max_depth(std::numeric_limits<uint16_t>::max());
}

//============================================================================//

void test_timing_manager()
{
    print_info(__FUNCTION__);

    auto tman = timing_manager_t::instance();
    tman->clear();

    bool _is_enabled = tman->is_enabled();
    tman->enable(true);

    tim_timer_t& t = tman->timer("timing_manager_test");
    t.start();

    for(auto itr : { 34, 36, 39, 40, 42, 38, 34, 42 })
        time_fibonacci(itr);

    t.stop();

    print_size(__FUNCTION__, __LINE__);
    tman->report();
    tman->set_output_stream("timing_report.out");
    tman->report();
    tman->write_json("timing_report.json");

    EXPECT_EQ(timing_manager_t::instance()->size(), 32);

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

    auto tman = timing_manager_t::instance();
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
    EXPECT_EQ(timing_manager_t::instance()->size(), 10);

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
    EXPECT_EQ(timing_manager_t::instance()->size(), 0);

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
    EXPECT_EQ(timing_manager_t::instance()->size(), 10);

    tman->enable(_is_enabled);
}

//============================================================================//

void test_timing_depth()
{
    print_info(__FUNCTION__);

    auto tman = timing_manager_t::instance();
    tman->clear();

    bool _is_enabled = tman->is_enabled();
    tman->enable(true);
    tman->set_output_stream(std::cout);

    print_depth(__FUNCTION__, __LINE__, false);
    int32_t _max_depth = tman->get_max_depth();
    tman->set_max_depth(3);
    print_depth(__FUNCTION__, __LINE__, false);
    {
        TIMEMORY_AUTO_TIMER();
        for(auto itr : { 38, 39, 40 })
            time_fibonacci(itr);
    }

    bool no_min;
    print_depth(__FUNCTION__, __LINE__, false);
    print_size(__FUNCTION__, __LINE__);
    tman->report(no_min = true);
    EXPECT_EQ(timing_manager_t::instance()->size(), 7);

    tman->enable(_is_enabled);
    tman->set_max_depth(_max_depth);
}

//============================================================================//

typedef std::vector<std::thread*> thread_list_t;

//============================================================================//

void thread_func(int32_t nfib, std::shared_future<void> fut)
{
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    int32_t nsize = timing_manager_t::instance()->size();
    if(nsize > 0)
        std::cerr << "thread-local timing_manager size: " << nsize << std::endl;

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

void test_timing_thread()
{
    print_info(__FUNCTION__);

    auto tman = timing_manager_t::instance();
    tman->clear();

    bool _is_enabled = tman->is_enabled();
    tman->enable(true);
    tman->set_output_stream(std::cout);

    int num_threads = 12;
    thread_list_t threads(num_threads, nullptr);

    {
        TIMEMORY_AUTO_TIMER();
        {
            std::stringstream ss;
            ss << "[" << num_threads << "_threads]";
            TIMEMORY_AUTO_TIMER(ss.str());

            std::promise<void> prom;
            std::shared_future<void> fut = prom.get_future().share();

            for(auto& itr : threads)
                itr = create_thread(43, fut);

            std::this_thread::sleep_for(std::chrono::milliseconds(2000));

            prom.set_value();

            join_thread(threads.begin(), threads);
        }
    }

    for(auto& itr : threads)
        delete itr;

    threads.clear();

    // divide the threaded clocks that are merge
    tman->merge(true);

    bool no_min;
    print_depth(__FUNCTION__, __LINE__, false);
    print_size(__FUNCTION__, __LINE__);
    tman->report(no_min = true);
    ASSERT_TRUE(timing_manager_t::instance()->size() >= 36);

    tman->enable(_is_enabled);
}

//============================================================================//

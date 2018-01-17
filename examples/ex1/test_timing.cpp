/*
Copyright (c) 2015-2017 by the parties listed in the AUTHORS file.
All rights reserved.  Use of this source code is governed by
a BSD-style license that can be found in the LICENSE file.
*/


#include <cmath>
#include <chrono>
#include <thread>
#include <fstream>
#include <unordered_map>
#include <vector>
#include <iterator>

#ifndef DEBUG
#   define DEBUG
#endif

#include <cassert>

#include <timemory/timing_manager.hpp>
#include <timemory/auto_timer.hpp>
#include <timemory/signal_detection.hpp>

using namespace std;

typedef tim::util::timer          tim_timer_t;
typedef tim::util::timing_manager timing_manager_t;

typedef std::chrono::duration<int64_t>                      seconds_type;
typedef std::chrono::duration<int64_t, std::milli>          milliseconds_type;
typedef std::chrono::duration<int64_t, std::ratio<60*60>>   hours_type;

// ASSERT_NEAR
// EXPECT_EQ
// EXPECT_FLOAT_EQ
// EXPECT_DOUBLE_EQ

#define EXPECT_EQ(lhs, rhs) if(lhs != rhs) { \
    std::stringstream ss; ss << #lhs << " != " << #rhs << " @ line " << __LINE__ \
    << " of " << __FILE__; \
    throw std::runtime_error(ss.str()); }

#define ASSERT_FALSE(expr) assert(!(expr))

//----------------------------------------------------------------------------//
// fibonacci calculation
int64_t fibonacci(int32_t n)
{
    if (n < 2) return n;
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

void test_timing_manager();
void test_timing_toggle();
void test_timing_thread();

//============================================================================//

int main()
{
    tim::EnableSignalDetection();

    int num_fail = 0;
    int num_test = 0;

#define RUN_TEST(func) \
    try \
    { \
        num_test += 1; \
        func (); \
    } \
    catch(std::exception& e) \
    { \
        std::cerr << e.what() << std::endl; \
        num_fail += 1; \
    }

    RUN_TEST(test_timing_manager);
    RUN_TEST(test_timing_toggle);
    RUN_TEST(test_timing_thread);

    std::cout << "\nDone.\n" << std::endl;

    if(num_fail > 0)
        std::cout << "Tests failed: " << num_fail << "/" << num_test << std::endl;
    else
        std::cout << "Tests passed: " << (num_test - num_fail) << "/" << num_test
                  << std::endl;

    exit(num_fail);
}
//============================================================================//

void test_timing_manager()
{
    std::cout << "\nTesting " << __FUNCTION__ << "...\n" << std::endl;

    timing_manager_t* tman = timing_manager_t::instance();
    tman->clear();

    bool _is_enabled = tman->is_enabled();
    tman->enable(true);

    tim_timer_t& t = tman->timer("timing_manager_test");
    t.start();

    for(auto itr : { 35, 37, 39, 41, 43, 39, 35, 43 })
        time_fibonacci(itr);

    t.stop();

    tman->report();
    tman->set_output_stream("timing_report.out");
    tman->report();
    tman->write_json("timing_report.json");

    EXPECT_EQ(timing_manager_t::instance()->size(), 6);

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
    std::cout << "\nTesting " << __FUNCTION__ << "...\n" << std::endl;

    timing_manager_t* tman = timing_manager_t::instance();
    tman->clear();

    bool _is_enabled = tman->is_enabled();
    tman->enable(true);
    tman->set_output_stream(std::cout);

    tman->enable(true);
    {
        TIMEMORY_AUTO_TIMER("@toggle_on");
        time_fibonacci(43);
    }
    tman->report();
    EXPECT_EQ(timing_manager_t::instance()->size(), 2);

    tman->clear();
    tman->enable(false);
    {
        TIMEMORY_AUTO_TIMER("@toggle_off");
        time_fibonacci(41);
        //std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    tman->report();
    EXPECT_EQ(timing_manager_t::instance()->size(), 0);

    tman->clear();
    tman->enable(true);
    {
        TIMEMORY_AUTO_TIMER("@toggle_on");
        time_fibonacci(43);
        tman->enable(false);
        TIMEMORY_AUTO_TIMER("@toggle_off");
        time_fibonacci(39);
    }
    tman->report();
    EXPECT_EQ(timing_manager_t::instance()->size(), 2);

    tman->enable(_is_enabled);
}

//============================================================================//

typedef std::vector<std::thread*> thread_list_t;

//============================================================================//

std::thread* create_thread(int32_t nfib)
{
    TIMEMORY_AUTO_TIMER();
    static int32_t n = 0;
    return new std::thread(time_fibonacci, nfib + (n++)%2);
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
    std::cout << "\nTesting " << __FUNCTION__ << "...\n" << std::endl;
    timing_manager_t* tman = timing_manager_t::instance();
    tman->clear();

    bool _is_enabled = tman->is_enabled();
    tman->enable(true);
    tman->set_output_stream(std::cout);

    int num_threads = 16;
    thread_list_t threads(num_threads, nullptr);

    {
        TIMEMORY_AUTO_TIMER();
        {
            std::stringstream ss;
            ss << "@" << num_threads << "_threads";
            TIMEMORY_AUTO_TIMER(ss.str());

            for(auto& itr : threads)
                itr = create_thread(42);

            join_thread(threads.begin(), threads);
        }
    }

    bool no_min = true;
    tman->report(no_min);

    std::cout << "Timing manager size: " << timing_manager_t::instance()->size()
              << std::endl;

    for(auto& itr : threads)
        delete itr;

    threads.clear();

    tman->enable(_is_enabled);

    EXPECT_EQ(timing_manager_t::instance()->size(), 21);
}
//============================================================================//

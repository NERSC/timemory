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

// C headers
#include <cmath>
#include <cassert>

// C++ headers
#include <chrono>
#include <thread>
#include <fstream>
#include <unordered_map>
#include <vector>
#include <iterator>
#include <future>

// TiMemory headers
#include <timemory/manager.hpp>
#include <timemory/auto_timer.hpp>
#include <timemory/signal_detection.hpp>
#include <timemory/mpi.hpp>

typedef tim::timer      tim_timer_t;
typedef tim::manager    manager_t;

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
int64_t fibonacci(uint64_t n)
{
    return (n < 2) ? n : (fibonacci(n-1) + fibonacci(n-2));
}

//----------------------------------------------------------------------------//

void print_info(const std::string&);
void print_size(const std::string&, int64_t, bool = true);

//============================================================================//

uint64_t run_total_test(int _sleep, uint64_t nfib)
{
    print_info(__FUNCTION__);
    print_size(__FUNCTION__, __LINE__, false);
    // check total already exists
    EXPECT_EQ(tim::manager::instance()->size(), 1);

    std::this_thread::sleep_for(std::chrono::seconds(_sleep));
    uint64_t n = fibonacci(nfib);

    // check total still only entry that exists
    print_size(__FUNCTION__, __LINE__, false);
    EXPECT_EQ(tim::manager::instance()->size(), 1);

    return n;
}

//============================================================================//

int main(int argc, char** argv)
{
    tim::enable_signal_detection();

    int sleep_seconds = 2;
    int nfib = 40;
    if(argc > 1)
        sleep_seconds = atoi(argv[1]);

    if(argc > 2)
        nfib = atoi(argv[2]);

    int num_fail = 0;
    int num_test = 0;
    uint64_t result = 0;

#define RUN_TEST(func, _assign, _sleep, _nfib) { \
    try { num_test += 1; _assign = func ( _sleep, _nfib ); } catch(std::exception& e) \
    { std::cerr << e.what() << std::endl; num_fail += 1; } }

    try
    {
        RUN_TEST(run_total_test, result, sleep_seconds, nfib);
    }
    catch(std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }

    std::cout << "[" << argv[0] << "] fibonacci(" << nfib << ") = " << result
              << std::endl;
    std::cout << (*tim::manager::instance()) << std::endl;

    if(num_fail > 0)
        std::cout << "Tests failed: " << num_fail << "/" << num_test << std::endl;
    else
        std::cout << "Tests passed: " << (num_test - num_fail) << "/" << num_test
                  << std::endl;

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

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

#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <future>
#include <iterator>
#include <thread>
#include <unordered_map>
#include <vector>

#include <timemory/auto_timer.hpp>
#include <timemory/auto_tuple.hpp>
#include <timemory/component_tuple.hpp>
#include <timemory/environment.hpp>
#include <timemory/manager.hpp>
#include <timemory/mpi.hpp>
#include <timemory/papi.hpp>
#include <timemory/rusage.hpp>
#include <timemory/signal_detection.hpp>
#include <timemory/testing.hpp>

using namespace tim::component;

using auto_tuple_t = tim::auto_tuple<real_clock, system_clock, thread_cpu_clock,
                                     thread_cpu_util, process_cpu_clock, process_cpu_util,
                                     peak_rss, current_rss, papi_event<PAPI_TOT_CYC, 1>>;

//--------------------------------------------------------------------------------------//
// fibonacci calculation
intmax_t
fibonacci(int32_t n)
{
    return (n < 2) ? n : fibonacci(n - 1) + fibonacci(n - 2);
}
//--------------------------------------------------------------------------------------//
// time fibonacci with return type and arguments
// e.g. std::function < int32_t ( int32_t ) >
intmax_t
time_fibonacci(int32_t n)
{
    return fibonacci(n);
}
//--------------------------------------------------------------------------------------//

void
print_info(const std::string&);
void
print_string(const std::string& str);
void
test_1_usage();
void
test_2_timing();
void
test_3_auto_tuple();

//======================================================================================//

int
main(int argc, char** argv)
{
    tim::env::parse();
    tim::standard_timing_components_t timing;
    timing.start();

    papi_event<PAPI_L1_DCM, 0>  evt_l1_dcm;
    papi_event<PAPI_L1_ICM, 0>  evt_l1_icm;
    papi_event<PAPI_L1_TCM, 0>  evt_l1_tcm;
    papi_event<PAPI_TOT_CYC, 1> evt_tot_cyc;
    /*
    tim::papi::init();
    // tim::papi::set_debug(2);
    std::size_t nevents   = 4;
    int*        event_set = new int[1];
    long long*  values    = new long long[nevents];
    memset(event_set, 0, sizeof(int));
    memset(values, 0, nevents * sizeof(long long));
    tim::papi::add_event(*event_set, PAPI_L1_DCM);
    tim::papi::add_event(*event_set, PAPI_L1_ICM);
    tim::papi::add_event(*event_set, PAPI_L1_TCM);
    tim::papi::add_event(*event_set, PAPI_TOT_CYC);
    tim::papi::start(*event_set);
    */
    evt_l1_dcm.start();
    evt_l1_icm.start();
    evt_l1_tcm.start();
    evt_tot_cyc.start();

    CONFIGURE_TEST_SELECTOR(3);

    int num_fail = 0;
    int num_test = 0;

    std::cout << "# tests: " << tests.size() << std::endl;
    try
    {
        RUN_TEST(1, test_1_usage, num_test, num_fail);
        RUN_TEST(2, test_2_timing, num_test, num_fail);
        RUN_TEST(3, test_3_auto_tuple, num_test, num_fail);
    }
    catch(std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }

    timing.stop();
    std::cout << "\nTests runtime: " << timing << std::endl;

    evt_l1_dcm.stop();
    evt_l1_icm.stop();
    evt_l1_tcm.stop();
    evt_tot_cyc.stop();

    std::cout << evt_l1_dcm << std::endl;
    std::cout << evt_l1_icm << std::endl;
    std::cout << evt_l1_tcm << std::endl;
    std::cout << evt_tot_cyc << std::endl;

    /*
    tim::papi::read(*event_set, values);
    for(std::size_t i = 0; i < nevents; ++i)
    {
        std::cout << "PAPI value [" << i << "] = " << values[i] << std::endl;
    }
    tim::papi::stop(*event_set, values);
    for(std::size_t i = 0; i < nevents; ++i)
    {
        std::cout << "PAPI value [" << i << "] = " << values[i] << std::endl;
    }
    delete[] event_set;
    delete[] values;
    */

    TEST_SUMMARY(argv[0], num_test, num_fail);

    exit(num_fail);
}

//======================================================================================//

void
print_info(const std::string& func)
{
    if(tim::mpi_rank() == 0)
        std::cout << "\n[" << tim::mpi_rank() << "]\e[1;33m TESTING \e[0m["
                  << "\e[1;36m" << func << "\e[0m"
                  << "]...\n"
                  << std::endl;
}

//======================================================================================//

void
print_string(const std::string& str)
{
    std::stringstream _ss;
    _ss << "[" << tim::mpi_rank() << "] " << str << std::endl;
    std::cout << _ss.str();
}

//======================================================================================//

template <typename _Tp>
void
serialize(const std::string& fname, const std::string& title, const _Tp& obj)
{
    static constexpr auto spacing = cereal::JSONOutputArchive::Options::IndentChar::space;
    std::stringstream     ss;
    {
        // ensure json write final block during destruction before the file is closed
        //                                  args: precision, spacing, indent size
        cereal::JSONOutputArchive::Options opts(12, spacing, 4);
        cereal::JSONOutputArchive          oa(ss, opts);
        oa(cereal::make_nvp(title, obj));
    }
    std::ofstream ofs(fname.c_str());
    ofs << ss.str() << std::endl;
}

//======================================================================================//

void
test_1_usage()
{
    print_info(__FUNCTION__);
    TIMEMORY_AUTO_TUPLE(auto_tuple_t, "");

    typedef tim::component_tuple<peak_rss, current_rss, stack_rss, data_rss, num_swap,
                                 num_io_in, num_io_out, num_minor_page_faults,
                                 num_major_page_faults>
        measurement_t;

    measurement_t _use_beg;
    measurement_t _use_delta;
    measurement_t _use_end;

    _use_beg.record();
    _use_delta.start();
    fibonacci(30);
    _use_delta.stop();
    _use_end.record();

    std::cout << "usage (begin): " << _use_beg << std::endl;
    std::cout << "usage (delta): " << _use_delta << std::endl;
    std::cout << "usage (end):   " << _use_end << std::endl;

    std::vector<std::pair<std::string, measurement_t>> measurements = {
        { "begin", _use_beg }, { "delta", _use_delta }, { "end", _use_end }
    };
    serialize("rusage.json", "usage", measurements);
}

//======================================================================================//

void
test_2_timing()
{
    print_info(__FUNCTION__);

    typedef tim::component_tuple<real_clock, system_clock, user_clock, cpu_clock,
                                 cpu_util, thread_cpu_clock, thread_cpu_util,
                                 process_cpu_clock, process_cpu_util, monotonic_clock,
                                 monotonic_raw_clock>
        measurement_t;
    using pair_t = std::pair<std::string, measurement_t>;

    static std::mutex    mtx;
    std::deque<pair_t>   measurements;
    measurement_t        runtime;
    std::atomic_intmax_t ret;
    std::stringstream    lambda_ss;

    {
        TIMEMORY_AUTO_TUPLE(auto_tuple_t, "");

        auto run_fib = [&](long n) {
            TIMEMORY_AUTO_TUPLE(auto_tuple_t, "");
            measurement_t _tm;
            _tm.start();
            ret += fibonacci(n);
            _tm.stop();
            mtx.lock();
            std::stringstream ss;
            ss << "fibonacci(" << n << ")";
            measurements.push_back(pair_t(ss.str(), _tm));
            lambda_ss << "thread fibonacci(" << n << "): " << _tm << std::endl;
            mtx.unlock();
        };

        runtime.start();
        {
            std::thread _t1(run_fib, 43);
            std::thread _t2(run_fib, 43);

            run_fib(40);

            _t1.join();
            _t2.join();
        }
        runtime.stop();
    }

    std::cout << "\n" << lambda_ss.str() << std::endl;
    std::cout << "total runtime: " << runtime << std::endl;
    std::cout << "fibonacci total: " << ret.load() << "\n" << std::endl;

    measurements.push_front(pair_t("run", runtime));
    serialize("timing.json", "runtime", measurements);
}

//======================================================================================//

void
test_3_auto_tuple()
{
    peak_rss prss;
    // just record the peak rss
    prss.measure();
    std::cout << "Current peak rss: " << prss << std::endl;

    prss.start();
    // do something, where you want delta peak rss
    prss.stop();
    std::cout << "Change in peak rss: " << prss << std::endl;

    print_info(__FUNCTION__);

    // measure multiple clock time + resident set sizes
    using full_set_t =
        tim::auto_tuple<real_clock, thread_cpu_clock, thread_cpu_util, process_cpu_clock,
                        process_cpu_util, peak_rss, current_rss>;
    // measure wall-clock, thread cpu-clock + process cpu-utilization
    using small_set_t = tim::auto_tuple<real_clock, thread_cpu_clock, process_cpu_util>;

    std::atomic_intmax_t ret;
    {
        // accumulate metrics on full run
        TIMEMORY_BASIC_AUTO_TUPLE(full_set_t, "[total]");

        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        // run a fibonacci calculation and accumulate metric
        auto run_fibonacci = [&](long n) {
            auto man = tim::manager::instance();
            TIMEMORY_AUTO_TUPLE(small_set_t, "[fibonacci_" + std::to_string(n) + "]");
            ret += fibonacci(n);
        };

        // run shorter fibonacci calculations on two threads
        std::thread t(run_fibonacci, 42);
        // run longer fibonacci calculation on main thread
        run_fibonacci(43);

        t.join();
    }
    std::cout << "\nfibonacci total: " << ret.load() << "\n" << std::endl;
}

//======================================================================================//

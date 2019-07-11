// MIT License
//
// Copyright (c) 2019, The Regents of the University of California,
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

#include <chrono>
#include <iostream>
#include <thread>
#include <timemory/signal_detection.hpp>
#include <timemory/testing.hpp>
#include <timemory/timemory.hpp>

using namespace tim::component;
using float_type   = double;
using roofline_t   = cpu_roofline<float_type, PAPI_DP_OPS>;
using auto_tuple_t = tim::auto_tuple<real_clock, cpu_clock, roofline_t>;
using auto_list_t  = tim::auto_list<real_clock, cpu_clock, roofline_t>;

//--------------------------------------------------------------------------------------//

float_type
fib(float_type n);
void
check(auto_list_t& l);
void
check_const(const auto_list_t& l);

//--------------------------------------------------------------------------------------//

int
main(int argc, char** argv)
{
    tim::settings::json_output() = true;
    tim::timemory_init(argc, argv);
    tim::print_env();

    using comp_tuple_t = typename auto_tuple_t::component_type;
    comp_tuple_t _main("overall timer", true);

    roofline_t::operation_function_t roof_func = []() {
        using _Tp     = float_type;
        auto add_func = [](_Tp& a, const _Tp& b, const _Tp& c) { a = b + c; };
        auto fma_func = [](_Tp& a, const _Tp& b, const _Tp& c) { a = a * b + c; };
        auto l1_size  = tim::ert::cache_size::get<1>();
        auto l2_size  = tim::ert::cache_size::get<2>();
        auto l3_size  = tim::ert::cache_size::get<3>();
        std::cout << "[INFO]> L1 cache size: " << (l1_size / tim::units::kilobyte)
                  << " KB, L2 cache size: " << (l2_size / tim::units::kilobyte)
                  << " KB, L3 cache size: " << (l3_size / tim::units::kilobyte) << " KB\n"
                  << std::endl;
        tim::ert::exec_params params(16, 8 * l3_size);
        auto op_counter = new tim::ert::cpu::operation_counter<_Tp>(params, 64);
        tim::ert::cpu_ops_main<1>(*op_counter, add_func);
        tim::ert::cpu_ops_main<4>(*op_counter, fma_func);
        return op_counter;
    };
    roofline_t::get_finalize_function() = roof_func;

    _main.start();
    for(auto n : { 35, 38, 44 })
    {
        auto label = tim::str::join("", "fibonacci(", n, ")");
        TIMEMORY_BLANK_AUTO_TUPLE(auto_tuple_t, label);
        auto ret = fib(n);
        printf("fib(%i) = %.2f\n", n, ret);
    }
    _main.stop();

    std::cout << "\n" << _main << "\n" << std::endl;
    auto_list_t l(__FUNCTION__, false);
    check(l);
    check_const(l);
    std::cout << std::endl;
}

//--------------------------------------------------------------------------------------//

#define ftwo static_cast<float_type>(2.0)
#define fone static_cast<float_type>(1.0)

float_type
fib(float_type n)
{
    return (n < ftwo) ? n : fone * (fib(n - 1) + fib(n - 2));
}

//--------------------------------------------------------------------------------------//

void
check_const(const auto_list_t& l)
{
    const real_clock* rc = l.get<real_clock>();
    std::cout << "type: " << tim::demangle(typeid(rc).name()) << std::endl;
}

//--------------------------------------------------------------------------------------//

void
check(auto_list_t& l)
{
    real_clock* rc = l.get<real_clock>();
    std::cout << "type: " << tim::demangle(typeid(rc).name()) << std::endl;
}

//--------------------------------------------------------------------------------------//

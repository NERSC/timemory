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
//

#if defined(DEBUG) && !defined(VERBOSE)
#    define VERBOSE
#endif

#include "ex_gotcha_lib.hpp"
#include "timemory/timemory.hpp"

#include <array>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <set>
#include <string>
#include <vector>

using namespace tim;
using namespace tim::component;
using std::cout;
using std::vector;

//======================================================================================//

static auto&
get_intercepts()
{
    static uint64_t _instance{ 0 };
    return _instance;
}

//======================================================================================//

namespace tim
{
namespace component
{
struct exp_intercept : public base<exp_intercept, void>
{
    double operator()(double val)
    {
#if defined(VERBOSE)
        if(tim::settings::verbose() > 0)
            printf("\texecuting modified exp function : %20.3f...", val);
#endif
        ++get_intercepts();
        return exp(val);
    }
};
}  // namespace component
}  // namespace tim

//======================================================================================//

using wc_t         = tim::component_bundle<TIMEMORY_API, wall_clock>;
using exptime_t    = gotcha<2, wc_t>;
using exp2expf_t   = gotcha<1, std::tuple<>, exp_intercept>;
using exp_bundle_t = tim::component_bundle<TIMEMORY_API, exp2expf_t*, exptime_t*>;

using exp2expf_ot = typename exp2expf_t::operator_type;
static_assert(std::is_same<exp2expf_ot, exp_intercept>::value,
              "exp2expf_t operator_type is not exp_intercept");
static_assert(exp2expf_t::components_size == 0, "exp2expf_t should have no components");
static_assert(exp2expf_t::differentiator_is_component, "exp2expf_t won't replace exp");

using exptime_ot = typename exptime_t::operator_type;
using exptime_ct = typename exptime_t::component_type;
static_assert(std::is_same<exptime_ot, void>::value,
              "exptime_t operator_type is not exp_intercept");
static_assert(exptime_t::components_size == 1, "exptime_t should have no components");
static_assert(std::is_same<exptime_ct, wc_t>::value,
              "exptime_t has incorrect components");
static_assert(!exptime_t::differentiator_is_component, "exp2expf_t won't replace exp");

//======================================================================================//

extern "C" double
exp(double);

extern double
sum_exp(const vector<double>&);

//======================================================================================//

static auto use_intercept = tim::get_env("EXP_INTERCEPT", true);
static auto use_timers    = tim::get_env("EXP_TIMERS", true);

bool
init()
{
    //
    // configure the initializer for the gotcha component which replaces exp with expf
    //
    exp2expf_t::get_initializer() = []() {
        puts("Generating exp intercept...");
        TIMEMORY_C_GOTCHA(exp2expf_t, 0, exp);
    };

    //
    // configure the initializer for the gotcha components which places wall-clock
    // timers around exp and sum_exp
    //
    exptime_t::get_initializer() = []() {
        puts("Generating exp timers...");
        TIMEMORY_C_GOTCHA(exptime_t, 0, exp);
        TIMEMORY_CXX_GOTCHA(exptime_t, 1, sum_exp);
    };

    exp_bundle_t::get_initializer() = [=](exp_bundle_t& obj) {
        if(use_intercept) obj.initialize<exp2expf_t>();
        if(use_timers) obj.initialize<exptime_t>();
    };

    return true;
}
//
//  static initialization will run the init function
//
static auto did_init = init();

//======================================================================================//

int
main(int argc, char** argv)
{
    if(!did_init)
        throw std::runtime_error("Error! static initialization did not execute!");

    tim::timemory_init(argc, argv);

    uint64_t n = 100000;
    if(argc > 1) n = atoi(argv[1]);

    double ret = 0.0;
    for(uint64_t i = 0; i < n; ++i)
    {
        exp_bundle_t obj("example");
        obj.start();
        ret += sum_exp({ i + 1.0, 2.0 * (i + 1.0) });
        obj.stop();
    }

    auto sz = tim::storage<wall_clock>::instance()->size();
    std::cout << "\nusing intercept    : " << std::boolalpha << use_intercept << '\n';
    std::cout << "exp -> expf        : " << get_intercepts() << "x" << std::endl;
    std::cout << "using timers       : " << std::boolalpha << use_timers << '\n';
    std::cout << "wall_clock records : " << sz << '\n' << std::endl;

    tim::timemory_finalize();

    auto rc_intercept =
        (use_intercept && get_intercepts() != 2 * n) ? EXIT_FAILURE : EXIT_SUCCESS;
    auto rc_timers = (use_timers && sz == 0) ? EXIT_FAILURE : EXIT_SUCCESS;

    return rc_intercept + rc_timers;
}

//======================================================================================//

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
struct exp_replace : public base<exp_replace, void>
{
    double operator()(double val)
    {
#if defined(VERBOSE)
        if(tim::settings::verbose() > 0)
            printf("\texecuting modified exp function : %20.3f...", val);
#endif
        ++get_intercepts();
        return static_cast<double>(expf(static_cast<float>(val)));
    }
};
}  // namespace component
}  // namespace tim

//======================================================================================//

using wc_t         = tim::component_bundle<TIMEMORY_API, wall_clock>;
using exp_time_t   = gotcha<2, wc_t>;
using exp_repl_t   = gotcha<2, std::tuple<>, exp_replace>;
using exp_bundle_t = tim::component_bundle<TIMEMORY_API, exp_repl_t*, exp_time_t*>;

using exp2expf_ot = typename exp_repl_t::operator_type;
static_assert(std::is_same<exp2expf_ot, exp_replace>::value,
              "exp_repl_t operator_type is not exp_replace");
static_assert(exp_repl_t::components_size == 0, "exp_repl_t should have no components");
static_assert(exp_repl_t::differ_is_component, "exp_repl_t won't replace exp");

using exptime_ot = typename exp_time_t::operator_type;
using exptime_ct = typename exp_time_t::bundle_type;
static_assert(std::is_same<exptime_ot, void>::value,
              "exp_time_t operator_type is not exp_replace");
static_assert(exp_time_t::components_size == 1, "exp_time_t should have no components");
static_assert(std::is_same<exptime_ct, wc_t>::value,
              "exp_time_t has incorrect components");
static_assert(!exp_time_t::differ_is_component, "exp_repl_t won't replace exp");

//======================================================================================//

extern "C" double
exp(double);

extern double
sum_exp(const vector<double>&);

//======================================================================================//

static auto use_intercept = tim::get_env("EXP_REPLACE", true);
static auto use_timers    = tim::get_env("EXP_TIMERS", true);

bool
init_gotcha()
{
    //
    // configure the initializer for the gotcha component which replaces exp with expf
    //
    exp_repl_t::get_initializer() = []() {
        puts("Generating exp intercept...");
        TIMEMORY_C_GOTCHA(exp_repl_t, 0, exp);
        TIMEMORY_DERIVED_GOTCHA(exp_repl_t, 1, exp, "__exp_finite");
        // exp might actually resolve to above symbol
    };

    //
    // configure the initializer for the gotcha components which places wall-clock
    // timers around exp and sum_exp
    //
    exp_time_t::get_initializer() = []() {
        puts("Generating exp timers...");
        TIMEMORY_C_GOTCHA(exp_time_t, 0, exp);
        TIMEMORY_CXX_GOTCHA(exp_time_t, 1, sum_exp);
    };

    exp_bundle_t::get_initializer() = [=](exp_bundle_t& obj) {
        if(use_intercept) obj.initialize<exp_repl_t>();
        if(use_timers) obj.initialize<exp_time_t>();
    };

    return true;
}

//======================================================================================//

int
main(int argc, char** argv)
{
    if(!init_gotcha())
        throw std::runtime_error("Error! initialization failed!");

    tim::timemory_init(argc, argv);

    uint64_t n = 1000;
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

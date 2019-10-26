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

#include "ex_gotcha_lib.hpp"

#include <timemory/timemory.hpp>

#include <chrono>
#include <cmath>
#include <string>
#include <thread>

namespace ext
{
//--------------------------------------------------------------------------------------//

template <typename _Tp, typename _Func, typename _Incr>
_Tp
work(const std::string& fname, int nitr, _Func&& func, _Incr&& incr)
{
    _Tp val = 2.0;
    _Tp sum = 0.0;

    for(int i = 0; i < nitr; ++i)
    {
        sum += func(val);
        printf("\t[itr: %i]> %-6s %-4s(%6.3f) = %15.2f\n", i,
               tim::demangle(typeid(_Tp).name()).c_str(), fname.c_str(), val, sum);
        val = incr(val, i + 1);
    }
    return sum;
}

//--------------------------------------------------------------------------------------//

tuple_t
do_exp_work(int nitr)
{
    printf("\n");
    PRINT_HERE("");
    printf("\n");

    auto fsum = work<float>("expf", nitr, [](float val) -> float { return expf(val); },
                            [](float val, int i) -> float { return val + 0.25 * i; });

    printf("\n");

    auto dsum = work<double>("exp", nitr, [](double val) -> double { return exp(val); },
                             [](double val, int i) -> double { return val + 0.25 * i; });

    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    return tuple_t(fsum, dsum);
}

//--------------------------------------------------------------------------------------//

tuple_t
do_cos_work(int nitr, const std::pair<float, double>& pair)
{
    printf("\n");
    PRINT_HERE("");
    printf("\n");

    auto fsum = work<float>("cosf", nitr, [](float val) -> float { return cosf(val); },
                            [](float val, int i) -> float { return val + 0.25 * i; });

    printf("\n");

    auto dsum = work<double>("cos", nitr, [](double val) -> double { return cos(val); },
                             [](double val, int i) -> double { return val + 0.25 * i; });

    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    auto _pair = pair;
    auto _R    = ext::do_cos_work_ref(nitr, _pair);

    printf("\n");
    printf("[iterations=%i]> (R)  single-precision cos = %f\n", nitr, std::get<0>(_R));
    printf("[iterations=%i]> (R)  double-precision cos = %f\n", nitr, std::get<1>(_R));

    return tuple_t(fsum, dsum);
}

//--------------------------------------------------------------------------------------//

tuple_t
do_cos_work_ref(int nitr, std::pair<float, double>& _pair)
{
    printf("\n");
    PRINT_HERE("");
    printf("\n");

    auto fsum = work<float>("cosf", nitr, [](float val) -> float { return cosf(val); },
                            [](float val, int i) -> float { return val + 0.25 * i; });

    printf("\n");

    auto dsum = work<double>("cos", nitr, [](double val) -> double { return cos(val); },
                             [](double val, int i) -> double { return val + 0.25 * i; });

    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    auto _RK = ext::do_cos_work_cref(nitr, _pair);

    printf("\n");
    printf("[iterations=%i]> (RK) single-precision cos = %f\n", nitr, std::get<0>(_RK));
    printf("[iterations=%i]> (RK) double-precision cos = %f\n", nitr, std::get<1>(_RK));

    return tuple_t(fsum, dsum);
}

//--------------------------------------------------------------------------------------//

tuple_t
do_cos_work_cref(int nitr, const std::pair<float, double>&)
{
    printf("\n");
    PRINT_HERE("");
    printf("\n");

    auto fsum = work<float>("cosf", nitr, [](float val) -> float { return cosf(val); },
                            [](float val, int i) -> float { return val + 0.25 * i; });

    printf("\n");

    auto dsum = work<double>("cos", nitr, [](double val) -> double { return cos(val); },
                             [](double val, int i) -> double { return val + 0.25 * i; });

    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    return tuple_t(fsum, dsum);
}

//--------------------------------------------------------------------------------------//

}  // namespace ext

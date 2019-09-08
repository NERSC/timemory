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

#if !defined(DEBUG)
#    define DEBUG
#endif

#include "test_gotcha_lib.hpp"
#include <timemory/timemory.hpp>

#include <array>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <set>
#include <string>
#include <unistd.h>
#include <vector>

//--------------------------------------------------------------------------------------//
// make the namespace usage a little clearer
//
namespace settings
{
using namespace tim::settings;
}
namespace mpi
{
using namespace tim::mpi;
}
namespace trait
{
using namespace tim::trait;
}

//======================================================================================//

using namespace tim::component;

using auto_timer_t = tim::component_tuple<real_clock, cpu_clock>;
// this should fail
// using auto_tuple_t = tim::component_tuple<real_clock, cpu_clock, cpu_util, gotcha<1,
// auto_timer_t, int>>;
using auto_tuple_t = tim::component_tuple<real_clock, cpu_clock, peak_rss>;

constexpr size_t _Pn = 3;
constexpr size_t _Mn = 15;
constexpr size_t _Sn = 8;
constexpr size_t _Cn = 12;

using put_gotcha_t = tim::component::gotcha<_Pn, auto_timer_t>;
using std_gotcha_t = tim::component::gotcha<_Sn, auto_timer_t, int>;
using cos_gotcha_t = tim::component::gotcha<_Cn, auto_timer_t>;
using mpi_gotcha_t = tim::component::gotcha<_Mn, auto_tuple_t>;

//======================================================================================//

template <typename _Tp>
struct type_id
{
    using string_t = std::string;
    static std::string name() { return typeid(_Tp).name(); }
};

template <typename _Tp>
struct type_id<const _Tp>
{
    using string_t = std::string;
    static std::string name() { return string_t("K") + typeid(_Tp).name(); }
};

template <typename _Tp>
struct type_id<const _Tp&>
{
    using string_t = std::string;
    static std::string name() { return string_t("RK") + typeid(_Tp).name(); }
};

template <typename _Tp>
struct type_id<_Tp&>
{
    using string_t = std::string;
    static std::string name() { return string_t("R") + typeid(_Tp).name(); }
};

//======================================================================================//

template <typename... _Args>
struct cxx_mangler
{
    static std::string mangle(std::string func)
    {
        std::string ret   = "_Z";
        auto        delim = tim::delimit(func, ":()<>");
        if(delim.size() > 1) ret += "N";
        for(const auto& itr : delim)
        {
            ret += std::to_string(itr.length());
            ret += itr;
        }
        ret += "E";
        auto arg_string = TIMEMORY_JOIN("", type_id<_Args>::name()...);
        ret += arg_string;
        printf("[generated_mangle]> %s --> %s\n", func.c_str(), ret.c_str());
        return ret;
    }
};

template <typename... _Args>
struct cxx_mangler<std::tuple<_Args...>>
{
    static std::string mangle(std::string func)
    {
        return cxx_mangler<_Args...>::mangle(func);
    }
};

template <typename _Func,
          typename _Tuple = typename tim::function_traits<_Func>::arg_tuple>
std::string
mangle(const std::string& func)
{
    return cxx_mangler<_Tuple>::mangle(func);
}

//======================================================================================//

void
init()
{
    // Mangled name generation:
    /**
    TIMEMORY_INC=../source
    CEREAL_INC=../external/cereal/include
    SRC_DIR=../examples/ex-gotcha
    FILE=test_gotcha_lib
    g++ -S -fverbose-asm -I${TIMEMORY_INC} -I${CEREAL_INC} ${SRC_DIR}/${FILE}.cpp
    as -alhnd ${FILE}.s > ${FILE}.asm
    grep '\.globl' ${FILE}.asm
    **/

    std::string exp_func   = "ext::do_exp_work";
    std::string cos_func   = "ext::do_cos_work";
    std::string cosR_func  = "ext::do_cos_work_ref";
    std::string cosRK_func = "ext::do_cos_work_cref";

    std::string real_exp_mangle   = "_ZN3ext10do_exp_workEi";
    std::string real_cos_mangle   = "_ZN3ext11do_cos_workEiRKSt4pairIfdE";
    std::string real_cosR_mangle  = "_ZN3ext15do_cos_work_refEiRSt4pairIfdE";
    std::string real_cosRK_mangle = "_ZN3ext16do_cos_work_crefEiRKSt4pairIfdE";

    std::string test_exp_mangle   = mangle<decltype(ext::do_exp_work)>(exp_func);
    std::string test_cos_mangle   = mangle<decltype(ext::do_cos_work)>(cos_func);
    std::string test_cosR_mangle  = mangle<decltype(ext::do_cos_work_ref)>(cosR_func);
    std::string test_cosRK_mangle = mangle<decltype(ext::do_cos_work_cref)>(cosRK_func);

    printf("[real]>      %24s  -->  %s\n", exp_func.c_str(), real_exp_mangle.c_str());
    printf("[test]>      %24s  -->  %s\n", exp_func.c_str(), test_exp_mangle.c_str());
    printf("[real]>      %24s  -->  %s\n", cos_func.c_str(), real_cos_mangle.c_str());
    printf("[test]>      %24s  -->  %s\n", cos_func.c_str(), test_cos_mangle.c_str());
    printf("[real]> (R)  %24s  -->  %s\n", cosR_func.c_str(), real_cosR_mangle.c_str());
    printf("[test]> (R)  %24s  -->  %s\n", cosR_func.c_str(), test_cosR_mangle.c_str());
    printf("[real]> (RK) %24s  -->  %s\n", cosRK_func.c_str(), real_cosRK_mangle.c_str());
    printf("[test]> (RK) %24s  -->  %s\n", cosRK_func.c_str(), test_cosRK_mangle.c_str());


    put_gotcha_t::configure<0, int, const char*>("puts");

    // TIMEMORY_GOTCHA(std_gotcha_t, 0, cosf);
    // TIMEMORY_GOTCHA(std_gotcha_t, 2, expf);
    std_gotcha_t::configure<1, double, double>("cos");
    std_gotcha_t::configure<3, double, double>("exp");
    std_gotcha_t::configure<2, ext::tuple_t, int>(test_exp_mangle);

    cos_gotcha_t::configure<1, ext::tuple_t, int, ext::tuple_t>(test_cos_mangle);
    cos_gotcha_t::configure<2, ext::tuple_t, int, ext::tuple_t>(test_cosR_mangle);
    cos_gotcha_t::configure<3, ext::tuple_t, int, ext::tuple_t>(test_cosRK_mangle);

    // mpi_gotcha_t::configure<0, int, int*, char***>("MPI_Init");
    TIMEMORY_GOTCHA(mpi_gotcha_t, 0, MPI_Init);
    TIMEMORY_GOTCHA(mpi_gotcha_t, 1, MPI_Barrier);
    mpi_gotcha_t::configure<2, int, MPI_Comm, int*>("MPI_Comm_rank");
    mpi_gotcha_t::configure<3, int, MPI_Comm, int*>("MPI_Comm_size");
    mpi_gotcha_t::configure<4, int>("MPI_Finalize");
    TIMEMORY_GOTCHA(mpi_gotcha_t, 5, MPI_Bcast);
    TIMEMORY_GOTCHA(mpi_gotcha_t, 6, MPI_Scan);
    TIMEMORY_GOTCHA(mpi_gotcha_t, 7, MPI_Allreduce);
    TIMEMORY_GOTCHA(mpi_gotcha_t, 8, MPI_Reduce);
    TIMEMORY_GOTCHA(mpi_gotcha_t, 9, MPI_Alltoall);
    TIMEMORY_GOTCHA(mpi_gotcha_t, 10, MPI_Allgather);
    TIMEMORY_GOTCHA(mpi_gotcha_t, 11, MPI_Gather);
    TIMEMORY_GOTCHA(mpi_gotcha_t, 12, MPI_Scatter);

    printf("put gotcha is available: %s\n",
           trait::as_string<trait::is_available<put_gotcha_t>>().c_str());
    printf("std gotcha is available: %s\n",
           trait::as_string<trait::is_available<std_gotcha_t>>().c_str());
    printf("mpi gotcha is available: %s\n",
           trait::as_string<trait::is_available<mpi_gotcha_t>>().c_str());
}

//======================================================================================//

int
main(int argc, char** argv)
{    
    init();

    settings::width()        = 12;
    settings::precision()    = 6;
    settings::timing_units() = "msec";
    settings::memory_units() = "kB";
    settings::verbose()      = 1;
    tim::timemory_init(argc, argv);  // parses environment, sets output paths

    mpi::initialize(argc, argv);
    MPI_Barrier(MPI_COMM_WORLD);
    mpi::barrier();

    int size = 1;
    int rank = 0;

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    size = std::max<int>(size, mpi::size());
    rank = std::max<int>(rank, mpi::rank());

    auto rank_size = (rank + 1) * (size + 1);

    MPI_Barrier(MPI_COMM_WORLD);
    mpi::barrier();

    printf("mpi size = %i\n", (int) size);
    printf("mpi rank = %i\n", (int) rank);
    printf("mpi ((rank + 1) * (size + 1)) = %i\n", (int) rank_size);

    int nitr = 5;
    if(argc > 1) nitr = atoi(argv[1]);

    auto _pair = std::pair<float, double>(0., 0.);
    auto _cos  = ext::do_cos_work(nitr, std::ref(_pair));
    printf("[iterations=%i]>      single-precision cos = %f\n", nitr, std::get<0>(_cos));
    printf("[iterations=%i]>      double-precision cos = %f\n", nitr, std::get<1>(_cos));

    auto _R = ext::do_cos_work_ref(nitr, _pair);
    printf("[iterations=%i]> (R)  single-precision cos = %f\n", nitr, std::get<0>(_R));
    printf("[iterations=%i]> (R)  double-precision cos = %f\n", nitr, std::get<1>(_R));

    auto _RK = ext::do_cos_work_cref(nitr, _pair);
    printf("[iterations=%i]> (RK) single-precision cos = %f\n", nitr, std::get<0>(_RK));
    printf("[iterations=%i]> (RK) double-precision cos = %f\n", nitr, std::get<1>(_RK));

    auto _exp = ext::do_exp_work(nitr);
    printf("[iterations=%i]>      single-precision exp = %f\n", nitr, std::get<0>(_exp));
    printf("[iterations=%i]>      double-precision exp = %f\n", nitr, std::get<1>(_exp));

}

//======================================================================================//

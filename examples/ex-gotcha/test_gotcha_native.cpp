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

#include <timemory/timemory.hpp>

#include "gotcha/gotcha.h"
#include "gotcha/gotcha_types.h"

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
namespace settings {
using namespace tim::settings;
}
namespace mpi {
using namespace tim::mpi;
}

//--------------------------------------------------------------------------------------//

using binding_t = struct gotcha_binding_t;
using wrappee_t = gotcha_wrappee_handle_t;
template <size_t _Nt>
using binding_array_t = std::array<binding_t, _Nt>;
template <size_t _Nt>
using wrappee_array_t = std::array<wrappee_t, _Nt>;
template <size_t _Nt>
using wrapids_array_t = std::array<std::string, _Nt>;

//--------------------------------------------------------------------------------------//

template <size_t _Nt>
static size_t&
get_size()
{
    static size_t _instance;
    return _instance;
}

//--------------------------------------------------------------------------------------//

template <size_t _Nt>
static binding_array_t<_Nt>&
get_bindings()
{
    static binding_array_t<_Nt> _instance;
    return _instance;
}

//--------------------------------------------------------------------------------------//

template <size_t _Nt>
static wrappee_array_t<_Nt>&
get_wrappees()
{
    static wrappee_array_t<_Nt> _instance;
    return _instance;
}

//--------------------------------------------------------------------------------------//

template <size_t _Nt>
static wrapids_array_t<_Nt>&
get_wrappids()
{
    static wrapids_array_t<_Nt> _instance;
    return _instance;
}

//--------------------------------------------------------------------------------------//

template <size_t _N, size_t _Nt, typename _Ret, typename... _Args>
static _Ret
wrapper(_Args&&... _args)
{
    auto _orig = (_Ret(*)(_Args && ...)) gotcha_get_wrappee(get_wrappees<_Nt>().at(_N));
    auto _id   = get_wrappids<_Nt>().at(_N);
    printf("[%s]> wrapped %s...\n", __FUNCTION__, _id.c_str());
    TIMEMORY_BLANK_AUTO_TIMER(_id);
    return _orig(std::forward<_Args>(_args)...);
}

//--------------------------------------------------------------------------------------//

template <size_t _N, size_t _Nt, typename _Ret, typename... _Args>
static void
add_gotcha(const std::string& fname)
{
    auto& _wrappees = get_wrappees<_Nt>();
    auto& _bindings = get_bindings<_Nt>();
    auto& _wrappids = get_wrappids<_Nt>();

    get_size<_Nt>()++;
    _wrappids[_N] = fname;
    gotcha_binding_t _binding{ fname.c_str(), (void*) wrapper<_N, _Nt, _Ret, _Args...>, &_wrappees.at(_N) };
    _bindings[_N] = std::move(_binding);
}

//--------------------------------------------------------------------------------------//
/*
static gotcha_wrappee_handle_t orig_puts_handle;

//--------------------------------------------------------------------------------------//

struct gotcha_binding_t iofuncs[] = {
    { "puts", (void*) wrapper<0, int, const char*>, &orig_puts_handle },
};
*/

//--------------------------------------------------------------------------------------//

int
init()
{
    enum gotcha_error_t result;
    gotcha_set_priority("timemory", 0);

    constexpr size_t _Nt = 18;
    add_gotcha<0, _Nt, int>("MPI_Finalize");
    add_gotcha<1, _Nt, int, const char*>("puts");
    add_gotcha<2, _Nt, int, MPI_Comm>("MPI_Barrier");
    add_gotcha<3, _Nt, int, int*, char***>("MPI_Init");
    add_gotcha<4, _Nt, int, MPI_Comm, int*>("MPI_Comm_rank");
    add_gotcha<5, _Nt, int, MPI_Comm, int*>("MPI_Comm_size");
    /*
    add_gotcha<6, _Nt, int, int*, char***, int, int*>("MPI_Init_thread");
    add_gotcha<7, _Nt, int, void*, int, MPI_Datatype, int, MPI_Comm>("MPI_Bcast");
    add_gotcha<8, _Nt, int, void*, void*, int, MPI_Datatype, MPI_Op, MPI_Comm>(
        "MPI_Scan");
    add_gotcha<9, _Nt, int, void*, void*, int, MPI_Datatype, MPI_Op, MPI_Comm>(
        "MPI_Allreduce");
    add_gotcha<10, _Nt, int, void*, void*, int, MPI_Datatype, MPI_Op, int, MPI_Comm>(
        "MPI_Reduce");
    add_gotcha<11, _Nt, int, void*, int, MPI_Datatype, void*, int, MPI_Datatype,
               MPI_Comm>("MPI_Alltoall");
    add_gotcha<12, _Nt, int, void*, int, MPI_Datatype, void*, int, MPI_Datatype,
               MPI_Comm>("MPI_Allgather");
    add_gotcha<13, _Nt, int, void*, int, MPI_Datatype, void*, int, MPI_Datatype, int,
               MPI_Comm>("MPI_Gather");
    add_gotcha<14, _Nt, int, void*, int, MPI_Datatype, void*, int, MPI_Datatype, int,
               MPI_Comm>("MPI_Scatter");
    add_gotcha<15, _Nt, int, void*, int, MPI_Datatype, void*, int*, int*, MPI_Datatype,
               MPI_Comm>("MPI_Allgatherv");
    add_gotcha<16, _Nt, int, void*, int, MPI_Datatype, void*, int*, int*, MPI_Datatype,
               int, MPI_Comm>("MPI_Gatherv");
    add_gotcha<17, _Nt, int, void*, int*, int*, MPI_Datatype, void*, int, MPI_Datatype,
               int, MPI_Comm>("MPI_Scatterv");
               */
    auto& _bindings = get_bindings<_Nt>();
    if(settings::verbose() > 0 || settings::debug())
    {
        for(size_t i = 0; i < get_size<_Nt>(); ++i)
        {
            std::cout << "Wrapped: " << get_wrappids<_Nt>()[i]
                      << ", wrapped pointer: " << _bindings.at(i).wrapper_pointer
                      << ", function_handle: " << _bindings.at(i).function_handle << ", name: " << _bindings.at(i).name
                      << std::endl;
        }
    }

    result = gotcha_wrap(_bindings.data(), get_size<_Nt>(), "timemory");

    if(result != GOTCHA_SUCCESS)
    {
        // fprintf(stderr, "gotcha_wrap returned %d\n", (int) result);
        return result;
    }
    return 0;
}

//--------------------------------------------------------------------------------------//

long
fibonacci(long n)
{
    return (n < 2) ? n : (fibonacci(n - 1) + fibonacci(n - 2));
}

//======================================================================================//

int
main(int argc, char** argv)
{
    settings::verbose() = 1;
    tim::timemory_init(argc, argv);  // parses environment, sets output paths

    int rc = init();
    if(rc != 0) exit(EXIT_FAILURE);

    mpi::initialize(argc, argv);
    const int n = 10;
    puts("Test");
    auto ret = fibonacci(n);
    printf("fibonacci(%i) = %li\n", n, ret);
}

//======================================================================================//

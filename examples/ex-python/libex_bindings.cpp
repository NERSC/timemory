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

#include "pybind11/cast.h"
#include "pybind11/chrono.h"
#include "pybind11/eval.h"
#include "pybind11/functional.h"
#include "pybind11/iostream.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "pybind11/stl.h"

#include <random>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#if defined(USE_MPI) || defined(TIMEMORY_USE_MPI)
#    include <mpi.h>
#endif

#if defined(_OPENMP)
#    include <omp.h>
#endif

namespace py = pybind11;

template <typename Tp>
using vector_t = std::vector<Tp>;

static std::mt19937 rng;

//--------------------------------------------------------------------------------------//

template <typename Tp>
inline vector_t<Tp>
generate(const int64_t& nsize)
{
    std::vector<Tp> sendbuf(nsize, 0.0);
    std::mt19937    rng;
    rng.seed(54561434UL);
    auto dist = [&]() { return std::generate_canonical<Tp, 10>(rng); };
    std::generate(sendbuf.begin(), sendbuf.end(), [&]() { return dist(); });
    return sendbuf;
}

//--------------------------------------------------------------------------------------//

template <typename Tp>
inline vector_t<Tp>
allreduce(const vector_t<Tp>& sendbuf)
{
    vector_t<Tp> recvbuf(sendbuf.size(), 0.0);
#if defined(USE_MPI) || defined(TIMEMORY_USE_MPI)
    auto dtype = (std::is_same<Tp, float>::value) ? MPI_FLOAT : MPI_DOUBLE;
    MPI_Allreduce(sendbuf.data(), recvbuf.data(), sendbuf.size(), dtype, MPI_SUM,
                  MPI_COMM_WORLD);
#else
    throw std::runtime_error("No MPI support!");
#endif
    return recvbuf;
}

//--------------------------------------------------------------------------------------//

double
run(int nitr, int nsize)
{
    rng.seed(54561434UL);

    printf("[%s] Running MPI algorithm with %i iterations and %i entries...\n", __func__,
           nitr, nsize);

    double dsum = 0.0;
#pragma omp parallel for
    for(int i = 0; i < nitr; ++i)
    {
        auto dsendbuf = generate<double>(nsize);
        auto drecvbuf = allreduce(dsendbuf);
        auto dtmp     = std::accumulate(drecvbuf.begin(), drecvbuf.end(), 0.0);
#pragma omp atomic
        dsum += dtmp;
    }
    return dsum;
}

//--------------------------------------------------------------------------------------//

PYBIND11_MODULE(libex_python_bindings, ex)
{
    auto _run = [](int nitr, int nsize) {
        try
        {
            py::gil_scoped_release release;
#if defined(_OPENMP)
            int nrank = 1;
            MPI_Comm_size(MPI_COMM_WORLD, &nrank);
            omp_set_num_threads(std::thread::hardware_concurrency() / nrank);
#endif
            return run(nitr, nsize);
        } catch(std::exception& e)
        {
            fprintf(stderr, "Error! %s\n", e.what());
            throw;
        }
        return 0.0;
    };

    ex.def("run", _run, "Run a calculation", py::arg("nitr") = 10,
           py::arg("nsize") = 5000);
}

//--------------------------------------------------------------------------------------//

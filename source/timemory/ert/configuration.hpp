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

#pragma once

#include "timemory/backends/cuda.hpp"
#include "timemory/backends/device.hpp"
#include "timemory/bits/settings.hpp"
#include "timemory/components/timing.hpp"
#include "timemory/ert/aligned_allocator.hpp"
#include "timemory/ert/data.hpp"
#include "timemory/ert/kernels.hpp"
#include "timemory/ert/types.hpp"

#include <cstdint>
#include <functional>

// default vectorization width
#if !defined(TIMEMORY_VEC)
#    define TIMEMORY_VEC 256
#endif

namespace tim
{
namespace ert
{
//======================================================================================//

template <typename _Device, typename _Tp, typename _ExecData, typename _Counter>
struct configuration
{
    using this_type       = configuration<_Device, _Tp, _ExecData, _Counter>;
    using ert_data_t      = _ExecData;
    using ert_params_t    = exec_params;
    using device_t        = _Device;
    using counter_t       = _Counter;
    using ert_counter_t   = counter<device_t, _Tp, ert_data_t, counter_t>;
    using ert_data_ptr_t  = std::shared_ptr<ert_data_t>;
    using executor_func_t = std::function<ert_counter_t(ert_data_ptr_t)>;
    using get_uint64_t    = std::function<uint64_t()>;

    //----------------------------------------------------------------------------------//

    static get_uint64_t& get_num_threads()
    {
        static get_uint64_t _instance = []() {
            if(settings::ert_num_threads() > 0)
                return settings::ert_num_threads();
            // for checking if gpu
            static constexpr bool is_gpu = device::is_gpu<_Device>::value;
            return (is_gpu) ? settings::ert_num_threads_gpu()
                            : settings::ert_num_threads_cpu();
        };
        return _instance;
    }

    //----------------------------------------------------------------------------------//

    static get_uint64_t& get_num_streams()
    {
        static get_uint64_t _instance = []() { return settings::ert_num_streams(); };
        return _instance;
    }

    //----------------------------------------------------------------------------------//

    static get_uint64_t& get_grid_size()
    {
        static get_uint64_t _instance = []() { return settings::ert_grid_size(); };
        return _instance;
    }

    //----------------------------------------------------------------------------------//

    static get_uint64_t& get_block_size()
    {
        static get_uint64_t _instance = []() { return settings::ert_block_size(); };
        return _instance;
    }

    //----------------------------------------------------------------------------------//

    static get_uint64_t& get_alignment()
    {
        static get_uint64_t _instance = []() {
            return std::max<uint64_t>(settings::ert_alignment(), 8 * sizeof(_Tp));
        };
        return _instance;
    }

    //----------------------------------------------------------------------------------//

    static get_uint64_t& get_min_working_size()
    {
        static get_uint64_t _instance = []() {
            if(settings::ert_min_working_size() > 0)
                return settings::ert_min_working_size();
            static constexpr bool is_gpu = device::is_gpu<_Device>::value;
            return (is_gpu) ? settings::ert_min_working_size_gpu()
                            : settings::ert_min_working_size_cpu();
        };
        return _instance;
    }

    //----------------------------------------------------------------------------------//

    static get_uint64_t& get_max_data_size()
    {
        static get_uint64_t _instance = []() -> uint64_t {
            if(settings::ert_max_data_size() > 0)
                return settings::ert_max_data_size();
            static constexpr bool is_gpu = device::is_gpu<_Device>::value;
            if(is_gpu)
                return settings::ert_max_data_size_gpu();
            else
                return 2 * ert::cache_size::get_max();
        };
        return _instance;
    }

    //----------------------------------------------------------------------------------//
    /// configure the number of threads, number of streams, block size, grid size, and
    /// alignment
    template <typename Dev                                            = _Device,
              enable_if_t<std::is_same<Dev, device::cpu>::value, int> = 0>
    static void configure(uint64_t nthreads, uint64_t alignment = sizeof(_Tp),
                          uint64_t nstreams = 0, uint64_t block_size = 0,
                          uint64_t grid_size = 0)
    {
        get_num_threads() = [=]() -> uint64_t { return nthreads; };
        get_num_streams() = [=]() -> uint64_t { return nstreams; };
        get_grid_size()   = [=]() -> uint64_t { return grid_size; };
        get_block_size()  = [=]() -> uint64_t { return block_size; };
        get_alignment()   = [=]() -> uint64_t { return alignment; };
    }

    //----------------------------------------------------------------------------------//
    /// configure the number of threads, number of streams, block size, grid size, and
    /// alignment
    template <typename Dev                                            = _Device,
              enable_if_t<std::is_same<Dev, device::gpu>::value, int> = 0>
    static void configure(uint64_t nthreads, uint64_t alignment = sizeof(_Tp),
                          uint64_t nstreams = 1, uint64_t block_size = 1024,
                          uint64_t grid_size = 0)
    {
        get_num_threads() = [=]() -> uint64_t { return nthreads; };
        get_num_streams() = [=]() -> uint64_t { return nstreams; };
        get_grid_size()   = [=]() -> uint64_t { return grid_size; };
        get_block_size()  = [=]() -> uint64_t { return block_size; };
        get_alignment()   = [=]() -> uint64_t { return alignment; };
    }

    //----------------------------------------------------------------------------------//

    static executor_func_t& get_executor()
    {
        static executor_func_t _instance = [](ert_data_ptr_t data) {
            using lli = long long int;
            // configuration sizes
            auto _mws_size   = get_min_working_size()();
            auto _max_size   = get_max_data_size()();
            auto _num_thread = get_num_threads()();
            auto _num_stream = get_num_streams()();
            auto _grid_size  = get_grid_size()();
            auto _block_size = get_block_size()();
            auto _align_size = get_alignment()();

            // execution parameters
            ert_params_t params(_mws_size, _max_size, _num_thread, _num_stream,
                                _grid_size, _block_size);
            // operation _counter instance
            ert_counter_t _counter(params, data, _align_size);

            // set bytes per element
            _counter.bytes_per_element = sizeof(_Tp);
            // set number of memory accesses per element from two functions
            _counter.memory_accesses_per_element = 2;

            auto dtype = demangle(typeid(_Tp).name());

            printf(
                "\n[ert::executor]> "
                "working-set = %lli, max-size = %lli, num-thread = %lli, num-stream = "
                "%lli, grid-size = %lli, block-size = %lli, align-size = %lli, data-type "
                "= %s\n",
                (lli) _mws_size, (lli) _max_size, (lli) _num_thread, (lli) _num_stream,
                (lli) _grid_size, (lli) _block_size, (lli) _align_size, dtype.c_str());

            return _counter;
        };
        return _instance;
    }

public:
    get_uint64_t    num_threads      = this_type::get_num_threads();
    get_uint64_t    num_streams      = this_type::get_num_streams();
    get_uint64_t    min_working_size = this_type::get_min_working_size();
    get_uint64_t    max_data_size    = this_type::get_max_data_size();
    get_uint64_t    alignment        = this_type::get_alignment();
    get_uint64_t    grid_size        = this_type::get_grid_size();
    get_uint64_t    block_size       = this_type::get_block_size();
    executor_func_t executor         = this_type::get_executor();
};

//======================================================================================//

template <typename _Device, typename _Tp, typename _ExecData, typename _Counter>
struct executor
{
    static_assert(!std::is_same<_Device, device::gpu>::value,
                  "Error! Device should not be gpu");

    //----------------------------------------------------------------------------------//
    // useful aliases
    //
    using configuration_type = configuration<_Device, _Tp, _ExecData, _Counter>;
    using counter_type       = counter<_Device, _Tp, _ExecData, _Counter>;
    using this_type          = executor<_Device, _Tp, _ExecData, _Counter>;
    using callback_type      = std::function<void(counter_type&)>;

public:
    //----------------------------------------------------------------------------------//
    //  standard invocation with no callback specialization
    //
    executor(configuration_type& config, std::shared_ptr<_ExecData> _data)
    {
        try
        {
            auto _counter = config.executor(_data);
            callback(_counter);
        } catch(std::exception& e)
        {
            std::cerr << "\n\nEXCEPTION:\n";
            std::cerr << "\t" << e.what() << "\n\n" << std::endl;
        }
    }

    //----------------------------------------------------------------------------------//
    //  specialize the counter callback
    //
    template <typename _Func>
    executor(configuration_type& config, std::shared_ptr<_ExecData> _data,
             _Func&& _counter_callback)
    {
        try
        {
            auto _counter = config.executor(_data);
            _counter.set_callback(std::forward<_Func>(_counter_callback));
            callback(_counter);
        } catch(std::exception& e)
        {
            std::cerr << "\n\nEXCEPTION:\n";
            std::cerr << "\t" << e.what() << "\n\n" << std::endl;
        }
    }

public:
    //----------------------------------------------------------------------------------//
    //
    callback_type callback = get_callback();

public:
    //----------------------------------------------------------------------------------//
    //
    static callback_type& get_callback()
    {
        static callback_type _instance = [](counter_type& _counter) {
            this_type::execute(_counter);
        };
        return _instance;
    }

    //----------------------------------------------------------------------------------//
    //
    static void execute(counter_type& _counter)
    {
        // vectorization number of ops
        static constexpr const int SIZE_BITS = sizeof(_Tp) * 8;
        static_assert(SIZE_BITS > 0, "Calculated bits size is not greater than zero");
        static constexpr const int VEC = TIMEMORY_VEC / SIZE_BITS;
        static_assert(VEC > 0, "Calculated vector size is zero");

        // functions
        auto store_func = [](_Tp& a, const _Tp& b) { a = b; };
        // auto mult_func  = [](_Tp& a, const _Tp& b, const _Tp& c) { a = b * c; };
        auto add_func = [](_Tp& a, const _Tp& b, const _Tp& c) { a = b + c; };
        auto fma_func = [](_Tp& a, const _Tp& b, const _Tp& c) { a = a * b + c; };

        // set bytes per element
        _counter.bytes_per_element = sizeof(_Tp);
        // set number of memory accesses per element from two functions
        _counter.memory_accesses_per_element = 2;

        // set the label
        _counter.label = "scalar_add";
        // run the kernels
        ops_main<1>(_counter, add_func, store_func);

        // set the label
        // _counter.label = "vector_mult";
        // run the kernels
        // ops_main<VEC / 2, VEC, 2 * VEC, 4 * VEC>(_counter, mult_func, store_func);

        // set the label
        _counter.label = "vector_fma";
        // run the kernels
        ops_main<VEC / 2, VEC, 2 * VEC, 4 * VEC>(_counter, fma_func, store_func);
    }
};

//======================================================================================//

template <typename _Tp, typename _ExecData, typename _Counter>
struct executor<device::gpu, _Tp, _ExecData, _Counter>
{
    using _Device = device::gpu;
    static_assert(std::is_same<_Device, device::gpu>::value,
                  "Error! Device should be gpu");

    //----------------------------------------------------------------------------------//
    // useful aliases
    //
    using configuration_type = configuration<_Device, _Tp, _ExecData, _Counter>;
    using counter_type       = counter<_Device, _Tp, _ExecData, _Counter>;
    using this_type          = executor<_Device, _Tp, _ExecData, _Counter>;
    using callback_type      = std::function<void(counter_type&)>;

public:
    //----------------------------------------------------------------------------------//
    //  standard invocation with no callback specialization
    //
    executor(configuration_type& config, std::shared_ptr<_ExecData> _data)
    {
        try
        {
            auto _counter = config.executor(_data);
            callback(_counter);
        } catch(std::exception& e)
        {
            std::cerr << "\n\nEXCEPTION:\n";
            std::cerr << "\t" << e.what() << "\n\n" << std::endl;
        }
    }

    //----------------------------------------------------------------------------------//
    //  specialize the counter callback
    //
    template <typename _Func>
    executor(configuration_type& config, std::shared_ptr<_ExecData> _data,
             _Func&& _counter_callback)
    {
        try
        {
            auto _counter = config.executor(_data);
            _counter.set_callback(std::forward<_Func>(_counter_callback));
            callback(_counter);
        } catch(std::exception& e)
        {
            std::cerr << "\n\nEXCEPTION:\n";
            std::cerr << "\t" << e.what() << "\n\n" << std::endl;
        }
    }

public:
    //----------------------------------------------------------------------------------//
    //
    callback_type callback = get_callback();

public:
    //----------------------------------------------------------------------------------//
    //
    static callback_type& get_callback()
    {
        static callback_type _instance = [](counter_type& _counter) {
            this_type::execute(_counter);
        };
        return _instance;
    }

    //----------------------------------------------------------------------------------//
    // The enclosing parent function for an extended __host__ __device__
    // lambda must allow its address to be taken
    static void execute(counter_type& _counter)
    {
        // functions
        auto store_func = [] TIMEMORY_DEVICE_LAMBDA(_Tp & a, const _Tp& b) { a = b; };
        auto add_func   = [] TIMEMORY_DEVICE_LAMBDA(_Tp & a, const _Tp& b, const _Tp& c) {
            a = b + c;
        };
        // auto mult_func = [] TIMEMORY_LAMBDA(_Tp & a, const _Tp& b, const _Tp& c) {
        //    a = b * c;
        //};
        auto fma_func = [] TIMEMORY_DEVICE_LAMBDA(_Tp & a, const _Tp& b, const _Tp& c) {
            a = a * b + c;
        };

        // set bytes per element
        _counter.bytes_per_element = sizeof(_Tp);
        // set number of memory accesses per element from two functions
        _counter.memory_accesses_per_element = 2;

        // set the label
        _counter.label = "scalar_add";
        // run the kernels
        ops_main<1>(_counter, add_func, store_func);

        // set the label
        // _counter.label = "vector_mult";
        // run the kernels
        // ops_main<4, 16, 64, 128, 256, 512>(_counter, mult_func, store_func);

        // set the label
        _counter.label = "vector_fma";
        // run the kernels
        ops_main<4, 16, 64, 128, 256, 512>(_counter, fma_func, store_func);
    }
};

//======================================================================================//
/// for variadic expansion to set the callback
///
template <typename _Executor>
struct callback
{
    template <typename _Func>
    callback(_Func&& f)
    {
        _Executor::get_callback() = f;
    }

    template <typename _Func>
    callback(_Executor& _exec, _Func&& f)
    {
        _exec.callback = f;
    }
};

//======================================================================================//

}  // namespace ert

}  // namespace tim

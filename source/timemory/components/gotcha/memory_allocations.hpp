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

#pragma once

#include "timemory/api.hpp"
#include "timemory/components/base.hpp"
#include "timemory/components/gotcha/components.hpp"
#include "timemory/components/gotcha/types.hpp"
#include "timemory/mpl/concepts.hpp"
#include "timemory/mpl/policy.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/operations/types/construct.hpp"
#include "timemory/types.hpp"
#include "timemory/variadic/component_tuple.hpp"
#include "timemory/variadic/types.hpp"

#include <memory>
#include <string>

#if defined(__GNUC__) && (__GNUC__ >= 6)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

namespace tim
{
namespace component
{
//
struct malloc_gotcha
: base<malloc_gotcha, double>
, public concepts::external_function_wrapper
{
#if defined(TIMEMORY_USE_CUDA)
    static constexpr size_t data_size = 9;
#else
    static constexpr size_t data_size = 3;
#endif

    using value_type   = double;
    using this_type    = malloc_gotcha;
    using base_type    = base<this_type, value_type>;
    using storage_type = typename base_type::storage_type;
    using string_hash  = std::hash<std::string>;

    // required static functions
    static std::string label() { return "malloc_gotcha"; }
    static std::string description()
    {
#if defined(TIMEMORY_USE_CUDA)
        return "GOTCHA wrapper for memory allocation functions: malloc, calloc, free, "
               "cudaMalloc, cudaMallocHost, cudaMallocManaged, cudaHostAlloc, cudaFree, "
               "cudaFreeHost";
#else
        return "GOTCHA wrapper for memory allocation functions: malloc, calloc, free";
#endif
    }

    using base_type::accum;
    using base_type::set_started;
    using base_type::set_stopped;
    using base_type::value;

    template <typename Tp>
    using gotcha_component_type = push_back_t<Tp, this_type>;

    template <typename Tp>
    using gotcha_type =
        gotcha<data_size, push_back_t<Tp, this_type>, type_list<this_type>>;

    template <typename Tp>
    using component_type = push_back_t<Tp, gotcha_type<Tp>>;

    static void global_finalize()
    {
        for(auto& itr : get_cleanup_list())
            itr();
        get_cleanup_list().clear();
    }

public:
    template <typename Tp>
    static void configure();

    template <typename Tp>
    static void tear_down();

public:
    TIMEMORY_DEFAULT_OBJECT(malloc_gotcha)

public:
    void start() { value = 0; }

    void stop()
    {
        // value should be updated via audit in-between start() and stop()
        accum += value;
    }

    TIMEMORY_NODISCARD double get() const { return accum / base_type::get_unit(); }

    TIMEMORY_NODISCARD double get_display() const { return get(); }

    void set_prefix();

    /// nbytes is passed to malloc
    void audit(audit::incoming, size_t nbytes)
    {
        DEBUG_PRINT_HERE("%s(%i)", m_prefix, (int) nbytes);
        // malloc
        value = (nbytes);
        DEBUG_PRINT_HERE("value: %12.8f, accum: %12.8f", value, accum);
    }

    /// nmemb and size is passed to calloc
    void audit(audit::incoming, size_t nmemb, size_t size)
    {
        DEBUG_PRINT_HERE("%s(%i, %i)", m_prefix, (int) nmemb, (int) size);
        // calloc
        value = (nmemb * size);
        DEBUG_PRINT_HERE("value: %12.8f, accum: %12.8f", value, accum);
    }

    /// void* is returned from malloc and calloc
    void audit(audit::outgoing, void* ptr)
    {
        DEBUG_PRINT_HERE("%s(%p)", m_prefix, ptr);
        if(ptr)
        {
            get_allocation_map()[ptr] = value;
            DEBUG_PRINT_HERE("value: %12.8f, accum: %12.8f", value, accum);
        }
    }

    /// void* is passed to free
    void audit(audit::incoming, void* ptr)
    {
        DEBUG_PRINT_HERE("%s(%p)", m_prefix, ptr);
        auto itr = get_allocation_map().find(ptr);
        if(itr != get_allocation_map().end())
        {
            value = itr->second;
            DEBUG_PRINT_HERE("value: %12.8f, accum: %12.8f", value, accum);
            get_allocation_map().erase(itr);
        }
        else
        {
            if(settings::verbose() > 1 || settings::debug())
            {
                printf("[%s]> free of unknown pointer size: %p\n",
                       this_type::get_label().c_str(), ptr);
            }
        }
    }

    //----------------------------------------------------------------------------------//

#if defined(TIMEMORY_USE_CUDA)

    //----------------------------------------------------------------------------------//
    // cudaMalloc, cudaMallocHost
    void audit(audit::incoming, void** devPtr, size_t size)
    {
        DEBUG_PRINT_HERE("%s(void**, %lu)", m_prefix, (unsigned long) size);
        // malloc
        value       = (size);
        m_last_addr = devPtr;
        DEBUG_PRINT_HERE("value: %12.8f, accum: %12.8f", value, accum);
    }

    //----------------------------------------------------------------------------------//
    // cudaHostAlloc / cudaMallocManaged
    void audit(audit::incoming, void** hostPtr, size_t size, unsigned int flags)
    {
        DEBUG_PRINT_HERE("%s(void**, %lu)", m_prefix, (unsigned long) size);
        value       = (size);
        m_last_addr = hostPtr;
        DEBUG_PRINT_HERE("value: %12.8f, accum: %12.8f", value, accum);
        consume_parameters(flags);
    }

    //----------------------------------------------------------------------------------//
    // cudaMalloc and cudaHostAlloc
    void audit(audit::outgoing, cuda::error_t err)
    {
        if(m_last_addr)
        {
            void* ptr                 = (void*) ((char**) (m_last_addr)[0]);
            get_allocation_map()[ptr] = value;
            if(err != cuda::success_v && (settings::debug() || settings::verbose() > 1))
            {
                PRINT_HERE("%s did not return cudaSuccess, values may be corrupted",
                           m_prefix);
            }
        }
    }

#endif

    //----------------------------------------------------------------------------------//

    void set_prefix(const char* _prefix) { m_prefix = _prefix; }

    //----------------------------------------------------------------------------------//

    this_type& operator+=(const this_type& rhs)
    {
        value += rhs.value;
        accum += rhs.accum;
        return *this;
    }

    //----------------------------------------------------------------------------------//

    this_type& operator-=(const this_type& rhs)
    {
        value -= rhs.value;
        accum -= rhs.accum;
        return *this;
    }

private:
    using alloc_map_t  = std::unordered_map<void*, size_t>;
    using clean_list_t = std::vector<std::function<void()>>;

    static clean_list_t& get_cleanup_list()
    {
        static clean_list_t _instance{};
        return _instance;
    }

    static alloc_map_t& get_allocation_map()
    {
        static thread_local alloc_map_t _instance{};
        return _instance;
    }

private:
    const char* m_prefix = nullptr;
#if defined(TIMEMORY_USE_CUDA)
    void** m_last_addr = nullptr;
#endif
};
//
//--------------------------------------------------------------------------------------//
//
#if defined(TIMEMORY_USE_GOTCHA)
//
template <typename Tp>
inline void
malloc_gotcha::configure()
{
    // static_assert(!std::is_same<Type, malloc_gotcha>::value,
    //              "Error! Cannot configure with self as the type!");

    using tuple_t           = push_back_t<Tp, this_type>;
    using local_gotcha_type = gotcha<data_size, tuple_t, type_list<this_type>>;

    local_gotcha_type::get_default_ready() = false;
    local_gotcha_type::get_initializer()   = []() {
        local_gotcha_type::template configure<0, void*, size_t>("malloc");
        local_gotcha_type::template configure<1, void*, size_t, size_t>("calloc");
        local_gotcha_type::template configure<2, void, void*>("free");
        // TIMEMORY_C_GOTCHA(local_gotcha_type, 0, malloc);
        // TIMEMORY_C_GOTCHA(local_gotcha_type, 1, calloc);
        // TIMEMORY_C_GOTCHA(local_gotcha_type, 2, free);
#    if defined(TIMEMORY_USE_CUDA)
        local_gotcha_type::template configure<3, cudaError_t, void**, size_t>(
            "cudaMalloc");
        local_gotcha_type::template configure<4, cudaError_t, void**, size_t>(
            "cudaMallocHost");
        local_gotcha_type::template configure<5, cudaError_t, void**, size_t,
                                              unsigned int>("cudaMallocManaged");
        local_gotcha_type::template configure<6, cudaError_t, void**, size_t,
                                              unsigned int>("cudaHostAlloc");
        local_gotcha_type::template configure<7, cudaError_t, void*>("cudaFree");
        local_gotcha_type::template configure<8, cudaError_t, void*>("cudaFreeHost");
#    endif
    };

    get_cleanup_list().emplace_back([]() { malloc_gotcha::tear_down<Tp>(); });
}
//
template <typename Tp>
inline void
malloc_gotcha::tear_down()
{
    // static_assert(!std::is_same<Type, malloc_gotcha>::value,
    //              "Error! Cannot configure with self as the type!");

    using tuple_t           = push_back_t<Tp, this_type>;
    using local_gotcha_type = gotcha<data_size, tuple_t, type_list<this_type>>;

    local_gotcha_type::get_default_ready() = false;
    local_gotcha_type::get_initializer()   = []() {};
    local_gotcha_type::disable();
}
//
#endif
//
/// \struct tim::component::memory_allocations
/// \brief This component wraps malloc, calloc, free, cudaMalloc, cudaFree via
/// GOTCHA and tracks the number of bytes requested/freed in each call.
/// This component is useful for detecting the locations where memory re-use
/// would provide a performance benefit.
///
struct memory_allocations
: base<memory_allocations, void>
, public concepts::external_function_wrapper
, private policy::instance_tracker<memory_allocations, true>
{
    using value_type   = void;
    using this_type    = memory_allocations;
    using base_type    = base<this_type, value_type>;
    using tracker_type = policy::instance_tracker<memory_allocations, true>;

    using malloc_gotcha_t = typename malloc_gotcha::gotcha_type<component_tuple_t<>>;
    using malloc_bundle_t = component_tuple_t<malloc_gotcha_t>;
    using data_pointer_t  = std::unique_ptr<malloc_bundle_t>;

    static std::string label() { return "memory_allocations"; }
    static std::string description()
    {
        return "Number of bytes allocated/freed instead of peak/current memory usage: "
               "free(malloc(10)) + free(malloc(10)) would use 10 bytes but this would "
               "report 20 bytes";
    }

    static void global_init() { malloc_gotcha::configure<component_tuple_t<>>(); }
    static void global_finalize() { malloc_gotcha::tear_down<component_tuple_t<>>(); }

    void start()
    {
        auto _cnt = tracker_type::start();
        if(_cnt.first == 0 && _cnt.second == 0 && !get_data())
        {
            get_data() = std::make_unique<malloc_bundle_t>();
            get_data()->start();
        }
    }

    void stop()
    {
        auto _cnt = tracker_type::stop();
        if(_cnt.first == 0 && _cnt.second == 0 && get_data())
        {
            get_data()->stop();
            get_data().reset(nullptr);
        }
    }

private:
    static data_pointer_t& get_data()
    {
        static auto _instance = data_pointer_t{};
        return _instance;
    }
};
//
}  // namespace component
}  // namespace tim

#if defined(__GNUC__) && (__GNUC__ >= 6)
#    pragma GCC diagnostic pop
#endif

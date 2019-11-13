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

#include "timemory/bits/settings.hpp"
#include "timemory/components/base.hpp"
#include "timemory/components/gotcha.hpp"
#include "timemory/components/types.hpp"
#include "timemory/mpl/apply.hpp"

#if defined(TIMEMORY_USE_CUDA)
#    include "timemory/backends/cuda.hpp"
#endif

#include <cstdint>
#include <string>
#include <tuple>

namespace tim
{
//
// clang-format off
namespace component { struct malloc_gotcha; }
// clang-format on
//
//======================================================================================//

namespace trait
{
template <>
struct supports_args<component::malloc_gotcha, std::tuple<std::string, size_t>>
: std::true_type
{};

template <>
struct supports_args<component::malloc_gotcha, std::tuple<std::string, size_t, size_t>>
: std::true_type
{};

template <>
struct supports_args<component::malloc_gotcha, std::tuple<std::string, void*>>
: std::true_type
{};

#if defined(TIMEMORY_USE_CUDA)
template <>
struct supports_args<component::malloc_gotcha, std::tuple<std::string, void**, size_t>>
: std::true_type
{};

template <>
struct supports_args<component::malloc_gotcha, std::tuple<std::string, cuda::error_t>>
: std::true_type
{};
#endif

template <>
struct uses_memory_units<component::malloc_gotcha> : std::true_type
{};

template <>
struct is_memory_category<component::malloc_gotcha> : std::true_type
{};

template <>
struct requires_prefix<component::malloc_gotcha> : std::true_type
{};

}  // namespace trait

namespace component
{
struct malloc_gotcha
: base<malloc_gotcha, double, policy::global_init, policy::global_finalize>
{
#if defined(TIMEMORY_USE_CUDA)
    static constexpr uintmax_t data_size = 5;
    static constexpr uintmax_t num_alloc = 3;
#else
    static constexpr uintmax_t data_size = 3;
    static constexpr uintmax_t num_alloc = 2;
#endif

    // clang-format off
    using value_type   = double;
    using this_type    = malloc_gotcha;
    using base_type    = base<this_type, value_type, policy::global_init, policy::global_finalize>;
    using storage_type = typename base_type::storage_type;
    using string_hash  = std::hash<std::string>;
    // clang-format on

    // formatting
    static const short precision = 3;
    static const short width     = 12;

    // required static functions
    static std::string label() { return "malloc_gotcha"; }
    static std::string description() { return "GOTCHA wrapper for memory allocation"; }
    static std::string display_unit() { return "MB"; }
    static int64_t     unit() { return units::megabyte; }
    static value_type  record() { return value_type{ 0.0 }; }

    using base_type::accum;
    using base_type::is_transient;
    using base_type::set_started;
    using base_type::set_stopped;
    using base_type::value;

public:
    template <typename... _Types>
    struct gotcha_spec;

    template <typename... _Types, template <typename...> class _Tuple>
    struct gotcha_spec<_Tuple<_Types...>>
    {
        using gotcha_component_type = _Tuple<_Types..., this_type>;
        using gotcha_type           = gotcha<data_size, gotcha_component_type, this_type>;
        using component_type        = _Tuple<_Types..., gotcha_type>;

        static std::function<void()>& get_initializer()
        {
            static std::function<void()> _lambda = []() {
                /*
#if defined(TIMEMORY_USE_CUDA)
                TIMEMORY_C_GOTCHA(gotcha_type, 0, malloc);
                TIMEMORY_C_GOTCHA(gotcha_type, 1, calloc);
                TIMEMORY_C_GOTCHA(gotcha_type, 2, cudaMalloc);
                TIMEMORY_C_GOTCHA(gotcha_type, 3, free);
                TIMEMORY_C_GOTCHA(gotcha_type, 4, cudaFree);
#else
                TIMEMORY_C_GOTCHA(gotcha_type, 0, malloc);
                TIMEMORY_C_GOTCHA(gotcha_type, 1, calloc);
                TIMEMORY_C_GOTCHA(gotcha_type, 2, free);
#endif
                */
            };
            return _lambda;
        }
    };

    template <typename... _LhsTypes, typename... _RhsTypes,
              template <typename...> class _Lhs, template <typename...> class _Rhs,
              template <typename, typename> class _Hybrid>
    struct gotcha_spec<_Hybrid<_Lhs<_LhsTypes...>, _Rhs<_RhsTypes...>>>
    {
        using gotcha_component_type = _Hybrid<_Lhs<_LhsTypes...>, _Rhs<_RhsTypes...>>;
        using gotcha_type           = gotcha<data_size, gotcha_component_type, this_type>;
        using component_type =
            _Hybrid<_Lhs<_LhsTypes..., gotcha_type>, _Rhs<_RhsTypes...>>;

        static std::function<void()>& get_initializer()
        {
            static std::function<void()> _lambda = []() {};
            return _lambda;
        }
    };

    //----------------------------------------------------------------------------------//

    static void invoke_global_init(storage_type*) {}

    //----------------------------------------------------------------------------------//

    static void invoke_global_finalize(storage_type*) {}

    //----------------------------------------------------------------------------------//

    static uintmax_t get_index(uintmax_t _hash)
    {
        uintmax_t idx = std::numeric_limits<uintmax_t>::max();
        for(uintmax_t i = 0; i < get_hash_array().size(); ++i)
        {
            if(_hash == get_hash_array()[i])
                idx = i;
        }
        return idx;
    }

public:
    //----------------------------------------------------------------------------------//

    malloc_gotcha(const std::string& _prefix = "")
    : prefix_hash(string_hash()(_prefix))
    , prefix_idx(get_index(prefix_hash))
    , prefix(_prefix)
    {
        value = 0.0;
        accum = 0.0;
    }

    ~malloc_gotcha()                = default;
    malloc_gotcha(const this_type&) = default;
    malloc_gotcha(this_type&&)      = default;
    malloc_gotcha& operator=(const this_type&) = default;
    malloc_gotcha& operator=(this_type&&) = default;

public:
    //----------------------------------------------------------------------------------//

    void start()
    {
        set_started();
        value = record();
    }

    void stop()
    {
        // value should be update via customize in-between start() and stop()
        auto tmp = record();
        accum += (value - tmp);
        value = std::move(std::max(value, tmp));
        set_stopped();
    }

    //----------------------------------------------------------------------------------//

    double get_display() const { return get(); }

    //----------------------------------------------------------------------------------//

    double get() const { return accum / base_type::get_unit(); }

    //----------------------------------------------------------------------------------//

    void customize(const std::string& fname, size_t nbytes)
    {
        auto _hash = string_hash()(fname);
        auto idx   = get_index(_hash);

        if(idx > get_hash_array().size())
        {
            if(settings::verbose() > 1 || settings::debug())
                printf("[%s]> unknown function: '%s'\n", this_type::label().c_str(),
                       fname.c_str());
            return;
        }

        if(_hash == prefix_hash)
        {
            // malloc
            value = (nbytes);
            accum += (nbytes);
        }
        else
        {
            if(settings::verbose() > 1 || settings::debug())
                printf("[%s]> skipped function '%s with hash %llu'\n",
                       this_type::label().c_str(), fname.c_str(),
                       (long long unsigned) _hash);
        }
    }

    //----------------------------------------------------------------------------------//

    void customize(const std::string& fname, size_t nmemb, size_t size)
    {
        auto _hash = string_hash()(fname);
        auto idx   = get_index(_hash);

        if(idx > get_hash_array().size())
        {
            if(settings::verbose() > 1 || settings::debug())
                printf("[%s]> unknown function: '%s'\n", this_type::label().c_str(),
                       fname.c_str());
            return;
        }

        if(_hash == prefix_hash)
        {
            // calloc
            value = (nmemb * size);
            accum += (nmemb * size);
        }
        else
        {
            if(settings::verbose() > 1 || settings::debug())
                printf("[%s]> skipped function '%s with hash %llu'\n",
                       this_type::label().c_str(), fname.c_str(),
                       (long long unsigned) _hash);
        }
    }

    //----------------------------------------------------------------------------------//

    void customize(const std::string& fname, void* ptr)
    {
        if(!ptr)
            return;
        auto _hash = string_hash()(fname);
        auto idx   = get_index(_hash);

        if(idx > get_hash_array().size())
        {
            if(settings::verbose() > 1 || settings::debug())
                printf("[%s]> unknown function: '%s'\n", this_type::label().c_str(),
                       fname.c_str());
            return;
        }

        // malloc
        if(idx < num_alloc)
            get_allocation_map()[ptr] = value;
        else
        {
            auto itr = get_allocation_map().find(ptr);
            if(itr != get_allocation_map().end())
            {
                value = itr->second;
                accum += itr->second;
                get_allocation_map().erase(itr);
            }
            else
            {
                if(settings::verbose() > 1 || settings::debug())
                    printf("[%s]> free of unknown pointer size: %p\n",
                           this_type::label().c_str(), ptr);
            }
        }
    }

    //----------------------------------------------------------------------------------//

#if defined(TIMEMORY_USE_CUDA)

    //----------------------------------------------------------------------------------//

    void customize(const std::string& fname, void** devPtr, size_t size)
    {
        auto _hash = string_hash()(fname);
        auto idx   = get_index(_hash);

        if(idx > get_hash_array().size())
        {
            if(settings::verbose() > 1 || settings::debug())
                printf("[%s]> unknown function: '%s'\n", this_type::label().c_str(),
                       fname.c_str());
            return;
        }

        if(_hash == prefix_hash)
        {
            // malloc
            value = (size);
            accum += (size);
            m_last_addr = devPtr;
        }
        else
        {
            if(settings::verbose() > 1 || settings::debug())
                printf("[%s]> skipped function '%s with hash %llu'\n",
                       this_type::label().c_str(), fname.c_str(),
                       (long long unsigned) _hash);
        }
    }

    //----------------------------------------------------------------------------------//

    void customize(const std::string& fname, cuda::error_t)
    {
        auto _hash = string_hash()(fname);
        auto idx   = get_index(_hash);

        if(idx > get_hash_array().size())
        {
            if(settings::verbose() > 1 || settings::debug())
                printf("[%s]> unknown function: '%s'\n", this_type::label().c_str(),
                       fname.c_str());
            return;
        }

        if(_hash == prefix_hash && idx < num_alloc)
        {
            // cudaMalloc
            if(m_last_addr)
            {
                void* ptr                 = (void*) ((char**) (m_last_addr)[0]);
                get_allocation_map()[ptr] = value;
            }
        }
        else if(_hash == prefix_hash && idx >= num_alloc)
        {
            // cudaFree
        }
        else
        {
            if(settings::verbose() > 1 || settings::debug())
                printf("[%s]> skipped function '%s with hash %llu'\n",
                       this_type::label().c_str(), fname.c_str(),
                       (long long unsigned) _hash);
        }
    }

    //----------------------------------------------------------------------------------//

#endif

    //----------------------------------------------------------------------------------//

    void set_prefix(const std::string& _prefix)
    {
        prefix      = _prefix;
        prefix_hash = add_hash_id(prefix);
        for(uintmax_t i = 0; i < get_hash_array().size(); ++i)
        {
            if(prefix_hash == get_hash_array()[i])
                prefix_idx = i;
        }
    }

    //----------------------------------------------------------------------------------//

    this_type& operator+=(const this_type& rhs)
    {
        value += rhs.value;
        accum += rhs.accum;
        if(rhs.is_transient)
            is_transient = rhs.is_transient;
        return *this;
    }

    //----------------------------------------------------------------------------------//

    this_type& operator-=(const this_type& rhs)
    {
        value -= rhs.value;
        accum -= rhs.accum;
        if(rhs.is_transient)
            is_transient = rhs.is_transient;
        return *this;
    }

private:
    using alloc_map_t  = std::unordered_map<void*, size_t>;
    using vaddr_map_t  = std::unordered_map<void**, size_t>;
    using hash_array_t = std::array<uintmax_t, data_size>;

    static alloc_map_t& get_allocation_map()
    {
        static thread_local alloc_map_t _instance;
        return _instance;
    }

    static vaddr_map_t& get_void_address_map()
    {
        static thread_local vaddr_map_t _instance;
        return _instance;
    }

    static hash_array_t& get_hash_array()
    {
        static auto _get = []() {
#if defined(TIMEMORY_USE_CUDA)
            hash_array_t _instance = {
                { string_hash()("malloc"),
                  string_hash()("calloc"),
                  string_hash()("cudaMalloc"),
                  string_hash()("free"),
                  string_hash("cudaFree") }
            };
#else
            hash_array_t _instance = { { string_hash()("malloc"), string_hash()("calloc"),
                                         string_hash()("free") } };
#endif

            return _instance;
        };

        static hash_array_t _instance = _get();
        return _instance;
    }

private:
    uintmax_t   prefix_hash = string_hash()("");
    uintmax_t   prefix_idx  = std::numeric_limits<uintmax_t>::max();
    std::string prefix      = "";
#if defined(TIMEMORY_USE_CUDA)
    void** m_last_addr = nullptr;
#endif
};

}  // namespace component

}  // namespace tim

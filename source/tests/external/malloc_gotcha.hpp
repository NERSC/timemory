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

#include "timemory/components/gotcha.hpp"
#include "timemory/details/settings.hpp"
#include "timemory/timemory.hpp"

#include <stdlib.h>

extern "C"
{
    extern void* malloc(size_t);
    extern void* calloc(size_t, size_t);
    extern void  free(void*);
}

namespace tim
{
namespace component
{
struct malloc_gotcha
: base<malloc_gotcha, std::array<size_t, 3>, policy::global_init, policy::global_finalize>
{
    static constexpr uintmax_t data_size = 3;
    using value_type                     = std::array<size_t, data_size>;
    using this_type                      = malloc_gotcha;
    using base_type =
        base<this_type, value_type, policy::global_init, policy::global_finalize>;
    using storage_type = typename base_type::storage_type;
    using string_hash  = std::hash<std::string>;

    static std::string label() { return "malloc_gotcha"; }
    static std::string description() { return "GOTCHA wrapper for memory allocation"; }
    static std::string display_unit() { return "GB"; }
    static int64_t     unit() { return units::gigabyte; }
    static const short precision                      = 3;
    static const short width                          = 8;
    static const std::ios_base::fmtflags format_flags = {};

    static value_type record() { return value_type{ 0, 0 }; }

    using base_type::accum;
    using base_type::is_transient;
    using base_type::value;

    malloc_gotcha()                 = default;
    ~malloc_gotcha()                = default;
    malloc_gotcha(const this_type&) = default;
    malloc_gotcha(this_type&&)      = default;
    malloc_gotcha& operator=(const this_type&) = default;
    malloc_gotcha& operator=(this_type&&) = default;

    void start() {}

    void stop() {}

    double get() const
    {
        auto& _obj = (is_transient) ? accum : value;
        return ((_obj[0] + _obj[1]) - _obj[2]) / base_type::get_unit();
    }

    double get_display() const { return get(); }

    void customize(const std::string& fname, size_t nbytes)
    {
        if(string_hash()(fname) == (*m_hash_array)[0])
        {
            // malloc
            value[0] = nbytes;
            accum[0] += nbytes;
        }
    }

    void customize(const std::string& fname, size_t nmemb, size_t size)
    {
        if(string_hash()(fname) == (*m_hash_array)[1])
        {
            // calloc
            value[1] = nmemb * size;
            accum[1] += nmemb * size;
        }
    }

    void customize(const std::string& fname, void* ptr)
    {
        auto _hash = string_hash()(fname);

        // malloc
        if(_hash == (*m_hash_array)[0])
            (*m_alloc_map)[ptr] = value[0];
        else if(_hash == (*m_hash_array)[1])
            (*m_alloc_map)[ptr] = value[1];
        else if(_hash == (*m_hash_array)[2])
        {
            auto itr = m_alloc_map->find(ptr);
            if(itr != m_alloc_map->end())
            {
                value[2] = itr->second;
                accum[2] += itr->second;
                m_alloc_map->erase(itr);
            }
            else
            {
                if(settings::verbose() > 1 || settings::debug())
                    printf("[%s]> free of unknown pointer size: %p\n",
                           this_type::label().c_str(), ptr);
            }
        }
    }

    static void invoke_global_init(storage_type*)
    {
        // add_hash_id("malloc");
        // add_hash_id("calloc");
        // add_hash_id("free");
    }

    static void invoke_global_finalize(storage_type*) {}

    //----------------------------------------------------------------------------------//

    this_type& operator+=(const this_type& rhs)
    {
        for(size_type i = 0; i < value.size(); ++i)
            value[i] += rhs.value[i];
        for(size_type i = 0; i < accum.size(); ++i)
            accum[i] += rhs.accum[i];
        if(rhs.is_transient)
            is_transient = rhs.is_transient;
        return *this;
    }

    //----------------------------------------------------------------------------------//

    this_type& operator-=(const this_type& rhs)
    {
        for(size_type i = 0; i < value.size(); ++i)
            value[i] -= rhs.value[i];
        for(size_type i = 0; i < accum.size(); ++i)
            accum[i] -= rhs.accum[i];
        if(rhs.is_transient)
            is_transient = rhs.is_transient;
        return *this;
    }

private:
    using alloc_map_t  = std::unordered_map<void*, size_t>;
    using hash_array_t = std::array<uintmax_t, data_size>;

    static alloc_map_t& get_allocation_map()
    {
        static thread_local alloc_map_t _instance;
        return _instance;
    }

    static hash_array_t& get_hash_array()
    {
        static auto _get = []() {
            hash_array_t _instance = { { string_hash()("malloc"), string_hash()("calloc"),
                                         string_hash()("free") } };
            return _instance;
        };

        static hash_array_t _instance = _get();
        return _instance;
    }

    alloc_map_t*  m_alloc_map  = &get_allocation_map();
    hash_array_t* m_hash_array = &get_hash_array();
};

}  // namespace component

namespace trait
{
template <>
struct supports_args<component::malloc_gotcha, std::tuple<std::string, size_t>>
: std::true_type
{
};

template <>
struct supports_args<component::malloc_gotcha, std::tuple<std::string, size_t, size_t>>
: std::true_type
{
};

template <>
struct supports_args<component::malloc_gotcha, std::tuple<std::string, void*>>
: std::true_type
{
};

template <>
struct uses_memory_units<component::malloc_gotcha> : std::true_type
{
};

template <>
struct is_memory_category<component::malloc_gotcha> : std::true_type
{
};
}  // namespace trait

}  // namespace tim

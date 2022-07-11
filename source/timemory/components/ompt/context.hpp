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

#include "timemory/components/ompt/backends.hpp"
#include "timemory/macros/language.hpp"  // string_view
#include "timemory/variadic/component_tuple.hpp"
#include "timemory/variadic/macros.hpp"

#include <atomic>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace tim
{
namespace openmp
{
template <typename Tp>
std::atomic<uint64_t>&
get_counter()
{
    static std::atomic<uint64_t> _v{ 0 };
    return _v;
}

template <typename BundleT, typename... Args>
inline void
context_store(string_view_cref_t _key, Args... _args)
{
    using bundle_type = BundleT;

    bundle_type _v{ _key };
    _v.construct(_args...);
    _v.start();
    _v.store(_args...);
    _v.stop();
}

template <typename Tp, typename... Args>
inline void
context_construct(string_view_cref_t _key, Tp& _data, ompt_data_t* _ompt_data,
                  Args... _args)
{
    using bundle_type = std::remove_pointer_t<std::decay_t<typename Tp::mapped_type>>;

    if(!_ompt_data)
        throw std::runtime_error(
            TIMEMORY_JOIN("", "Error! nullptr to ompt_data_t! key = ", _key));

    auto _idx = _ompt_data->value;
    if(_idx == 0)
        _idx = _ompt_data->value = ++get_counter<bundle_type>();

    if(_data[_idx])
        throw std::runtime_error(TIMEMORY_JOIN(
            "", "Error! attempt to overwrite an existing bundle! existing: ",
            _data.at(_idx)->key(), ", new: ", _key));

    _data.at(_idx) = new bundle_type{ _key };
    auto _v        = _data.at(_idx);
    _v->construct(_args...);
}

template <typename Tp, typename... Args>
inline void
context_start_constructed(string_view_cref_t _key, Tp& _data, ompt_data_t* _ompt_data,
                          Args... _args)
{
    if(!_ompt_data)
        throw std::runtime_error(
            TIMEMORY_JOIN("", "Error! nullptr to ompt_data_t! key = ", _key));

    auto _idx = _ompt_data->value;
    if(_ompt_data->value == 0)
        throw std::runtime_error(
            TIMEMORY_JOIN("", "Error! Missing value in ompt_data_t! key = ", _key));

    if(_data.count(_idx) == 0)
        throw std::runtime_error(TIMEMORY_JOIN("", "Error! data does not contain index ",
                                               _idx, "! key = ", _key));

    auto _v = _data.at(_idx);
    _v->start(_args...);
}

template <typename Tp, typename... Args>
inline void
context_start(string_view_cref_t _key, Tp& _data, ompt_data_t* _ompt_data, Args... _args)
{
    context_construct(_key, _data, _ompt_data, _args...);
    context_start_constructed(_key, _data, _ompt_data, _args...);
}

template <typename Tp, typename... Args>
inline bool
context_relaxed_stop(string_view_cref_t _key, Tp& _data, ompt_data_t* _ompt_data,
                     Args... _args)
{
    if(!_ompt_data)
        throw std::runtime_error(
            TIMEMORY_JOIN("", "Error! nullptr to ompt_data_t! key = ", _key));

    if(_ompt_data->value == 0)
        throw std::runtime_error(
            TIMEMORY_JOIN("", "Error! Missing value in ompt_data_t! key = ", _key));

    auto _idx = _ompt_data->value;

    if(_data.count(_idx) == 0)
        throw std::runtime_error(TIMEMORY_JOIN("", "Error! data does not contain index ",
                                               _idx, "! key = ", _key));

    auto _v = _data.at(_idx);

    if(!_v)
        return false;

    _v->stop(_args...);
    delete _v;
    _data.at(_idx) = nullptr;
    return true;
}

template <typename Tp, typename... Args>
inline void
context_stop(string_view_cref_t _key, Tp& _data, ompt_data_t* _ompt_data, Args... _args)
{
    if(!context_relaxed_stop(_key, _data, _ompt_data, _args...))
        throw std::runtime_error(
            TIMEMORY_JOIN("", "Error! attempt to stop a missing bundle! key: ", _key));
}

template <typename Tp, typename... Args>
inline void
context_endpoint(string_view_cref_t _key, Tp& _data, ompt_scope_endpoint_t endpoint,
                 ompt_data_t* _ompt_data, Args... _args)
{
    if(endpoint == ompt_scope_begin)
    {
        context_start(_key, _data, _ompt_data, _args...);
    }
    else if(endpoint == ompt_scope_end)
    {
        context_stop(_key, _data, _ompt_data, _args...);
    }
    else
    {
        throw std::runtime_error("Error! Unknown endpoint value :: " +
                                 std::to_string(static_cast<int>(endpoint)));
    }
}
}  // namespace openmp
}  // namespace tim

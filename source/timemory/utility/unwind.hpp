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

#include "timemory/defines.h"
#include "timemory/macros/compiler.hpp"
#include "timemory/macros/language.hpp"
#include "timemory/macros/os.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/utility/optional.hpp"
#include "timemory/utility/types.hpp"

#if defined(TIMEMORY_USE_LIBUNWIND)
#    include <libunwind.h>
#endif

#include <array>
#include <cstdint>
#include <string>

namespace tim
{
namespace unwind
{
//
//--------------------------------------------------------------------------------------//
//
#if defined(TIMEMORY_USE_LIBUNWIND)
//
//--------------------------------------------------------------------------------------//
//
struct entry
{
    unw_word_t register_addr = {};  // instruction/stack pointer

    TIMEMORY_DEFAULT_OBJECT(entry)

    entry(unw_word_t _addr)
    : register_addr{ _addr }
    {}

    unw_word_t address() const { return register_addr; }

    template <size_t BufferSize = 1024>
    std::string& get_name(unw_context_t _context, std::string&,
                          unw_word_t*   _off = nullptr) const;

    template <size_t BufferSize = 1024, bool Shrink = true>
    std::string get_name(unw_context_t _context, unw_word_t* _off = nullptr) const;
};
//
template <size_t N>
struct stack
{
    using array_type     = std::array<stl::optional<entry>, N>;
    using iterator       = typename array_type::iterator;
    using const_iterator = typename array_type::const_iterator;

    TIMEMORY_DEFAULT_OBJECT(stack)

    stack(unw_frame_regnum_t _regnum)
    : regnum{ _regnum }
    {}

    auto&       operator[](size_t _idx) { return call_stack[_idx]; }
    const auto& operator[](size_t _idx) const { return call_stack[_idx]; }

    auto&       at(size_t _idx) { return call_stack.at(_idx); }
    const auto& at(size_t _idx) const { return call_stack.at(_idx); }

    size_t size() const;
    bool   valid() const { return size() > 0; }
    bool   empty() const { return size() == 0; }

    iterator       begin() { return call_stack.begin(); }
    const_iterator begin() const { return call_stack.begin(); }

    iterator       end();
    const_iterator end() const;

    unw_frame_regnum_t regnum     = {};
    unw_cursor_t       cursor     = {};
    unw_context_t      context    = {};
    array_type         call_stack = {};
};
//
template <size_t N>
inline size_t
stack<N>::size() const
{
    size_t _n = 0;
    for(auto&& itr : call_stack)
        _n += (itr) ? 1 : 0;
    return _n;
}
//
template <size_t N>
inline typename stack<N>::iterator
stack<N>::end()
{
    auto itr = call_stack.begin();
    std::advance(itr, size());
    return itr;
}
//
template <size_t N>
inline typename stack<N>::const_iterator
stack<N>::end() const
{
    auto itr = call_stack.begin();
    std::advance(itr, size());
    return itr;
}
//
template <size_t DefaultBufferSize>
inline std::string&
entry::get_name(unw_context_t _context, std::string& _name, unw_word_t* _off) const
{
    // use this as a reference capacity for a default string since std::string
    // in most (if not all) C++ STL libraries have a default capacity > 0
    // (small string optimization)
    const auto _default_string_capacity = std::string{}.capacity();

    unw_word_t _off_v = {};  // offset

    // make sure there is enough space in the string
    if(_name.capacity() <= _default_string_capacity)
        _name.reserve(DefaultBufferSize);

    // unw_get_proc_name will fill until it hits \0 so move it to the end of
    // the strings capacity. Since we only resize to the capacity, there
    // is no reallocation
    if(_name.length() < _name.capacity())
        _name.resize(_name.capacity(), ' ');

    // get the procedure name and offset
    auto _err = unw_get_proc_name_by_ip(unw_local_addr_space, register_addr, _name.data(),
                                        _name.capacity(), &_off_v, &_context);

    // provide the error info if failed but if the procedure name is too long to fit in
    // the buffer provided and a truncated version of the name has been returned,
    // keep the truncated version of the name
    if(_err != 0 && _err != UNW_ENOMEM)
        _name.assign(unw_strerror(_err));

    // provide the offset if requested
    if(_off)
        *_off = _off_v;

    // there are going to be multiple \0 in the string because of the resize
    // and unw_get_proc_name. this should just update the string metadata
    // and not reallocate
    auto _pos = _name.find('\0');
    if(_pos < _name.length())
        _name = _name.assign(_name.data(), _name.data() + _pos);

    return _name;
}
//
template <size_t BufferSize, bool Shrink>
inline std::string
entry::get_name(unw_context_t _context, unw_word_t* _off) const
{
    std::string _name = {};
    _name.reserve(BufferSize);
    get_name<BufferSize>(_context, _name, _off);
    if(Shrink)
        _name.shrink_to_fit();
    return _name;
}
//
//--------------------------------------------------------------------------------------//
//
#else
//
//--------------------------------------------------------------------------------------//
//
struct entry
{};
//
template <size_t N>
struct stack
{};
//
//--------------------------------------------------------------------------------------//
//
#endif
//
//--------------------------------------------------------------------------------------//
//
}  // namespace unwind
}  // namespace tim

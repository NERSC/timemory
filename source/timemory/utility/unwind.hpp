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
#include "timemory/tpls/cereal/cereal/cereal.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/utility/optional.hpp"
#include "timemory/utility/types.hpp"

#if defined(TIMEMORY_USE_LIBUNWIND)
#    include "timemory/utility/dlinfo.hpp"
#    include "timemory/utility/procfs/maps.hpp"

#    include <libunwind.h>
#    if defined(unw_get_proc_name_by_ip)
#        define TIMEMORY_LIBUNWIND_HAS_PROC_NAME_BY_IP 1
#    else
#        define TIMEMORY_LIBUNWIND_HAS_PROC_NAME_BY_IP 0
#    endif
#endif

#include <array>
#include <cstdint>
#include <string>

#if defined(TIMEMORY_USE_LIBUNWIND)
namespace tim
{
namespace unwind
{
struct entry;
}
}  // namespace tim

namespace std
{
template <>
struct hash<tim::unwind::entry>
{
    size_t operator()(tim::unwind::entry) const;
};
}  // namespace std
#endif

namespace tim
{
namespace unwind
{
//
//--------------------------------------------------------------------------------------//
//
#if defined(TIMEMORY_USE_LIBUNWIND)
//
struct entry
{
    using error_handler_func_t = std::function<void(int, std::string&)>;

    TIMEMORY_DEFAULT_OBJECT(entry)

    entry(unw_word_t _addr, unw_cursor_t*);
    explicit entry(unw_word_t _addr);

#    if TIMEMORY_LIBUNWIND_HAS_PROC_NAME_BY_IP == 1
    unw_word_t register_addr = {};  // instruction/stack pointer
#    else
    unw_word_t   register_addr = {};  // instruction/stack pointer
    unw_cursor_t cursor        = {};  // copy of cursor is required
#    endif

    unw_word_t address() const { return register_addr; }

    int get_name(unw_context_t _context, char* _buffer, size_t _size,
                 unw_word_t* _off) const;

    template <size_t BufferSize = 1024>
    std::string& get_name(unw_context_t _context, std::string&,
                          unw_word_t* _off = nullptr, int* _err = nullptr) const;

    template <size_t BufferSize = 1024, bool Shrink = true>
    std::string get_name(unw_context_t _context, unw_word_t* _off = nullptr,
                         int* _err = nullptr) const;

    template <typename FuncT>
    static auto set_error_handler(FuncT&& _func)
    {
        auto _old       = error_handler();
        error_handler() = std::forward<FuncT>(_func);
        return _old;
    }

    template <typename ArchiveT>
    void serialize(ArchiveT& ar, const unsigned)
    {
        ar(cereal::make_nvp("address", register_addr));
    }

    bool operator==(entry _rhs) const { return (register_addr == _rhs.register_addr); }
    bool operator<(entry _rhs) const { return (register_addr < _rhs.register_addr); }
    bool operator>(entry _rhs) const { return (register_addr > _rhs.register_addr); }
    bool operator!=(entry _rhs) const { return !(*this == _rhs); }
    bool operator<=(entry _rhs) const { return (*this < _rhs) || (*this == _rhs); }
    bool operator>=(entry _rhs) const { return (*this > _rhs) || (*this == _rhs); }

private:
    static std::function<void(int, std::string&)>& error_handler()
    {
        static error_handler_func_t _v = [](int _err, std::string& _dest) {
            // provide the error info if failed but if the procedure name is too long to
            // fit in the buffer provided and a truncated version of the name has been
            // returned, keep the truncated version of the name
            if(_err != UNW_ENOMEM)
                _dest.assign(unw_strerror(_err));
        };
        return _v;
    };
};
//
//--------------------------------------------------------------------------------------//
//
struct processed_entry
{
    int         error        = 0;
    unw_word_t  address      = 0;
    unw_word_t  offset       = 0;
    unw_word_t  line_address = 0;   // line address in file
    std::string name         = {};  // function name
    std::string location     = {};  // file location
    dlinfo      info         = {};  // dynamic library info

    bool operator==(const processed_entry& _v) const;
    bool operator<(const processed_entry& _v) const;
    bool operator>(const processed_entry& _v) const;
    bool operator!=(const processed_entry& _v) const { return !(*this == _v); }
    bool operator<=(const processed_entry& _v) const { return !(*this > _v); }
    bool operator>=(const processed_entry& _v) const { return !(*this < _v); }

    template <typename ArchiveT>
    void serialize(ArchiveT& ar, const unsigned)
    {
        ar(cereal::make_nvp("error", error), cereal::make_nvp("address", address),
           cereal::make_nvp("offset", offset),
           cereal::make_nvp("line_address", line_address), cereal::make_nvp("name", name),
           cereal::make_nvp("location", location), cereal::make_nvp("dlinfo", info));
    }
};
//
inline bool
processed_entry::operator==(const processed_entry& _v) const
{
    return std::tie(error, address, offset, name, location) ==
           std::tie(_v.error, _v.address, _v.offset, _v.name, _v.location);
}

inline bool
processed_entry::operator<(const processed_entry& _v) const
{
    return std::tie(name, location, offset, address, error) <
           std::tie(_v.name, _v.location, _v.offset, _v.address, _v.error);
}

inline bool
processed_entry::operator>(const processed_entry& _v) const
{
    return !(*this == _v && *this < _v);
}
//
//--------------------------------------------------------------------------------------//
//
template <size_t N>
struct stack
{
    using array_type     = std::array<stl::optional<entry>, N>;
    using iterator       = typename array_type::iterator;
    using const_iterator = typename array_type::const_iterator;
    using cache_type     = std::unordered_map<entry, processed_entry>;

    TIMEMORY_DEFAULT_OBJECT(stack)

    explicit stack(unw_frame_regnum_t _regnum)
    : regnum{ _regnum }
    {}

    template <size_t RhsN, std::enable_if_t<N != RhsN, int> = 0>
    stack& operator=(stack<RhsN> _rhs);

    auto&       operator[](size_t _idx) { return call_stack[_idx]; }
    const auto& operator[](size_t _idx) const { return call_stack[_idx]; }

    auto&       at(size_t _idx) { return call_stack.at(_idx); }
    const auto& at(size_t _idx) const { return call_stack.at(_idx); }

    size_t size() const;
    bool   valid() const { return size() > 0; }
    bool   empty() const { return size() == 0; }

    template <typename Tp>
    auto& emplace_back(Tp&&);

    // shift the call-stack
    stack& shift(int64_t _n = -1);

    iterator       begin() { return call_stack.begin(); }
    const_iterator begin() const { return call_stack.begin(); }

    iterator       end();
    const_iterator end() const;

    unw_frame_regnum_t regnum     = {};
    unw_cursor_t       cursor     = {};
    unw_context_t      context    = {};
    array_type         call_stack = {};

    template <size_t DefaultBufferSize = 4096, bool Shrink = true>
    std::vector<processed_entry> get(cache_type* _cache              = nullptr,
                                     bool        _include_with_error = false) const;

    template <size_t DefaultBufferSize = 4096, bool Shrink = true>
    std::vector<processed_entry> get(bool _include_with_error) const
    {
        return get<DefaultBufferSize, Shrink>(nullptr, _include_with_error);
    }

    template <size_t RhsN>
    bool operator==(stack<RhsN> _rhs) const;

    template <typename ArchiveT>
    void TIMEMORY_CEREAL_SAVE_FUNCTION_NAME(ArchiveT& ar, const unsigned) const
    {
        auto _data = get();
        ar(cereal::make_nvp("regnum", regnum), cereal::make_nvp("data", _data));
    }

    template <typename ArchiveT>
    void TIMEMORY_CEREAL_LOAD_FUNCTION_NAME(ArchiveT& ar, const unsigned)
    {
        auto _data = std::vector<processed_entry>{};
        ar(cereal::make_nvp("regnum", regnum), cereal::make_nvp("data", _data));
        size_t _idx = 0;
        for(const auto& itr : _data)
            call_stack.at(_idx++) = entry{ itr.address };
    }
};
//
// finds the minimum size b/t LhsN and RhsN and then searches
// for the overlap.
template <size_t LhsN, size_t RhsN>
inline auto
get_common_stack(stack<LhsN> _lhs_v, stack<RhsN> _rhs_v)
{
    if constexpr(LhsN == RhsN)
    {
        return std::make_tuple(_lhs_v, _rhs_v);
    }
    else
    {
        constexpr size_t MinN = (LhsN < RhsN) ? LhsN : RhsN;
        constexpr size_t MaxN = (LhsN < RhsN) ? RhsN : LhsN;
        using return_type     = std::tuple<stack<MinN>, stack<MinN>>;
        stack<MinN>* _lhs     = nullptr;
        stack<MaxN>* _rhs     = nullptr;

        if constexpr(LhsN < RhsN)
        {
            _lhs = &_lhs_v;
            _rhs = &_rhs_v;
        }
        else
        {
            _lhs = &_rhs_v;
            _rhs = &_lhs_v;
        }

        auto _copy = [](const auto* _stack, size_t _beg) {
            constexpr size_t N    = (LhsN < RhsN) ? LhsN : RhsN;
            auto             _v   = stack<N>{};
            size_t           _idx = _beg;
            for(size_t i = 0; i < N; ++i, ++_idx)
            {
                if(_idx < _stack->call_stack.size())
                    _v.call_stack.at(i) = _stack->call_stack.at(_idx);
            }
            return _v;
        };

        //
        size_t _didx = _lhs->size() - (MaxN - MinN);
        size_t _lidx = 0;
        size_t _ridx = 0;
        for(size_t j = 0; j < _rhs->size(); ++j)
        {
            // only search, at most, in range of i to i + (MaxN - MinN)
            // we don't want to return them the overlap of just main/start_thread
            for(size_t i = 0; i < _didx; ++i)
            {
                if(_lhs->at(i)->address() == _rhs->at(j)->address())
                {
                    // first match
                    if(_lidx == 0 && _ridx == 0)
                    {
                        _lidx = i;
                        _ridx = j;
                    }
                    break;
                    if(_copy(_lhs, i) == _copy(_rhs, j))
                        return return_type{ _copy(_lhs, i), _copy(_rhs, j) };
                }
            }
        }

        return return_type{ _copy(_lhs, _lidx), _copy(_rhs, _ridx) };
    }
}
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
template <size_t N>
template <typename Tp>
inline auto&
stack<N>::emplace_back(Tp&& _v)
{
    size_t _n = size();
    if(_n < N)
    {
        call_stack.at(_n) = std::forward<Tp>(_v);
        return call_stack.at(_n);
    }
    return call_stack.back();
}
//
template <size_t N>
inline stack<N>&
stack<N>::shift(int64_t _n)
{
    using entry_t = stl::optional<entry>;
    std::vector<entry_t> _stack{};
    int64_t              _size = size();
    _stack.reserve(_size);
    for(auto itr : *this)
        _stack.emplace_back(itr);
    call_stack.fill(entry_t{});

    for(int64_t i = 0; i < _size; ++i)
    {
        int64_t _idx = i + _n;
        if(_idx >= 0 && _idx < static_cast<int64_t>(N))
        {
            call_stack.at(_idx) = _stack.at(i);
        }
    }
    return *this;
}
//
template <size_t N>
template <size_t DefaultBufferSize, bool Shrink>
std::vector<processed_entry>
stack<N>::get(cache_type* _cache, bool _include_with_error) const
{
    std::vector<processed_entry> _data{};
    _data.reserve(size());
    for(auto itr : *this)
    {
        if(itr)
        {
            if(_cache)
            {
                auto citr = _cache->find(*itr);
                if(citr != _cache->end())
                {
                    if(citr->second.error == 0 || _include_with_error)
                        _data.emplace_back(citr->second);
                    continue;
                }
            }
            processed_entry _v{};
            _v.address = itr->address();
            _v.name    = itr->template get_name<DefaultBufferSize, Shrink>(
                context, &_v.offset, &_v.error);
            _v.info     = dlinfo::construct(_v.address - _v.offset);
            _v.location = std::string{ _v.info.location.name };
            if(_v.info)
            {
                _v.line_address =
                    (_v.info.symbol.address() - _v.info.location.address()) + _v.offset;
            }
            else
            {
                auto _map = procfs::find_map(_v.address);
                if(!_map.is_empty())
                    _v.line_address = (_v.address - _map.start_address) + _map.offset;
            }
            if(_v.error == 0 || _include_with_error)
            {
                _data.emplace_back(_v);
            }
            if(_cache)
                _cache->emplace(*itr, _v);
        }
    }
    return _data;
}
//
template <size_t N>
template <size_t RhsN, std::enable_if_t<N != RhsN, int>>
inline stack<N>&
stack<N>::operator=(stack<RhsN> _rhs)
{
    static_assert(N != RhsN, "Error! Bad overload resolution");
    regnum  = _rhs.regnum;
    cursor  = _rhs.cursor;
    context = _rhs.context;
    for(size_t i = 0; i < std::min<size_t>(N, RhsN); ++i)
        call_stack[i] = _rhs.call_stack[i];
    return *this;
}
//
template <size_t N>
template <size_t RhsN>
inline bool
stack<N>::operator==(stack<RhsN> _rhs) const
{
    if constexpr(N == RhsN)
    {
        constexpr size_t LhsN = N;
        const auto&      _lhs = *this;
        for(size_t i = 0; i < LhsN; ++i)
        {
            size_t _n = ((_lhs[i]) ? 1 : 0) + ((_rhs[i]) ? 1 : 0);
            // if one has an entry and the other doesn't, not equal
            if(_n == 1)
                return false;
            else if(_n == 2)
            {
                if(_lhs[i]->address() != _rhs[i]->address())
                    return false;
            }
        }
        return true;
    }
    else
    {
        auto _common = get_common_stack(*this, _rhs);
        return (std::get<0>(_common) == std::get<1>(_common));
    }
}
//
inline entry::entry(unw_word_t _addr)
: register_addr{ _addr }
{}
//
#    if TIMEMORY_LIBUNWIND_HAS_PROC_NAME_BY_IP == 1
inline entry::entry(unw_word_t _addr, unw_cursor_t*)
: register_addr{ _addr }
{}
#    else
inline entry::entry(unw_word_t _addr, unw_cursor_t* _cursor)
: register_addr{ _addr }
, cursor{ *_cursor }
{}
#    endif
//
inline int
entry::get_name(unw_context_t _context, char* _buffer, size_t _size,
                unw_word_t* _off) const
{
#    if TIMEMORY_LIBUNWIND_HAS_PROC_NAME_BY_IP == 1
    return unw_get_proc_name_by_ip(unw_local_addr_space, register_addr, _buffer, _size,
                                   _off, &_context);
#    else
    (void) _context;
    auto* _cursor = &const_cast<unw_cursor_t&>(cursor);
    return unw_get_proc_name(_cursor, _buffer, _size, _off);
#    endif
}
//
template <size_t DefaultBufferSize>
inline std::string&
entry::get_name(unw_context_t _context, std::string& _name, unw_word_t* _off,
                int* _err_ret) const
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
    auto _err = get_name(_context, _name.data(), _name.capacity(), &_off_v);

    // provide the error
    if(_err_ret)
        *_err_ret = _err;

    // call the error handler
    if(_err != 0)
        error_handler()(_err, _name);

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
entry::get_name(unw_context_t _context, unw_word_t* _off, int* _err_ptr) const
{
    if(_err_ptr)
    {
        std::string _name = {};
        _name.reserve(BufferSize);
        get_name<BufferSize>(_context, _name, _off, _err_ptr);
        if(Shrink)
            _name.shrink_to_fit();
        return _name;
    }
    else
    {
        size_t      _buffer_size = BufferSize;
        int         _err         = 0;
        std::string _name        = {};
        do
        {
            _name.reserve(_buffer_size);
            get_name<BufferSize>(_context, _name, _off, &_err);
            if(_err == ENOMEM)
            {
                _buffer_size *= 2;
            }
        } while(_err == ENOMEM);
        if(Shrink)
            _name.shrink_to_fit();
        return _name;
    }
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

#if defined(TIMEMORY_USE_LIBUNWIND)
namespace std
{
inline size_t
hash<tim::unwind::entry>::operator()(tim::unwind::entry _v) const
{
    return std::hash<unw_word_t>{}(_v.address());
}
}  // namespace std
#endif

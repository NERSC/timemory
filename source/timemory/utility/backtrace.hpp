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

#include "timemory/macros/compiler.hpp"
#include "timemory/macros/language.hpp"
#include "timemory/macros/os.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/utility/types.hpp"

#if defined(TIMEMORY_USE_LIBUNWIND)
#    include <libunwind.h>
#endif

#if defined(TIMEMORY_UNIX)
#    include <execinfo.h>
#endif

// C library
#include <array>
#include <cctype>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

namespace tim
{
#if defined(TIMEMORY_UNIX)
//
TIMEMORY_UTILITY_INLINE std::string
                        demangle_backtrace(const char* cstr);
//
TIMEMORY_UTILITY_INLINE std::string
                        demangle_backtrace(const std::string& str);
//
TIMEMORY_UTILITY_INLINE std::string
                        demangle_unw_backtrace(const char* cstr);
//
TIMEMORY_UTILITY_INLINE std::string
                        demangle_unw_backtrace(const std::string& str);
//
template <size_t Depth, size_t Offset = 1>
TIMEMORY_NOINLINE inline auto
get_backtrace()
{
    static_assert(Depth > 0, "Error !(Depth > 0)");
    static_assert(Offset >= 0, "Error !(Offset >= 0)");

    // destination
    std::array<char[512], Depth> btrace{};
    for(auto& itr : btrace)
        itr[0] = '\0';

    // plus one for this stack-frame
    std::array<void*, Depth + Offset> buffer{};
    buffer.fill(nullptr);

    // size of returned buffer
    auto sz = ::backtrace(buffer.data(), Depth + Offset);
    // size of relevant data
    auto n = sz - Offset;

    // skip ahead (Offset + 1) stack frames
    char** bsym = ::backtrace_symbols(buffer.data() + Offset, n);

    // report errors
    if(bsym == nullptr)
    {
        ::perror("backtrace_symbols");
    }
    else
    {
        for(decltype(n) i = 0; i < n; ++i)
        {
            ::snprintf(btrace[i], sizeof(btrace[i]), "%s", bsym[i]);
        }
    }
    return btrace;
}
//
template <size_t Depth, size_t Offset = 1, bool WFuncOffset = true>
TIMEMORY_NOINLINE inline auto
get_unw_backtrace()
{
#    if defined(TIMEMORY_USE_LIBUNWIND)
    static_assert(Depth > 0, "Error !(Depth > 0)");
    static_assert(Offset >= 0, "Error !(Offset >= 0)");

    unw_cursor_t  cursor{};
    unw_context_t context{};

    // destination
    std::array<char[512], Depth> btrace{};
    for(auto& itr : btrace)
        itr[0] = '\0';

    // Initialize cursor to current frame for local unwinding.
    unw_getcontext(&context);
    if(unw_init_local(&cursor, &context) < 0)
    {
        return btrace;
    }

    size_t tot_idx = 0;
    while(unw_step(&cursor) > 0)
    {
        unw_word_t ip{};   // stack pointer
        unw_word_t off{};  // offset
        auto       _idx = ++tot_idx;
        if(_idx >= Depth + Offset)
            break;
        unw_get_reg(&cursor, UNW_REG_IP, &ip);
        if(ip == 0)
            break;
        constexpr size_t NameSize = (WFuncOffset) ? 496 : 512;
        char             name[NameSize];
        name[0] = '\0';
        if(unw_get_proc_name(&cursor, name, sizeof(name), &off) == 0)
        {
            if(_idx >= Offset)
            {
                auto _lidx = _idx - Offset;
                if(WFuncOffset && off != 0)
                    snprintf(btrace[_lidx], sizeof(btrace[_lidx]), "%s +0x%lx", name,
                             (long) off);
                else
                    snprintf(btrace[_lidx], sizeof(btrace[_lidx]), "%s", name);
            }
        }
    }
#    else
    std::array<char[512], Depth> btrace{};
    throw std::runtime_error("[timemory]> libunwind not available");
#    endif
    return btrace;
}
//
template <size_t Depth, size_t Offset = 1, typename Func>
TIMEMORY_NOINLINE inline auto
get_backtrace(Func&& func)
{
    static_assert(Depth > 0, "Error !(Depth > 0)");
    static_assert(Offset >= 0, "Error !(Offset >= 0)");

    using type = std::result_of_t<Func(const char[512])>;
    // destination
    std::array<type, Depth> btrace{};

    auto&& _data = ::tim::get_backtrace<Depth, Offset + 1>();
    auto   _n    = _data.size();
    for(decltype(_n) i = 0; i < _n; ++i)
        btrace[i] = func(_data[i]);
    return btrace;
}
//
template <size_t Depth, size_t Offset = 1, bool WFuncOffset = true, typename Func>
TIMEMORY_NOINLINE inline auto
get_unw_backtrace(Func&& func)
{
    static_assert(Depth > 0, "Error !(Depth > 0)");
    static_assert(Offset >= 0, "Error !(Offset >= 0)");

    using type = std::result_of_t<Func(const char[512])>;
    // destination
    std::array<type, Depth> btrace{};

    auto&& _data = ::tim::get_unw_backtrace<Depth, Offset + 1, WFuncOffset>();
    auto   _n    = _data.size();
    for(decltype(_n) i = 0; i < _n; ++i)
        btrace[i] = func(_data[i]);
    return btrace;
}
//
//--------------------------------------------------------------------------------------//
//
template <size_t Depth, size_t Offset = 1>
TIMEMORY_NOINLINE inline auto
get_demangled_backtrace()
{
    auto demangle_bt = [](const char cstr[512]) { return demangle_backtrace(cstr); };
    return get_backtrace<Depth, Offset + 1>(demangle_bt);
}
//
//--------------------------------------------------------------------------------------//
//
template <size_t Depth, size_t Offset = 1, bool WFuncOffset = true>
TIMEMORY_NOINLINE inline auto
get_demangled_unw_backtrace()
{
    auto demangle_bt = [](const char cstr[512]) { return demangle_unw_backtrace(cstr); };
    return get_unw_backtrace<Depth, Offset + 1, WFuncOffset>(demangle_bt);
}
//
//--------------------------------------------------------------------------------------//
//
template <size_t Depth, size_t Offset = 2>
TIMEMORY_NOINLINE inline std::ostream&
print_backtrace(std::ostream& os = std::cerr, std::string _prefix = "",
                const std::string& _info = "", const std::string& _indent = "    ")
{
    os << _indent.substr(0, _indent.length() / 2) << "Backtrace";
    if(!_info.empty())
        os << " " << _info;
    os << ":\n" << std::flush;
    auto bt = tim::get_backtrace<Depth, Offset>();
    if(!_prefix.empty() && _prefix.find_last_of(" \t") != _prefix.length() - 1)
        _prefix += " ";
    for(const auto& itr : bt)
    {
        if(strlen(itr) > 0)
            os << _indent << _prefix << itr << "\n";
    }
    os << std::flush;
    return os;
}
//
//--------------------------------------------------------------------------------------//
//
template <size_t Depth, size_t Offset = 2>
TIMEMORY_NOINLINE inline std::ostream&
print_demangled_backtrace(std::ostream& os = std::cerr, std::string _prefix = "",
                          const std::string& _info   = "",
                          const std::string& _indent = "    ")
{
    os << _indent.substr(0, _indent.length() / 2) << "Backtrace";
    if(!_info.empty())
        os << " " << _info;
    os << ":\n" << std::flush;
    auto bt = tim::get_demangled_backtrace<Depth, Offset>();
    if(!_prefix.empty() && _prefix.find_last_of(" \t") != _prefix.length() - 1)
        _prefix += " ";
    for(const auto& itr : bt)
    {
        if(itr.length() > 0)
            os << _indent << _prefix << itr << "\n";
    }
    os << std::flush;
    return os;
}
//
//--------------------------------------------------------------------------------------//
//
template <size_t Depth, size_t Offset = 2, bool WFuncOffset = true>
TIMEMORY_NOINLINE inline std::ostream&
print_unw_backtrace(std::ostream& os = std::cerr, std::string _prefix = "",
                    const std::string& _info = "", const std::string& _indent = "    ")
{
    os << _indent.substr(0, _indent.length() / 2) << "Backtrace";
    if(!_info.empty())
        os << " " << _info;
    os << ":\n" << std::flush;
    auto bt = tim::get_unw_backtrace<Depth, Offset, WFuncOffset>();
    if(!_prefix.empty() && _prefix.find_last_of(" \t") != _prefix.length() - 1)
        _prefix += " ";
    for(const auto& itr : bt)
    {
        if(strlen(itr) > 0)
            os << _indent << _prefix << itr << "\n";
    }
    os << std::flush;
    return os;
}
//
//--------------------------------------------------------------------------------------//
//
template <size_t Depth, size_t Offset = 3>
TIMEMORY_NOINLINE inline std::ostream&
print_demangled_unw_backtrace(std::ostream& os = std::cerr, std::string _prefix = "",
                              const std::string& _info   = "",
                              const std::string& _indent = "    ")
{
    os << _indent.substr(0, _indent.length() / 2) << "Backtrace";
    if(!_info.empty())
        os << " " << _info;
    os << ":\n" << std::flush;
    auto bt = tim::get_demangled_unw_backtrace<Depth, Offset>();
    if(!_prefix.empty() && _prefix.find_last_of(" \t") != _prefix.length() - 1)
        _prefix += " ";
    for(const auto& itr : bt)
    {
        if(itr.length() > 0)
            os << _indent << _prefix << itr << "\n";
    }
    os << std::flush;
    return os;
}
//
#else
//
// define these dummy functions since they are used in operation::decode
//
static inline auto
demangle_backtrace(const char* cstr)
{
    return std::string{ cstr };
}
//
static inline auto
demangle_backtrace(const std::string& str)
{
    return str;
}
//
template <size_t Depth, size_t Offset = 2>
static inline std::ostream&
print_backtrace(std::ostream& os = std::cerr, std::string = {}, std::string = {},
                std::string = {})
{
    os << "[timemory]> Backtrace not supported on this platform\n";
    return os;
}
//
template <size_t Depth, size_t Offset = 3>
static inline std::ostream&
print_demangled_backtrace(std::ostream& os = std::cerr, std::string = {},
                          std::string = {}, std::string = {})
{
    os << "[timemory]> Backtrace not supported on this platform\n";
    return os;
}
//
template <size_t Depth, size_t Offset = 2>
static inline std::ostream&
print_unw_backtrace(std::ostream& os = std::cerr, std::string = {}, std::string = {},
                    std::string = {})
{
    os << "[timemory]> libunwind backtrace not supported on this platform\n";
    return os;
}
//
template <size_t Depth, size_t Offset = 3>
static inline std::ostream&
print_demangled_unw_backtrace(std::ostream& os = std::cerr, std::string = {},
                              std::string = {}, std::string = {})
{
    os << "[timemory]> libunwind backtrace not supported on this platform\n";
    return os;
}
//
#endif
//
}  // namespace tim

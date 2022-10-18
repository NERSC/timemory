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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//

#ifndef TIMEMORY_SIGNALS_SIGNAL_HANDLERS_CPP_
#define TIMEMORY_SIGNALS_SIGNAL_HANDLERS_CPP_
#include "timemory/variadic/macros.hpp"
#endif

#ifndef TIMEMORY_SIGNALS_SIGNAL_HANDLERS_HPP_
#include "timemory/signals/signal_handlers.hpp"
#endif

#if defined(TIMEMORY_SIGNAL_AVAILABLE)

#include "timemory/backends/dmp.hpp"
#include "timemory/backends/process.hpp"
#include "timemory/backends/signals.hpp"
#include "timemory/backends/threading.hpp"
#include "timemory/defines.h"
#include "timemory/log/color.hpp"
#include "timemory/log/logger.hpp"
#include "timemory/utility/backtrace.hpp"
#include "timemory/utility/declaration.hpp"
#include "timemory/utility/macros.hpp"

#include <atomic>
#include <cfenv>
#include <chrono>
#include <csignal>
#include <cstring>
#include <dlfcn.h>
#include <initializer_list>
#include <iostream>
#include <set>
#include <type_traits>

namespace tim
{
namespace signals
{
using term_sigaction_t = struct sigaction;

TIMEMORY_SIGNALS_INLINE
void
termination_signal_handler(int sig, siginfo_t* sinfo, void* /* context */)
{
    static auto _blocked = std::atomic<int>{ 0 };
    {
        // provide some basic synchronization
        auto   _v = _blocked++;
        size_t _n = 0;
        while(_v > 0)
        {
            std::this_thread::yield();
            std::this_thread::sleep_for(std::chrono::milliseconds{ 1 });
            if(_blocked == 0)
                _v = _blocked++;
            if(_n++ >= 1000)
                break;
        }
    }

    signal_settings::disable(static_cast<sys_signal>(sig));
    termination_signal_message(sig, sinfo, std::cerr);
    disable_signal_detection();

#if defined(PSIGINFO_AVAILABLE)
    if(sinfo)
        psiginfo(sinfo, TIMEMORY_PROJECT_NAME " :: ");
#endif

    auto _v = --_blocked;
    {
        // provide some basic synchronization
        size_t _n = 0;
        while(_v > 0)
        {
            std::this_thread::yield();
            std::this_thread::sleep_for(std::chrono::milliseconds{ 1 });
            if(_blocked == 0)
                break;
            if(_n++ >= 1000)
                break;
        }
    }

    if(_v <= 1)
    {
        exit(sig);
        // kill(tim::process::get_id(), sig);
    }
}

TIMEMORY_SIGNALS_INLINE term_sigaction_t&
                        tim_signal_termaction()
{
    static term_sigaction_t _v = {};
    return _v;
}

//--------------------------------------------------------------------------------------//

TIMEMORY_SIGNALS_INLINE term_sigaction_t&
                        tim_signal_oldaction()
{
    static term_sigaction_t _v = {};
    return _v;
}

//--------------------------------------------------------------------------------------//

TIMEMORY_SIGNALS_INLINE
void
termination_signal_message(int sig, siginfo_t* sinfo, std::ostream& os)
{
    struct si_code_info
    {
        int              signum      = -1;
        int              code        = -1;
        std::string_view name        = {};
        std::string_view description = {};
    };

    std::string_view _name = {};
    std::string_view _desc = {};

    if(sinfo)
    {
        for(auto itr : std::initializer_list<si_code_info>{
                { SIGILL, ILL_ILLOPC, "ILL_ILLOPC", "Illegal opcode" },
                { SIGILL, ILL_ILLOPN, "ILL_ILLOPN", "Illegal operand" },
                { SIGILL, ILL_ILLADR, "ILL_ILLADR", "Illegal addressing mode" },
                { SIGILL, ILL_ILLTRP, "ILL_ILLTRP", "Illegal trap" },
                { SIGILL, ILL_PRVOPC, "ILL_PRVOPC", "Privileged opcode" },
                { SIGILL, ILL_PRVREG, "ILL_PRVREG", "Privileged register" },
                { SIGILL, ILL_COPROC, "ILL_COPROC", "Coprocessor error" },
                { SIGILL, ILL_BADSTK, "ILL_BADSTK", "Internal stack error" },
                { SIGFPE, FPE_INTDIV, "FPE_INTDIV", "Integer divide-by-zero" },
                { SIGFPE, FPE_INTOVF, "FPE_INTOVF", "Integer overflow" },
                { SIGFPE, FPE_FLTDIV, "FPE_FLTDIV", "Floating point divide-by-zero" },
                { SIGFPE, FPE_FLTOVF, "FPE_FLTOVF", "Floating point overflow" },
                { SIGFPE, FPE_FLTUND, "FPE_FLTUND", "Floating point underflow" },
                { SIGFPE, FPE_FLTRES, "FPE_FLTRES", "Floating point inexact result" },
                { SIGFPE, FPE_FLTINV, "FPE_FLTINV", "Invalid floating point operation" },
                { SIGFPE, FPE_FLTSUB, "FPE_FLTSUB", "Subscript out of range" },
                { SIGSEGV, SEGV_MAPERR, "SEGV_MAPERR", "Address not mapped" },
                { SIGSEGV, SEGV_ACCERR, "SEGV_ACCERR", "Invalid permissions" },
                { SIGBUS, BUS_ADRALN, "BUS_ADRALN", "Invalid address alignment" },
                { SIGBUS, BUS_ADRERR, "BUS_ADRERR", "Non-existent physical address" },
                { SIGBUS, BUS_OBJERR, "BUS_OBJERR", "Object-specific hardware error" },
                { SIGTRAP, TRAP_BRKPT, "TRAP_BRKPT", "Process breakpoint" },
                { SIGTRAP, TRAP_TRACE, "TRAP_TRACE", "Process trace trap" },
                { SIGCHLD, CLD_EXITED, "CLD_EXITED", "Child exited" },
                { SIGCHLD, CLD_KILLED, "CLD_KILLED",
                  "Child terminated abnormally and did not create a core file" },
                { SIGCHLD, CLD_DUMPED, "CLD_DUMPED",
                  "Child terminated abnormally and created a core file" },
                { SIGCHLD, CLD_TRAPPED, "CLD_TRAPPED", "Traced child trapped" },
                { SIGCHLD, CLD_STOPPED, "CLD_STOPPED", "Child stopped" },
                { SIGCHLD, CLD_CONTINUED, "CLD_CONTINUED", "Stopped child continued" },
                { SIGIO, POLL_IN, "POLL_IN", "Data input available" },
                { SIGIO, POLL_OUT, "POLL_OUT", "Output buffers available" },
                { SIGIO, POLL_MSG, "POLL_MSG", "Input message available" },
                { SIGIO, POLL_ERR, "POLL_ERR", "I/O error" },
                { SIGIO, POLL_PRI, "POLL_PRI", "High priority input available" },
                { SIGIO, POLL_HUP, "POLL_HUP", "Device disconnected" } })
        {
            if(itr.signum == sig && itr.code == sinfo->si_code)
            {
                _name = itr.name;
                _desc = itr.description;
                break;
            }
        }

        if(_name.empty() && _desc.empty())
        {
            for(auto itr : std::initializer_list<si_code_info>{
                    { -1, SI_USER, "SI_USER",
                      "Sent by kill(), pthread_kill(), raise(), abort() or alarm()" },
                    { -1, SI_QUEUE, "SI_QUEUE", "Sent by sigqueue()" },
                    { -1, SI_TIMER, "SI_TIMER",
                      "Generated by expiration of a timer set by timer_settimer()" },
                    { -1, SI_ASYNCIO, "SI_ASYNCIO",
                      "Generated by completion of an asynchronous I/O request" },
                    { -1, SI_MESGQ, "SI_MESGQ",
                      "Generated by arrival of a message on an empty message queue" } })
            {
                if(itr.code == sinfo->si_code)
                {
                    _name = itr.name;
                    _desc = itr.description;
                    break;
                }
            }
        }
    }

    char _label[512];
    memset(_label, '\0', sizeof(_label));
    if(dmp::is_initialized())
    {
        snprintf(_label, sizeof(_label),
                 "### ERROR ### [%s][PID=%i][RANK=%i][TID=%li] signal=%i",
                 TIMEMORY_PROJECT_NAME, process::get_id(), dmp::rank(),
                 threading::get_id(), sig);
    }
    else
    {
        snprintf(_label, sizeof(_label), "### ERROR ### [%s][PID=%i][TID=%li] signal=%i",
                 TIMEMORY_PROJECT_NAME, process::get_id(), threading::get_id(), sig);
    }

    const auto* _src_color   = (&os == &std::cerr) ? log::color::source() : "";
    const auto* _fatal_color = (&os == &std::cerr) ? log::color::fatal() : "";

    (void) _src_color;

    os << "\n";
    auto message = tim::log::stream(os, _fatal_color);
    message << _label;

    {
        std::string _lname = {};
        std::string _ldesc = {};
        std::tie(_lname, std::ignore, _ldesc) =
            signal_settings::get_info(static_cast<sys_signal>(sig));
        if(!_lname.empty())
        {
            message << " (" << _lname << ")";
            if(!_ldesc.empty())
                message << " " << _ldesc;
        }
    }

    if(sinfo && !_name.empty() && !_desc.empty())
        message << ". code: " << sinfo->si_code << " (" << _name << " :: " << _desc
                << ")";
    else
        message << ". code: " << sinfo->si_code;

    if(sinfo)
    {
        switch(sig)
        {
            case SIGILL:
            case SIGFPE:
                message << ", address of failing instruction: " << sinfo->si_addr;
                break;
            case SIGSEGV:
            case SIGBUS:
                message << ", address of faulting memory reference: " << sinfo->si_addr;
                break;
            case SIGIO: message << ", band event: " << sinfo->si_band; break;
            default: break;
        }
    }

    char prefix[512];
    memset(prefix, '\0', 512 * sizeof(char));
    sprintf(prefix, "[PID=%i][TID=%i]", (int) process::get_id(),
            (int) threading::get_id());

    {
        size_t ntot = 0;
        auto   bt   = timemory_get_backtrace<64>();
        for(const auto& itr : bt)
        {
            if(strlen(itr) == 0)
                continue;
            ++ntot;
        }

        message << "\nBacktrace:\n";
        for(size_t i = 0; i < bt.size(); ++i)
        {
            if(strlen(bt.at(i)) == 0)
                continue;
            message << prefix << "[" << i << '/' << ntot << "] " << bt.at(i) << "\n";
        }
    }

    /*
    struct template_data
    {
        std::string      value  = {};
        std::string_view name   = {};
        size_t           pos    = std::string::npos;
        size_t           length = 0;

        template_data(std::string&& _v, std::string_view _name)
        : value{ std::move(_v) }
        , name{ _name }
        {
            pos       = value.find(name);
            size_t _n = 0;
            if(pos != std::string::npos)
            {
                for(size_t i = pos + name.length(); i < value.length(); ++i)
                {
                    if(value.at(i) == '<')
                        ++_n;
                    else if(_n > 0 && value.at(i) == '>')
                        --_n;
                    if(_n == 0)
                    {
                        length = i - pos + 1;
                        break;
                    }
                }
            }
        }

        operator bool() const
        {
            return !name.empty() && pos > 2 && pos < value.length() && length > 0;
        }
    };*/

    auto _replace = [](std::string _v, std::string_view _old, std::string_view _new) {
        // start at 1 to avoid replacing when it starts with string, e.g. do not replace:
        //      std::__cxx11::basic_string<...>::~basic_string
        auto _pos = size_t{ 1 };
        while((_pos = _v.find(_old, _pos)) != std::string::npos)
            _v = _v.replace(_pos, _old.length(), _new);
        return _v;
    };

    bool _do_patch =
        std::getenv(TIMEMORY_SETTINGS_PREFIX "BACKTRACE_DISABLE_PATCH") == nullptr;
    auto _patch_demangled = [_replace, _do_patch](std::string _v) {
        _v =
            (_do_patch)
                ? _replace(
                      _replace(_replace(_replace(std::move(_v), demangle<std::string>(),
                                                 "std::string"),
                                        demangle<std::string_view>(), "std::string_view"),
                               " > >", ">>"),
                      "> >", ">>")
                : _v;
        return _v;
        /*
        auto _data = template_data{ std::move(_v), "std::allocator" };
        do
        {
            _data = template_data{ std::move(_data.value), "std::allocator" };
            if(_data)
                _data.value.replace(_data.pos, _data.length, "ALLOCATOR");
        } while(_data);
        return _data.value;*/
    };

    {
        size_t ntot = 0;
        auto   bt   = get_demangled_native_backtrace<64>();
        for(const auto& itr : bt)
        {
            if(itr.empty())
                continue;
            ++ntot;
        }

        message << "\nBacktrace (demangled):\n";
        for(size_t i = 0; i < bt.size(); ++i)
        {
            if(bt.at(i).empty())
                continue;
            message << prefix << "[" << i << '/' << ntot << "] "
                    << _patch_demangled(bt.at(i)) << "\n";
        }
    }

#if defined(TIMEMORY_USE_LIBUNWIND)
    {
        size_t ntot = 0;
        auto   bt   = get_demangled_unw_backtrace<64>();
        for(const auto& itr : bt)
        {
            if(itr.empty())
                continue;
            ++ntot;
        }

        message << "\nBacktrace (demangled):\n";
        for(size_t i = 0; i < bt.size(); ++i)
        {
            if(bt.at(i).empty())
                continue;
            message << prefix << "[" << i << '/' << ntot << "] "
                    << _patch_demangled(bt.at(i)) << "\n";
        }
    }
#    if defined(TIMEMORY_USE_BFD)
    {
        unwind::set_bfd_verbose(8);
        struct bt_line_info
        {
            bool        first    = false;
            int64_t     index    = 0;
            int64_t     line     = 0;
            std::string name     = {};
            std::string location = {};
        };
        size_t ntot   = 0;
        auto   _bt    = get_unw_stack<64, 2>().get(nullptr, true);
        auto   _lines = std::vector<bt_line_info>{};
        for(const auto& itr : _bt)
        {
            int64_t _idx = ntot++;
            if(itr.lineinfo.found && !itr.lineinfo.lines.empty())
            {
                bool _first = true;
                // make sure the function at the top is the one that
                // is shown in the other backtraces
                auto _line_info = itr.lineinfo.lines;
                std::reverse(_line_info.begin(), _line_info.end());
                for(const auto& litr : _line_info)
                {
                    _lines.emplace_back(bt_line_info{
                        _first, _idx, int64_t{ litr.line },
                        _patch_demangled(demangle(litr.name)), demangle(litr.location) });
                    _first = false;
                }
            }
            else
            {
                _lines.emplace_back(bt_line_info{ true, _idx, int64_t{ 0 },
                                                  _patch_demangled(demangle(itr.name)),
                                                  demangle(itr.location) });
            }
        }

        message << "\nBacktrace (lineinfo):\n";
        for(const auto& itr : _lines)
        {
            auto _get_loc = [&]() -> std::string {
                auto&& _loc = (itr.location.empty()) ? std::string{ "??" } : itr.location;
                return (itr.line == 0) ? TIMEMORY_JOIN(":", _loc, "?")
                                       : TIMEMORY_JOIN(":", _loc, itr.line);
            };

            if(itr.first)
                message << prefix << "[" << itr.index << '/' << ntot << "]\n";

            message << "    " << _src_color << "[" << _get_loc() << "]" << _fatal_color
                    << " " << itr.name << "\n";
        }
    }
#    endif
#endif

    os << "\n" << std::flush;

    try
    {
        signal_settings::exit_action(sig);
    } catch(std::exception& e)
    {
        std::cerr << "signal_settings::exit_action(" << sig << ") threw an exception"
                  << std::endl;
        std::cerr << e.what() << std::endl;
    }
}

//--------------------------------------------------------------------------------------//

TIMEMORY_SIGNALS_INLINE
bool
enable_signal_detection(signal_settings::signal_set_t operations)
{
    if(!signal_settings::allow())
    {
        if(signal_settings::is_active())
            disable_signal_detection();
        return false;
    }

    // don't re-enable
    if(signal_settings::is_active())
        return false;

    if(operations.empty())
    {
        operations = signal_settings::get_enabled();
    }
    else
    {
        auto _enabled = signal_settings::get_enabled();
        if(!_enabled.empty())
        {
            for(const auto& itr : _enabled)
                signal_settings::disable(itr);
        }
        signal_settings::check_environment();
        for(const auto& itr : operations)
            signal_settings::enable(itr);
    }

    std::set<int> _signals;
    for(auto operation : operations)
        _signals.insert(static_cast<int>(operation));

    sigfillset(&tim_signal_termaction().sa_mask);
    for(const auto& itr : _signals)
        sigdelset(&tim_signal_termaction().sa_mask, itr);
    tim_signal_termaction().sa_sigaction = termination_signal_handler;
    tim_signal_termaction().sa_flags     = SA_SIGINFO;
    for(const auto& itr : _signals)
    {
        sigaction(itr, &tim_signal_termaction(), &tim_signal_oldaction());
    }
    signal_settings::set_active(true);

    return true;
}

TIMEMORY_SIGNALS_INLINE void
disable_signal_detection()
{
    // don't re-disable
    if(!signal_settings::is_active())
        return;

    sigemptyset(&tim_signal_termaction().sa_mask);
    tim_signal_termaction().sa_handler = SIG_DFL;

    auto _disable = [](const signal_settings::signal_set_t& _set) {
        for(auto itr : _set)
        {
            int _itr = static_cast<int>(itr);
            sigaction(_itr, &tim_signal_termaction(), nullptr);
        }
    };

    _disable(signal_settings::get_enabled());
    _disable(signal_settings::get_disabled());

    signal_settings::set_active(false);
}

TIMEMORY_SIGNALS_INLINE void
update_signal_detection(const signal_settings::signal_set_t& _signals)
{
    if(signal_settings::allow())
    {
        disable_signal_detection();
        enable_signal_detection(_signals);
    }
}
}  // namespace signals
}  // namespace tim

#endif

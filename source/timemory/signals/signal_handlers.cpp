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
#endif

#include "timemory/defines.h"

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
#include "timemory/signals/signal_settings.hpp"
#include "timemory/unwind/bfd.hpp"
#include "timemory/unwind/processed_entry.hpp"
#include "timemory/utility/backtrace.hpp"
#include "timemory/utility/declaration.hpp"
#include "timemory/utility/demangle.hpp"
#include "timemory/utility/filepath.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/utility/procfs/maps.hpp"
#include "timemory/variadic/macros.hpp"

#include <atomic>
#include <cfenv>
#include <chrono>
#include <csignal>
#include <cstring>
#include <dlfcn.h>
#include <initializer_list>
#include <iostream>
#include <set>
#include <tuple>
#include <type_traits>

namespace tim
{
namespace signals
{
TIMEMORY_SIGNALS_INLINE
void
termination_signal_handler(int _sig, siginfo_t* _sinfo, void* _context)
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

    fprintf(stderr, "\n%s[%s][%i][%li] Signal %i caught : ", log::color::warning(),
            TIMEMORY_PROJECT_NAME, process::get_id(), threading::get_id(), _sig);
#if defined(PSIGINFO_AVAILABLE)
    if(_sinfo)
        psiginfo(_sinfo, "");
    else
        psignal(_sig, "");
#else
    psignal(_sig, "");
#endif
    fprintf(stderr, "%s", log::color::end());

    signal_settings::disable(static_cast<sys_signal>(_sig));
    disable_signal_detection();
    termination_signal_message(_sig, _sinfo, std::cerr);
    signal_settings::exit_action(_sig, _sinfo, _context);

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

    if(_v < 1)
    {
        auto _pid = process::get_id();
        fprintf(stderr, "\n");
        TIMEMORY_PRINTF_WARNING(stderr, "Killing process %i with signal %i...\n", _pid,
                                _sig);
        kill(_pid, _sig);
    }
}

//--------------------------------------------------------------------------------------//

#if defined(TIMEMORY_USE_BFD)

using file_map_t = std::unordered_map<std::string, std::shared_ptr<unwind::bfd_file>>;

TIMEMORY_SIGNALS_INLINE file_map_t&
                        tim_signal_file_maps()
{
    static file_map_t _v = {};
    return _v;
}
#endif

//--------------------------------------------------------------------------------------//

TIMEMORY_SIGNALS_INLINE
void
termination_signal_message(int _sig, siginfo_t* _sinfo, std::ostream& os)
{
    constexpr size_t buffer_size = bt_max_length;

    struct si_code_info
    {
        int              signum      = -1;
        int              code        = -1;
        std::string_view name        = {};
        std::string_view description = {};
    };

    std::string_view _name = {};
    std::string_view _desc = {};

    if(_sinfo)
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
            if(itr.signum == _sig && itr.code == _sinfo->si_code)
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
                if(itr.code == _sinfo->si_code)
                {
                    _name = itr.name;
                    _desc = itr.description;
                    break;
                }
            }
        }
    }

    char _label[buffer_size];
    memset(_label, '\0', sizeof(_label));
    if(dmp::is_initialized())
    {
        snprintf(_label, sizeof(_label),
                 "### ERROR ### [%s][PID=%i][RANK=%i][TID=%li] signal=%i",
                 TIMEMORY_PROJECT_NAME, process::get_id(), dmp::rank(),
                 threading::get_id(), _sig);
    }
    else
    {
        snprintf(_label, sizeof(_label), "### ERROR ### [%s][PID=%i][TID=%li] signal=%i",
                 TIMEMORY_PROJECT_NAME, process::get_id(), threading::get_id(), _sig);
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
            signal_settings::get_info(static_cast<sys_signal>(_sig));
        if(!_lname.empty())
        {
            message << " (" << _lname << ")";
            if(!_ldesc.empty())
                message << " " << _ldesc;
        }
    }

    if(_sinfo && !_name.empty() && !_desc.empty())
        message << ". code: " << _sinfo->si_code << " (" << _name << " :: " << _desc
                << ")";
    else
        message << ". code: " << _sinfo->si_code;

    if(_sinfo)
    {
        switch(_sig)
        {
            case SIGILL:
            case SIGFPE:
                message << ", address of failing instruction: " << _sinfo->si_addr;
                break;
            case SIGSEGV:
            case SIGBUS:
                message << ", address of faulting memory reference: " << _sinfo->si_addr;
                break;
            case SIGIO: message << ", band event: " << _sinfo->si_band; break;
            default: break;
        }
    }

    char prefix[buffer_size];
    memset(prefix, '\0', buffer_size * sizeof(char));
    sprintf(prefix, "[PID=%i][TID=%i]", (int) process::get_id(),
            (int) threading::get_id());

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

    auto _replace = [](std::string _v, const std::string& _old, const std::string& _new) {
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
        message << "\nBacktrace:\n";
        message.flush();

        size_t ntot = 0;
        auto   bt   = timemory_get_backtrace<64, 3>();
        for(const auto& itr : bt)
        {
            auto _len = strnlen(itr, buffer_size);
            if(_len == 0 || _len >= buffer_size)
                continue;
            ++ntot;
        }

        for(size_t i = 0; i < bt.size(); ++i)
        {
            auto* itr  = bt.at(i);
            auto  _len = strnlen(itr, buffer_size);
            if(_len == 0 || _len >= buffer_size)
                continue;
            message << prefix << "[" << i << '/' << ntot << "] " << itr << "\n";
            os << std::flush;
        }
    }

    {
        message << "\nBacktrace (demangled):\n";
        message.flush();

        size_t ntot = 0;
        auto   bt   = get_native_backtrace<64, 3>();
        for(const auto& itr : bt)
        {
            auto _len = strnlen(itr, buffer_size);
            if(_len == 0 || _len >= buffer_size)
                continue;
            ++ntot;
        }

        for(size_t i = 0; i < bt.size(); ++i)
        {
            auto* itr  = bt.at(i);
            auto  _len = strnlen(itr, buffer_size);
            if(_len == 0 || _len >= buffer_size)
                continue;
            message << prefix << "[" << i << '/' << ntot << "] "
                    << _patch_demangled(demangle_native_backtrace(itr)) << "\n";
            os << std::flush;
        }
    }

#if defined(TIMEMORY_USE_LIBUNWIND)
    {
        message << "\nBacktrace (demangled):\n";
        message.flush();

        size_t ntot = 0;
        auto   bt   = get_unw_backtrace<64, 3>();
        for(const auto& itr : bt)
        {
            auto _len = strnlen(itr, buffer_size);
            if(_len == 0 || _len >= buffer_size)
                continue;
            ++ntot;
        }

        for(size_t i = 0; i < bt.size(); ++i)
        {
            auto* itr  = bt.at(i);
            auto  _len = strnlen(itr, buffer_size);
            if(_len == 0 || _len >= buffer_size)
                continue;
            auto _v = std::string{};
            _v.resize(_len);
            for(size_t j = 0; j < _len; ++j)
                _v.at(j) = itr[j];
            message << prefix << "[" << i << '/' << ntot << "] "
                    << _patch_demangled(demangle_backtrace(_v)) << "\n";
            os << std::flush;
        }
    }
#    if defined(TIMEMORY_USE_BFD)
    {
        message << "\nBacktrace (lineinfo):\n";
        message.flush();

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
        auto   _bt    = get_unw_stack<64, 3>();
        auto   _lines = std::vector<bt_line_info>{};
        auto&  _files = tim_signal_file_maps();
        for(const auto& itr : _bt)
        {
            if(!itr)
                continue;
            unwind::processed_entry _entry{};
            _entry.address = itr->address();
            _entry.name = itr->template get_name<1024, false>(_bt.context, &_entry.offset,
                                                              &_entry.error);

            unwind::processed_entry::construct(_entry, &_files, true);
            int64_t _idx = ntot++;
            if(_entry.lineinfo.found && !_entry.lineinfo.lines.empty())
            {
                bool _first = true;
                // make sure the function at the top is the one that
                // is shown in the other backtraces
                auto _line_info = _entry.lineinfo.lines;
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
                                                  _patch_demangled(demangle(_entry.name)),
                                                  demangle(_entry.location) });
            }
        }

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
        signal_settings::exit_action(_sig);
    } catch(std::exception& e)
    {
        std::cerr << "signal_settings::exit_action(" << _sig << ") threw an exception"
                  << std::endl;
        std::cerr << e.what() << std::endl;
    }
}

//--------------------------------------------------------------------------------------//

TIMEMORY_SIGNALS_INLINE
void
update_file_maps()
{
#if defined(TIMEMORY_USE_BFD)
    auto& _maps = procfs::get_maps(process::get_id(), true);
    for(const auto& itr : _maps)
    {
        auto& _files = tim_signal_file_maps();
        auto  _loc   = itr.pathname;
        if(!_loc.empty() && filepath::exists(_loc))
        {
            auto _add_file = [&_files](const auto& _val) {
                if(_files.find(_val) == _files.end())
                {
                    std::stringstream _msg{};
                    _msg << "Reading '" << _val << "'...";
                    unwind::bfd_message(2, _msg.str());
                    _files.emplace(_val, std::make_shared<unwind::bfd_file>(_val));
                    auto _val_real = filepath::realpath(_val);
                    if(_val != _val_real)
                        _files.emplace(_val_real, _files.at(_val));
                }
            };

            _add_file(_loc);
        }
    }
#endif
}

//--------------------------------------------------------------------------------------//

TIMEMORY_SIGNALS_INLINE
bool
enable_signal_detection(signal_settings::signal_set_t             _sys_signals,
                        const signal_settings::signal_function_t& _func)
{
    if(!signal_settings::allow())
    {
        if(signal_settings::is_active())
            disable_signal_detection();
        return false;
    }

    // call now to minimize allocations when delivering signals
    update_file_maps();

    if(_sys_signals.empty())
    {
        _sys_signals = signal_settings::get_enabled();
    }
    else
    {
        for(const auto& itr : _sys_signals)
            signal_settings::enable(itr);
        signal_settings::check_environment();
        for(auto itr : signal_settings::get_disabled())
        {
            if(_sys_signals.count(itr) > 0)
                _sys_signals.erase(itr);
        }
    }

    for(auto itr : _sys_signals)
    {
        auto  _signum = static_cast<int>(itr);
        auto* _entry  = signal_settings::get(itr);

        if(!_entry)
            continue;
        if(_func)
            _entry->functor = _func;
        if(_entry->active)
            continue;

        auto& _action = _entry->current;
        auto& _former = _entry->previous;

        sigfillset(&_action.sa_mask);
        sigdelset(&_action.sa_mask, _signum);
        _action.sa_flags     = SA_SIGINFO | SA_RESTART;
        _action.sa_sigaction = termination_signal_handler;
        sigaction(_signum, &_action, &_former);
        _entry->active = true;
    }

    return true;
}

TIMEMORY_SIGNALS_INLINE void
disable_signal_detection(signal_settings::signal_set_t _sys_signals)
{
    if(_sys_signals.empty())
        _sys_signals = signal_settings::get_active();

    for(auto itr : _sys_signals)
    {
        auto  _signum = static_cast<int>(itr);
        auto* _entry  = signal_settings::get(itr);
        if(!_entry)
            continue;
        if(!_entry->active)
            continue;

        auto _action = signal_settings::sigaction_t{};

        sigemptyset(&_action.sa_mask);
        _action.sa_flags     = 0;
        _action.sa_sigaction = [](int, siginfo_t*, void*) {};
        _action.sa_handler   = SIG_DFL;
        sigaction(_signum, &_action, nullptr);
        _entry->active   = false;
        _entry->current  = {};
        _entry->previous = _action;
    }
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

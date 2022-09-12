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

//======================================================================================//
// This global method should be used on LINUX or MacOSX platforms with gcc,
// clang, or intel compilers for activating signal detection and forcing
// exception being thrown that can be handled when detected.
//======================================================================================//

#pragma once

#include "timemory/backends/dmp.hpp"
#include "timemory/backends/process.hpp"
#include "timemory/backends/signals.hpp"
#include "timemory/backends/threading.hpp"
#include "timemory/log/color.hpp"
#include "timemory/log/logger.hpp"
#include "timemory/utility/backtrace.hpp"
#include "timemory/utility/declaration.hpp"
#include "timemory/utility/macros.hpp"

#include <cfenv>
#include <csignal>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <set>
#include <type_traits>

#if defined(SIGNAL_AVAILABLE)
#    include <dlfcn.h>
#endif

//======================================================================================//
//
// declarations
//
namespace tim
{
inline namespace signals
{
// No    Name         Default Action       Description
// 1     SIGHUP       terminate process    terminal line hangup
// 2     SIGINT       terminate process    interrupt program
// 3     SIGQUIT      create core image    quit program
// 4     SIGILL       create core image    illegal instruction
// 5     SIGTRAP      create core image    trace trap
// 6     SIGABRT      create core image    abort program (formerly SIGIOT)
// 7     SIGEMT       create core image    emulate instruction executed
// 8     SIGFPE       create core image    floating-point exception
// 9     SIGKILL      terminate process    kill program
// 10    SIGBUS       create core image    bus error
// 11    SIGSEGV      create core image    segmentation violation
// 12    SIGSYS       create core image    non-existent system call invoked
// 13    SIGPIPE      terminate process    write on a pipe with no reader
// 14    SIGALRM      terminate process    real-time timer expired
// 15    SIGTERM      terminate process    software termination signal
// 16    SIGURG       discard signal       urgent condition present on socket
// 18    SIGTSTP      stop process         stop signal generated from keyboard
// 24    SIGXCPU      terminate process    cpu time limit exceeded (see
// setrlimit(2)) 25    SIGXFSZ      terminate process    file size limit
// exceeded (see setrlimit(2)) 26    SIGVTALRM    terminate process    virtual
// time alarm (see setitimer(2)) 27    SIGPROF      terminate process profiling
// timer alarm (see setitimer(2))
//
//--------------------------------------------------------------------------------------//
//
enum class sys_signal : int
{
    Hangup       = SIGHUP,   // 1
    Interrupt    = SIGINT,   // 2
    Quit         = SIGQUIT,  // 3
    Illegal      = SIGILL,
    Trap         = SIGTRAP,
    Abort        = SIGABRT,
    Emulate      = SIGEMT,
    FPE          = SIGFPE,
    Kill         = SIGKILL,
    Bus          = SIGBUS,
    SegFault     = SIGSEGV,
    System       = SIGSYS,
    Pipe         = SIGPIPE,
    Alarm        = SIGALRM,
    Terminate    = SIGTERM,
    Urgent       = SIGURG,
    Stop         = SIGTSTP,
    CPUtime      = SIGXCPU,
    FileSize     = SIGXFSZ,
    VirtualAlarm = SIGVTALRM,
    ProfileAlarm = SIGPROF,
    User1        = SIGUSR1,
    User2        = SIGUSR2
};
//
//----------------------------------------------------------------------------------//
//
class signal_settings
{
public:
    using signal_set_t      = std::set<sys_signal>;
    using signal_function_t = std::function<void(int)>;
    using descript_tuple_t  = std::tuple<std::string, int, std::string>;

public:
    static bool&            allow();
    static bool             is_active();
    static void             set_active(bool val);
    static void             enable(const sys_signal&);
    static void             disable(const sys_signal&);
    static std::string      str(const sys_signal&);
    static std::string      str(bool report_disabled = false);
    static void             check_environment();
    static void             set_exit_action(signal_function_t _f);
    static void             exit_action(int errcode);
    static descript_tuple_t get_info(const sys_signal&);

    static signal_set_t get_enabled();
    static signal_set_t get_disabled();
    static signal_set_t get_default();
    static bool&        enable_all();
    static bool&        disable_all();

protected:
    struct signals_data
    {
        signals_data();
        ~signals_data()                   = default;
        signals_data(const signals_data&) = default;
        signals_data(signals_data&&)      = default;
        signals_data& operator=(const signals_data&) = default;
        signals_data& operator=(signals_data&&) = default;

        bool              signals_active    = false;
        bool              enable_all        = false;
        bool              disable_all       = false;
        signal_function_t signals_exit_func = [](int) {};
        signal_set_t      signals_enabled   = {};
        signal_set_t      signals_disabled  = {
            sys_signal::Hangup,       sys_signal::Interrupt, sys_signal::Trap,
            sys_signal::Emulate,      sys_signal::FPE,       sys_signal::Kill,
            sys_signal::System,       sys_signal::Pipe,      sys_signal::Alarm,
            sys_signal::Terminate,    sys_signal::Urgent,    sys_signal::Stop,
            sys_signal::CPUtime,      sys_signal::FileSize,  sys_signal::VirtualAlarm,
            sys_signal::ProfileAlarm, sys_signal::User1,     sys_signal::User2
        };
        // default signals to catch
        signal_set_t signals_default = { sys_signal::Quit, sys_signal::Illegal,
                                         sys_signal::Abort, sys_signal::Bus,
                                         sys_signal::SegFault };
    };

    static signals_data& f_signals()
    {
        static signal_settings::signals_data instance{};
        return instance;
    }
};
//
//--------------------------------------------------------------------------------------//
//
inline bool&
signal_settings::allow()
{
    static bool _instance = true;
    return _instance;
}
//
bool enable_signal_detection(
    signal_settings::signal_set_t = signal_settings::get_default());
//
template <typename Tp,
          enable_if_t<!std::is_enum<Tp>::value && std::is_integral<Tp>::value> = 0>
bool
enable_signal_detection(std::initializer_list<Tp>&&);
//
//--------------------------------------------------------------------------------------//
//
void
disable_signal_detection();
//
//--------------------------------------------------------------------------------------//
//
inline void
update_signal_detection(const signal_settings::signal_set_t& _signals)
{
    if(signal_settings::allow())
    {
        disable_signal_detection();
        enable_signal_detection(_signals);
    }
}
//
//--------------------------------------------------------------------------------------//
//
#if defined(SIGNAL_AVAILABLE)
static void
termination_signal_message(int sig, siginfo_t* sinfo, std::ostream& message);
#endif
//
}  // namespace signals
}  // namespace tim
//
//======================================================================================//
//
#if defined(SIGNAL_AVAILABLE)
//
//--------------------------------------------------------------------------------------//
//
// static void TIMEMORY_ATTRIBUTE(signal)
//    timemory_termination_signal_handler(int sig, siginfo_t* sinfo, void* /* context */);
//
static void
timemory_termination_signal_handler(int sig, siginfo_t* sinfo, void* /* context */)
{
    tim::sys_signal _sig = (tim::sys_signal)(sig);

    if(tim::signal_settings::get_enabled().find(_sig) ==
       tim::signal_settings::get_enabled().end())
    {
        std::stringstream ss;
        ss << "signal " << sig << " not caught";
        TIMEMORY_EXCEPTION(ss.str());
    }

    tim::termination_signal_message(sig, sinfo, std::cerr);

    tim::disable_signal_detection();

    std::stringstream message;
    message << "\n";

#    if defined(PSIGINFO_AVAILABLE)
    if(sinfo)
    {
        psiginfo(sinfo, message.str().c_str());
    }
    else
    {
        std::cerr << message.str() << std::endl;
    }
#    else
    std::cerr << message.str() << std::endl;
#    endif
    exit(sig);
}

//======================================================================================//

namespace tim
{
inline namespace signals
{
//--------------------------------------------------------------------------------------//

static struct sigaction&
tim_signal_termaction()
{
    static struct sigaction _instance;
    return _instance;
}

//--------------------------------------------------------------------------------------//

static struct sigaction&
tim_signal_oldaction()
{
    static struct sigaction _instance;
    return _instance;
}

//--------------------------------------------------------------------------------------//

static void
termination_signal_message(int sig, siginfo_t* sinfo, std::ostream& os)
{
    // std::stringstream message;
    auto&      message = os;
    sys_signal _sig    = (sys_signal)(sig);

    message << "\n### ERROR ### ";
    if(dmp::is_initialized())
        message << " [ rank : " << dmp::rank() << " ] ";
    message << "Error code : " << sig;
    if(sinfo)
        message << " @ " << sinfo->si_addr;
    message << " : " << signal_settings::str(_sig) << ". ";

    if(sig == SIGSEGV)
    {
        if(sinfo)
        {
            switch(sinfo->si_code)
            {
                case SEGV_MAPERR: message << "Address not mapped to object."; break;
                case SEGV_ACCERR:
                    message << "Invalid permissions for mapped object.";
                    break;
                default:
                    message << "Unknown segmentation fault error: " << sinfo->si_code
                            << ".";
                    break;
            }
        }
        else
        {
            message << "Segmentation fault (unknown).";
        }
    }
    else if(sig == SIGFPE)
    {
        if(sinfo)
        {
            switch(sinfo->si_code)
            {
                case FE_DIVBYZERO: message << "Floating point divide by zero."; break;
                case FE_OVERFLOW: message << "Floating point overflow."; break;
                case FE_UNDERFLOW: message << "Floating point underflow."; break;
                case FE_INEXACT: message << "Floating point inexact result."; break;
                case FE_INVALID: message << "Floating point invalid operation."; break;
                default:
                    message << "Unknown floating point exception error: "
                            << sinfo->si_code << ".";
                    break;
            }
        }
        else
        {
            message << "Unknown error.";
        }
    }

    message << std::endl;

    signal_settings::disable(_sig);

    char prefix[512];
    memset(prefix, '\0', 512 * sizeof(char));
    sprintf(prefix, "[PID=%i][TID=%i]", (int) process::get_id(),
            (int) threading::get_id());

#    if defined(TIMEMORY_USE_LIBUNWIND)
    {
        size_t ntot = 0;
        auto   bt   = get_demangled_unw_backtrace<64>();
        for(const auto& itr : bt)
        {
            if(itr.empty())
                continue;
            ++ntot;
        }

        log::stream(message, log::color::fatal()) << "\nBacktrace:\n";
        for(size_t i = 0; i < bt.size(); ++i)
        {
            if(bt.at(i).empty())
                continue;
            log::stream(message, log::color::fatal())
                << prefix << "[" << i << '/' << ntot << "]> " << bt.at(i) << "\n";
        }
    }
#    endif

    {
        size_t ntot = 0;
        auto   bt   = get_demangled_native_backtrace<64>();
        for(const auto& itr : bt)
        {
            if(itr.empty())
                continue;
            ++ntot;
        }

        log::stream(message, log::color::fatal()) << "\nBacktrace:\n";
        for(size_t i = 0; i < bt.size(); ++i)
        {
            if(bt.at(i).empty())
                continue;
            log::stream(message, log::color::fatal())
                << prefix << "[" << i << '/' << ntot << "]> " << bt.at(i) << "\n";
        }
    }
    // message << serr.str().c_str() << std::flush;
    os << std::flush;

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

inline bool
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
    tim_signal_termaction().sa_sigaction = timemory_termination_signal_handler;
    tim_signal_termaction().sa_flags     = SA_SIGINFO;
    for(const auto& itr : _signals)
    {
        sigaction(itr, &tim_signal_termaction(), &tim_signal_oldaction());
    }
    signal_settings::set_active(true);

    // if(settings::verbose() > 0 || settings::debug())
    //    std::cout << signal_settings::str() << std::endl;

    return true;
}

//--------------------------------------------------------------------------------------//

template <typename Tp,
          enable_if_t<!std::is_enum<Tp>::value && std::is_integral<Tp>::value>>
inline bool
enable_signal_detection(std::initializer_list<Tp>&& _signals)
{
    auto operations = signal_settings::signal_set_t{};
    for(const auto& itr : _signals)
        operations.insert(static_cast<sys_signal>(itr));
    return enable_signal_detection(operations);
}

//--------------------------------------------------------------------------------------//

inline void
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

//--------------------------------------------------------------------------------------//
}  // namespace signals
}  // namespace tim

//======================================================================================//

#else /* Not a supported architecture */

//======================================================================================//

namespace tim
{
inline namespace signals
{
inline bool enable_signal_detection(signal_settings::signal_set_t) { return false; }

template <typename Tp,
          enable_if_t<!std::is_enum<Tp>::value && std::is_integral<Tp>::value>>
inline bool
enable_signal_detection(std::initializer_list<Tp>&&)
{
    return false;
}

inline void
disable_signal_detection()
{}
}  // namespace signals
}  // namespace tim

//======================================================================================//

#endif

#include "timemory/utility/bits/signals.hpp"

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

/**
 * \file timemory/utility/declaration.hpp
 * \brief The declaration for the types for utility without definitions
 */

#pragma once

#include "timemory/backends/signals.hpp"

#include <functional>
#include <set>
#include <string>

#include <cfenv>
#include <csignal>

namespace tim
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

public:
    static bool&       allow();
    static bool        is_active();
    static void        set_active(bool val);
    static void        enable(const sys_signal&);
    static void        disable(const sys_signal&);
    static std::string str(const sys_signal&);
    static std::string str(bool report_disabled = false);
    static void        check_environment();
    static void        set_exit_action(signal_function_t _f);
    static void        exit_action(int errcode);

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
//----------------------------------------------------------------------------------//
//
}  // namespace tim

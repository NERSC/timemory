// MIT License
//
// Copyright (c) 2018, The Regents of the University of California, 
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
//

/** \file signal_detection.cpp
 * Handles signals emitted by application
 *
 */

#include "timemory/signal_detection.hpp"
#include <sstream>
#include <string>
#include <cstdlib>

namespace tim
{

//============================================================================//

namespace internal { void dummy_func(int) { return; } }

//============================================================================//

signal_settings::signals_data_t::signals_data_t()
: signals_active(false),
  signals_default({
                  sys_signal::sHangup,
                  sys_signal::sInterrupt,
                  sys_signal::sQuit,
                  sys_signal::sIllegal,
                  sys_signal::sTrap,
                  sys_signal::sAbort,
                  sys_signal::sEmulate,
                  sys_signal::sKill,
                  sys_signal::sBus,
                  sys_signal::sSegFault,
                  sys_signal::sSystem,
                  sys_signal::sPipe,
                  sys_signal::sAlarm,
                  sys_signal::sTerminate,
                  sys_signal::sUrgent,
                  sys_signal::sStop,
                  sys_signal::sCPUtime,
                  sys_signal::sFileSize,
                  sys_signal::sVirtualAlarm,
                  sys_signal::sProfileAlarm,
              }),
  signals_enabled(signals_default),
  signals_disabled(),
  signals_exit_func(internal::dummy_func)
{
#if defined(DEBUG)
    signals_default.insert(sys_signal::sFPE);
    signals_enabled.insert(sys_signal::sFPE);
#else
    signals_disabled.insert(sys_signal::sFPE);
#endif
}

//============================================================================//

signal_settings::signals_data_t signal_settings::f_signals =
        signal_settings::signals_data_t();

//============================================================================//

void
insert_and_remove(const sys_signal& _type,             // signal type
                  signal_settings::signal_set_t* _ins, // set to insert into
                  signal_settings::signal_set_t* _rem) // set to remove from

{
    _ins->insert(_type);
    auto itr = _rem->find(_type);
    if(itr != _rem->end())
        _rem->erase(itr);
}

//============================================================================//

void signal_settings::enable(const sys_signal& _type)
{
    insert_and_remove(_type, &f_signals.signals_enabled, &f_signals.signals_disabled);
}

//============================================================================//

void signal_settings::disable(const sys_signal& _type)
{
    insert_and_remove(_type, &f_signals.signals_disabled, &f_signals.signals_enabled);
}

//============================================================================//

void signal_settings::check_environment()
{
    typedef std::pair<std::string, sys_signal> match_t;

    auto _list =
    {
        match_t("HANGUP",       sys_signal::sHangup),
        match_t("INTERRUPT", 	sys_signal::sInterrupt),
        match_t("QUIT",         sys_signal::sQuit),
        match_t("ILLEGAL",      sys_signal::sIllegal),
        match_t("TRAP",         sys_signal::sTrap),
        match_t("ABORT",        sys_signal::sAbort),
        match_t("EMULATE",      sys_signal::sEmulate),
        match_t("FPE",          sys_signal::sFPE),
        match_t("KILL",         sys_signal::sKill),
        match_t("BUS",          sys_signal::sBus),
        match_t("SEGFAULT", 	sys_signal::sSegFault),
        match_t("SYSTEM",       sys_signal::sSystem),
        match_t("PIPE",         sys_signal::sPipe),
        match_t("ALARM",        sys_signal::sAlarm),
        match_t("TERMINATE", 	sys_signal::sTerminate),
        match_t("URGENT",       sys_signal::sUrgent),
        match_t("STOP",         sys_signal::sStop),
        match_t("CPUTIME",      sys_signal::sCPUtime),
        match_t("FILESIZE", 	sys_signal::sFileSize),
        match_t("VIRTUALALARM", sys_signal::sVirtualAlarm),
        match_t("PROFILEALARM", sys_signal::sProfileAlarm),
    };

    for(auto itr : _list)
    {
        int _enable = get_env<int>("SIGNAL_ENABLE_" + itr.first, 0);
        int _disable = get_env<int>("SIGNAL_DISABLE_" + itr.first, 0);

        if(_enable > 0)
            signal_settings::enable(itr.second);
        if(_disable > 0)
            signal_settings::disable(itr.second);
    }

    int _enable_all = get_env<int>("SIGNAL_ENABLE_ALL", 0);
    if(_enable_all > 0)
        for(const auto& itr : f_signals.signals_disabled)
            signal_settings::enable(itr);

    int _disable_all = get_env<int>("SIGNAL_DISABLE_ALL", 0);
    if(_disable_all > 0)
        for(const auto& itr : f_signals.signals_enabled)
            signal_settings::disable(itr);

}

//============================================================================//

std::string signal_settings::str(const sys_signal& _type)
{
    typedef std::tuple<std::string, int, std::string> descript_tuple_t;

    std::stringstream ss;
    auto descript = [&] (const descript_tuple_t& _data)
    {
        ss << " Signal: " << std::get<0>(_data)
           << " (error code: " << std::get<1>(_data) << ") "
           << std::get<2>(_data);
    };

    // some of these signals are not handled but added in case they are
    // enabled in the future
    static std::vector<descript_tuple_t> descript_data =
    {
        descript_tuple_t("SIGHUP", SIGHUP, "terminal line hangup"),
        descript_tuple_t("SIGINT", SIGINT, "interrupt program"),
        descript_tuple_t("SIGQUIT", SIGQUIT, "quit program"),
        descript_tuple_t("SIGILL", SIGILL, "illegal instruction"),
        descript_tuple_t("SIGTRAP", SIGTRAP, "trace trap"),
        descript_tuple_t("SIGABRT", SIGABRT, "abort program (formerly SIGIOT)"),
        descript_tuple_t("SIGEMT", SIGEMT, "emulate instruction executed"),
        descript_tuple_t("SIGFPE", SIGFPE, "floating-point exception"),
        descript_tuple_t("SIGKILL", SIGKILL, "kill program"),
        descript_tuple_t("SIGBUS", SIGBUS, "bus error"),
        descript_tuple_t("SIGSEGV", SIGSEGV, "segmentation violation"),
        descript_tuple_t("SIGSYS", SIGSYS, "non-existent system call invoked"),
        descript_tuple_t("SIGPIPE", SIGPIPE, "write on a pipe with no reader"),
        descript_tuple_t("SIGALRM", SIGALRM, "real-time timer expired"),
        descript_tuple_t("SIGTERM", SIGTERM, "software termination signal"),
        descript_tuple_t("SIGURG", SIGURG, "urgent condition present on socket"),
        descript_tuple_t("SIGSTOP", SIGSTOP, "stop (cannot be caught or ignored)"),
        descript_tuple_t("SIGTSTP", SIGTSTP, "stop signal generated from keyboard"),
        descript_tuple_t("SIGCONT", SIGCONT, "continue after stop"),
        descript_tuple_t("SIGCHLD", SIGCHLD, "child status has changed"),
        descript_tuple_t("SIGTTIN", SIGTTIN, "background read attempted from control terminal"),
        descript_tuple_t("SIGTTOU", SIGTTOU, "background write attempted to control terminal"),
        descript_tuple_t("SIGIO ", SIGIO, "I/O is possible on a descriptor"),
        descript_tuple_t("SIGXCPU", SIGXCPU, "cpu time limit exceeded"),
        descript_tuple_t("SIGXFSZ", SIGXFSZ, "file size limit exceeded"),
        descript_tuple_t("SIGVTALRM", SIGVTALRM, "virtual time alarm"),
        descript_tuple_t("SIGPROF", SIGPROF, "profiling timer alarm"),
        descript_tuple_t("SIGWINCH", SIGWINCH, "Window size change"),
        descript_tuple_t("SIGINFO", SIGINFO, "status request from keyboard"),
        descript_tuple_t("SIGUSR1", SIGUSR1, "User defined signal 1"),
        descript_tuple_t("SIGUSR2", SIGUSR2, "User defined signal 2")
    };

    int key = (int) _type;
    for(const auto& itr : descript_data)
        if(std::get<1>(itr) == key)
        {
            descript(itr);
            break;
        }

    return ss.str();
}

//============================================================================//

std::string signal_settings::str()
{
    std::stringstream ss;
    auto spacer = [&] () { return "    "; };

#if defined(SIGNAL_AVAILABLE)

    ss << std::endl
       << spacer() << "Signal detection activated. Signal exception settings:\n"
       << std::endl;

    ss << spacer() << "Enabled:" << std::endl;
    for(const auto& itr : f_signals.signals_enabled)
        ss << spacer() << spacer() << signal_settings::str(itr) << std::endl;

    ss << "\n" << spacer() << "Disabled:" << std::endl;
    for(const auto& itr : f_signals.signals_disabled)
        ss << spacer() << spacer() << signal_settings::str(itr) << std::endl;

#else

    ss << std::endl
       << spacer()
       << "Signal detection not available" << std::endl;

#endif

    return ss.str();
}

//============================================================================//

bool signal_settings::is_active()
{
    return f_signals.signals_active;
}

//============================================================================//

void signal_settings::set_active(bool val)
{
    f_signals.signals_active = val;
}

//============================================================================//

void signal_settings::set_exit_action(signal_function_t _f)
{
    f_signals.signals_exit_func = _f;
}

//============================================================================//

void signal_settings::exit_action(int errcode)
{
    f_signals.signals_exit_func(errcode);
}

//============================================================================//

const signal_settings::signal_set_t& signal_settings::enabled()
{
    check_environment();
    return f_signals.signals_enabled;
}

//============================================================================//

const signal_settings::signal_set_t& signal_settings::disabled()
{
    check_environment();
    return f_signals.signals_disabled;
}

//============================================================================//

const signal_settings::signal_set_t& signal_settings::get_enabled()
{
    return f_signals.signals_enabled;
}

//============================================================================//

const signal_settings::signal_set_t& signal_settings::get_disabled()
{
    return f_signals.signals_disabled;
}

//============================================================================//

const signal_settings::signal_set_t& signal_settings::get_default()
{
    return f_signals.signals_default;
}

//============================================================================//

} // namespace tim

//============================================================================//

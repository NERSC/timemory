//  MIT License
//  
//  Copyright (c) 2018, The Regents of the University of California, 
//  through Lawrence Berkeley National Laboratory (subject to receipt of any
//  required approvals from the U.S. Dept. of Energy).  All rights reserved.
//  
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//  
//  The above copyright notice and this permission notice shall be included in all
//  copies or substantial portions of the Software.
//  
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//  SOFTWARE.

#ifndef pytimemory_hpp_
#define pytimemory_hpp_

//============================================================================//

#include <future>
#include <chrono>
#include <functional>
#include <mutex>
#include <thread>
#include <atomic>
#include <iostream>
#include <cstdint>
#include <sstream>
#include <map>
#include <string>
#include <vector>
#include <memory>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>
#include <pybind11/numpy.h>
#include <pybind11/iostream.h>
#include <pybind11/eval.h>
#include <pybind11/embed.h>
#include <pybind11/cast.h>
#include <pybind11/pytypes.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace std::placeholders;  // for _1, _2, _3...
using namespace py::literals;

#include "timemory/macros.hpp"
#include "timemory/manager.hpp"
#include "timemory/timer.hpp"
#include "timemory/rss.hpp"
#include "timemory/auto_timer.hpp"
#include "timemory/signal_detection.hpp"
#include "timemory/rss.hpp"
#include "timemory/mpi.hpp"
#include "timemory/formatters.hpp"

typedef tim::manager                            manager_t;
typedef tim::timer                              tim_timer_t;
typedef tim::auto_timer                         auto_timer_t;
typedef tim::rss::usage                         rss_usage_t;
typedef tim::rss::usage_delta                   rss_delta_t;
typedef tim::sys_signal                         sys_signal_t;
typedef tim::signal_settings                    signal_settings_t;
typedef signal_settings_t::signal_set_t         signal_set_t;
typedef tim::format::timer                      timer_format_t;
typedef tim::format::rss                        rss_format_t;
typedef tim::format::base_formatter::unit_type  unit_type;

typedef py::array_t<double, py::array::c_style | py::array::forcecast> farray_t;

//============================================================================//

class manager_wrapper
{
public:
    manager_wrapper()
    : m_manager(manager_t::instance())
    { }

    ~manager_wrapper()
    { }

    // ensures thread-local version is called
    manager_t* get() { return manager_t::instance(); }

protected:
    manager_t* m_manager;
};

//============================================================================//

class auto_timer_decorator
{
public:
    auto_timer_decorator(auto_timer_t* _ptr = nullptr)
    : m_ptr(_ptr)
    {
        if(m_ptr)
            m_ptr->local_timer().summation_timer()->format()->align_width(true);
    }

    ~auto_timer_decorator()
    {
        delete m_ptr;
    }

    auto_timer_decorator& operator=(auto_timer_t* _ptr)
    {
        if(m_ptr)
            delete m_ptr;
        m_ptr = _ptr;
        m_ptr->local_timer().summation_timer()->format()->align_width(true);
        return *this;
    }

private:
    auto_timer_t* m_ptr;
};

//============================================================================//

#if _PYTHON_MAJOR_VERSION > 2
#   define PYOBJECT_SELF
#   define PYOBJECT_SELF_PARAM
#else
#   define PYOBJECT_SELF py::object,
#   define PYOBJECT_SELF_PARAM py::object
#endif

//============================================================================//

namespace pytim
{

//============================================================================//

typedef std::string string_t;

//============================================================================//
//
//                          TiMemory (general)
//
//============================================================================//

int get_line(int nback = 1)
{
    auto locals = py::dict("back"_a = nback);
    py::exec(R"(
             import sys
             result = int(sys._getframe(back).f_lineno)
             )", py::globals(), locals);
    auto ret = locals["result"].cast<int>();
    return ret;
}

//----------------------------------------------------------------------------//

string_t
get_func(int nback = 1)
{
    auto locals = py::dict("back"_a = nback);
    py::exec(R"(
             import sys
             result = ("{}".format(sys._getframe(back).f_code.co_name))
             )", py::globals(), locals);
    auto ret = locals["result"].cast<std::string>();
    return ret;
}

//----------------------------------------------------------------------------//

string_t
get_file(int nback = 2, bool only_basename = true,
         bool use_dirname = false, bool noquotes = false)
{
    auto locals = py::dict("back"_a = nback,
                           "only_basename"_a = only_basename,
                           "use_dirname"_a = use_dirname,
                           "noquotes"_a = noquotes);
    py::exec(R"(
             import sys
             import os
             from os.path import dirname
             from os.path import basename
             from os.path import join

             def get_fcode(back):
                 fname = '<module'
                 try:
                     fname = sys._getframe(back).f_code.co_filename
                 except:
                     fname = '<module>'
                 return fname

             result = None
             if only_basename:
                 if use_dirname:
                     result = ("{}".format(join(basename(dirname(get_fcode(back))),
                       basename(get_fcode(back)))))
                 else:
                     result = ("{}".format(basename(get_fcode(back))))
             else:
                 result = ("{}".format(get_fcode(back)))

             if noquotes is False:
                 result = ("'{}'".format(result))
             )",
             py::globals(), locals);
      auto ret = locals["result"].cast<std::string>();
      return ret;
}

//----------------------------------------------------------------------------//

signal_set_t
signal_list_to_set(py::list signal_list)
{
    signal_set_t signal_set;
    for(auto itr : signal_list)
        signal_set.insert(itr.cast<sys_signal_t>());
    return signal_set;
}

//----------------------------------------------------------------------------//

signal_set_t
get_default_signal_set()
{
    return tim::signal_settings::enabled();
}

//----------------------------------------------------------------------------//

void
enable_signal_detection(py::list signal_list = py::list())
{
    auto _sig_set = (signal_list.size() == 0)
                    ? get_default_signal_set()
                    : signal_list_to_set(signal_list);
    tim::enable_signal_detection(_sig_set);
}

//----------------------------------------------------------------------------//

void disable_signal_detection()
{
    tim::disable_signal_detection();
}

//----------------------------------------------------------------------------//


//============================================================================//
//
//                              INITITALIZATION
//
//============================================================================//

namespace init
{

//----------------------------------------------------------------------------//

manager_wrapper* manager()
{
    return new manager_wrapper();
}

//----------------------------------------------------------------------------//

tim_timer_t*
timer(std::string prefix = "", std::string format = "")
{
    if(prefix.empty())
    {
        std::stringstream keyss;
        keyss << get_func(1) << "@" << get_file(2) << ":" << get_line(1);
        prefix = keyss.str();
    }

    if(format.empty())
        format = tim::format::timer::default_format();

    return new tim_timer_t(prefix, format);
}

//----------------------------------------------------------------------------//

auto_timer_t*
auto_timer(const std::string& key = "",
           bool report_at_exit = false,
           int nback = 1,
           bool added_args = false)
{
    std::stringstream keyss;
    keyss << get_func(nback);

    if(added_args)
        keyss << key;
    else if(key != "" && key[0] != '@' && !added_args)
        keyss << "@";

    if(key != "" && !added_args)
        keyss << key;
    else
    {
        keyss << "@";
        keyss << get_file(nback+1);
        keyss << ":";
        keyss << get_line(nback);
    }
    auto op_line = get_line();
    return new auto_timer_t(keyss.str(), op_line, "pyc", report_at_exit);
}

//----------------------------------------------------------------------------//

auto_timer_decorator*
timer_decorator(const std::string& func,
                const std::string& file,
                int line,
                const std::string& key,
                bool added_args,
                bool report_at_exit)
{
    auto_timer_decorator* _ptr = new auto_timer_decorator();
    if(!auto_timer_t::alloc_next())
        return _ptr;

    std::stringstream keyss;
    keyss << func;

    // add arguments to end of function
    if(added_args)
        keyss << key;
    else if(key != "" && key[0] != '@' && !added_args)
        keyss << "@";

    if(key != "" && !added_args)
        keyss << key;
    else
    {
        keyss << "@";
        keyss << file;
        keyss << ":";
        keyss << line;
    }
    return &(*_ptr = new auto_timer_t(keyss.str(), line, "pyc", report_at_exit));
}

//----------------------------------------------------------------------------//

rss_usage_t*
rss_usage(std::string prefix = "", bool record = false,
          std::string format = "")
{
    rss_format_t _fmt(prefix);
    if(format.length() > 0)
        _fmt.format(format);
    rss_usage_t* _rss = new rss_usage_t(_fmt);
    if(record)
        _rss->record();
    return _rss;
}

//----------------------------------------------------------------------------//

rss_delta_t*
rss_delta(std::string prefix = "", std::string format = "")
{
    rss_format_t _fmt = tim::format::timer::default_rss_format();
    _fmt.prefix(prefix);
    if(format.length() > 0)
        _fmt.format(format);
    rss_delta_t* _rss = new rss_delta_t(_fmt);
    _rss->init();
    return _rss;
}

//----------------------------------------------------------------------------//

timer_format_t*
timing_format(const std::string& prefix = "",
              const std::string& format = timer_format_t::default_format(),
              unit_type unit = timer_format_t::default_unit(),
              py::object rss_format = py::none(),
              bool align_width = false)
{
    rss_format_t _rss_format = tim::format::timer::default_rss_format();
    if(!rss_format.is_none())
        _rss_format = *rss_format.cast<rss_format_t*>();

    return new timer_format_t(prefix, format, unit, _rss_format, align_width);
}

//----------------------------------------------------------------------------//

rss_format_t*
memory_format(const std::string& prefix = "",
              const std::string& format = rss_format_t::default_format(),
              unit_type unit = rss_format_t::default_unit(),
              bool align_width = false)
{
    return new rss_format_t(prefix, format, unit, align_width);
}

//----------------------------------------------------------------------------//

} // namespace init

//============================================================================//
//
//                              FORMAT
//
//============================================================================//

namespace format
{

//----------------------------------------------------------------------------//

void
set_timer_default(std::string format)
{
    // update C++
    tim::format::timer::default_format(format);
}

//----------------------------------------------------------------------------//

tim::format::timer
get_timer_default()
{
    // in case changed in python, update C++
    return tim::format::timer::default_format();
}

//----------------------------------------------------------------------------//

} // namespace format

//============================================================================//
//
//                              MANAGER
//
//============================================================================//

namespace manager
{

void
report(py::object man,
       bool ign_cutoff = false,
       bool serialize = false,
       std::string serial_filename = "")
{
    auto locals = py::dict();
    py::exec(R"(
             import os
             import timemory.options as options

             repfnm = options.report_filename
             serfnm = options.serial_filename
             do_ret = options.report_file
             do_ser = options.serial_file
             outdir = options.output_dir
             options.ensure_directory_exists('{}/test.txt'.format(outdir))

             # absolute paths
             absdir = os.path.abspath(outdir)

             if outdir in repfnm:
                 repabs = os.path.abspath(repfnm)
             else:
                 repabs = os.path.join(absdir, repfnm)

             if outdir in serfnm:
                 serabs = os.path.abspath(serfnm)
             else:
                 serabs = os.path.join(absdir, serfnm)
             )",
             py::globals(), locals);

    auto outdir = locals["outdir"].cast<std::string>();
    auto repfnm = locals["repfnm"].cast<std::string>();
    auto serfnm = locals["serfnm"].cast<std::string>();
    auto do_rep = locals["do_ret"].cast<bool>();
    auto do_ser = locals["do_ser"].cast<bool>();
    auto repabs = locals["repabs"].cast<std::string>();
    auto serabs = locals["serabs"].cast<std::string>();

    if(repfnm.find(outdir) != 0)
        repfnm = outdir + "/" + repfnm;

    if(serfnm.find(outdir) != 0)
        serfnm = outdir + "/" + serfnm;

    manager_t* _man
            = man.cast<manager_wrapper*>()->get();

    // set the output stream
    if(do_rep)
    {
        std::cout << "Outputting manager to '" << repfnm
                  << "'..." << std::endl;
        _man->set_output_stream(repfnm);

        man.attr("reported_files").cast<py::list>().append(repabs);
    }

    // report ASCII output
    _man->report(ign_cutoff);

    // handle the serialization
    if(!do_ser && serialize)
    {
        do_ser = true;
        if(!serial_filename.empty())
            serfnm = serial_filename;
        else if(serfnm.empty())
            serfnm = "output.json";
    }

    if(do_ser && manager_t::instance()->size() > 0)
    {
        std::cout << "Serializing manager to '" << serfnm
                  << "'..." << std::endl;
        _man->write_serialization(serfnm);
        man.attr("serialized_files").cast<py::list>().append(serabs);
    }
}

//----------------------------------------------------------------------------//

string_t
serialize(py::object man, std::string fname = "")
{
    if(fname.empty())
    {
        auto locals = py::dict();
        py::exec(R"(
                 import timemory.options as options
                 import os

                 result = options.serial_filename
                 if not options.output_dir in result:
                     result = os.path.join(options.output_dir, result)
                 options.ensure_directory_exists(result)
                 )",
                 py::globals(), locals);
        fname = locals["result"].cast<std::string>();
    }
    man.cast<manager_wrapper*>()->get()->write_serialization(fname);
    return fname;
}

//----------------------------------------------------------------------------//

string_t
write_ctest_notes(py::object man, std::string directory, bool append)
{
    py::list filenames = man.attr("reported_files").cast<py::list>();

    std::stringstream ss;
    ss << std::endl;
    ss << "IF(NOT DEFINED CTEST_NOTES_FILES)" << std::endl;
    ss << "    SET(CTEST_NOTES_FILES )" << std::endl;
    ss << "ENDIF(NOT DEFINED CTEST_NOTES_FILES)" << std::endl;
    ss << std::endl;

    // loop over ASCII report filenames
    for(const auto& itr : filenames)
    {
        std::string fname = itr.cast<std::string>();
    #if defined(_WIN32)
        while(fname.find("\\") != std::string::npos)
            fname = fname.replace(fname.find("\\"), 1, "/");
    #endif
        ss << "LIST(APPEND CTEST_NOTES_FILES \"" << fname << "\")" << std::endl;
    }

    ss << std::endl;
    ss << "IF(NOT \"${CTEST_NOTES_FILES}\" STREQUAL \"\")" << std::endl;
    ss << "    LIST(REMOVE_DUPLICATES CTEST_NOTES_FILES)" << std::endl;
    ss << "ENDIF(NOT \"${CTEST_NOTES_FILES}\" STREQUAL \"\")" << std::endl;
    ss << std::endl;

    // create directory (easier in Python)
    auto locals = py::dict("directory"_a = directory);
    py::exec(R"(
             import os

             if not os.path.exists(directory) and directory != '':
                 os.makedirs(directory)

             file_path = os.path.join(directory, "CTestNotes.cmake")
             )",
             py::globals(), locals);

    std::string file_path = locals["file_path"].cast<std::string>();
    std::ofstream outf;
    if(append)
        outf.open(file_path.c_str(), std::ios::app);
    else
        outf.open(file_path.c_str());

    if(outf)
        outf << ss.str();
    outf.close();

    return file_path;
}

//----------------------------------------------------------------------------//

} // namespace manager

//============================================================================//
//
//                          OPTIONS
//
//============================================================================//

namespace opt
{

//----------------------------------------------------------------------------//

void safe_mkdir(string_t directory)
{
    auto locals = py::dict("directory"_a = directory);
    py::exec(R"(
             import os
             if not os.path.exists(directory) and directory != '':
                 os.makedirs(directory)
             )",
             py::globals(), locals);
}

//----------------------------------------------------------------------------//

void ensure_directory_exists(string_t file_path)
{
    auto locals = py::dict("file_path"_a = file_path);
    py::exec(R"(
             import os

             directory = os.path.dirname(file_path)
             if not os.path.exists(directory) and directory != '':
                 os.makedirs(directory)
             )",
             py::globals(), locals);
}

//----------------------------------------------------------------------------//

py::object add_arguments(py::object parser = py::none(),
                         std::string fname = "")
{
    auto locals = py::dict("parser"_a = parser,
                           "fname"_a = fname);
    py::exec(R"(
             import sys
             import os
             from os.path import join
             import argparse
             import timemory

             if parser is None:
                parser = argparse.ArgumentParser()

             # Function to add default output arguments
             def get_file_tag(fname):
                 import os
                 _l = os.path.basename(fname).split('.')
                 if len(_l) > 1:
                     _l.pop()
                 return ("{}".format('_'.join(_l)))

             def_fname = "timing_report"
             if fname != "":
                 def_fname = '_'.join(["timing_report", get_file_tag(fname)])

             parser.add_argument('--output-dir', required=False,
                                 default='.', type=str, help="Output directory")
             parser.add_argument('--filename', required=False,
                                 default=def_fname, type=str,
                                 help="Filename for timing report w/o directory and w/o suffix")
             parser.add_argument('--disable-timers', required=False,
                                 action='store_false',
                                 dest='use_timers',
                                 help="Disable timers for script")
             parser.add_argument('--enable-timers', required=False,
                                 action='store_true',
                                 dest='use_timers', help="Enable timers for script")
             parser.add_argument('--disable-timer-serialization',
                                 required=False, action='store_false',
                                 dest='serial_file',
                                 help="Disable serialization for timers")
             parser.add_argument('--enable-timer-serialization',
                                 required=False, action='store_true',
                                 dest='serial_file',
                                 help="Enable serialization for timers")
             parser.add_argument('--max-timer-depth',
                                 help="Maximum timer depth",
                                 type=int,
                                 default=timemory.options.default_max_depth())
             parser.add_argument('--enable-dart',
                                 help="Print DartMeasurementFile tag for plots",
                                 required=False, action='store_true')
             parser.add_argument('--write-ctest-notes',
                                 help="Write a CTestNotes.cmake file for TiMemory ASCII output",
                                 required=False, action='store_true')

             parser.set_defaults(use_timers=True)
             parser.set_defaults(serial_file=True)
             parser.set_defaults(enable_dart=False)
             parser.set_defaults(write_ctest_notes=False)
             )",
             py::globals(), locals);
    return locals["parser"].cast<py::object>();
}

//----------------------------------------------------------------------------//

void
parse_args(py::object args)
{
    auto locals = py::dict("args"_a = args);
    py::exec(R"(
             import sys
             import os
             import timemory

             # Function to add default output arguments
             timemory.options.serial_file = args.serial_file
             timemory.options.use_timers = args.use_timers
             timemory.options.max_timer_depth = args.max_timer_depth
             timemory.options.output_dir = args.output_dir
             timemory.options.echo_dart = args.enable_dart
             timemory.options.ctest_notes = args.write_ctest_notes

             if args.filename:
                 timemory.options.set_report("{}.{}".format(args.filename, "out"))
                 timemory.options.set_serial("{}.{}".format(args.filename, "json"))

             timemory.toggle(timemory.options.use_timers)
             timemory.set_max_depth(timemory.options.max_timer_depth)
             )",
             py::globals(), locals);
}

//----------------------------------------------------------------------------//

py::object
add_arguments_and_parse(py::object parser = py::none(),
                        std::string fname = "")
{
    auto locals = py::dict("parser"_a = parser,
                           "fname"_a = fname);
    py::exec(R"(
             import timemory

             # Combination of timing.add_arguments and timing.parse_args but returns
             parser = timemory.options.add_arguments(parser, fname)
             args = parser.parse_args()
             timemory.options.parse_args(args)
             )",
             py::globals(), locals);
    return locals["args"].cast<py::object>();
}

//----------------------------------------------------------------------------//

py::object
add_args_and_parse_known(py::object parser = py::none(),
                         std::string fname = "")
{
    auto locals = py::dict("parser"_a = parser,
                           "fname"_a = fname);
    py::exec(R"(
             import timemory

             # Combination of timing.add_arguments and timing.parse_args but returns
             parser = timemory.options.add_arguments(parser, fname)
             args, left = parser.parse_known_args()
             timemory.options.parse_args(args)
             # replace sys.argv with unknown args only
             sys.argv = sys.argv[:1]+left
             )",
             py::globals(), locals);
    return locals["args"].cast<py::object>();

}

//----------------------------------------------------------------------------//

} // namespace opt


//============================================================================//

} // pytim

//============================================================================//

#endif


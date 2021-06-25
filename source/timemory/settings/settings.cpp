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

#ifndef TIMEMORY_SETTINGS_SETTINGS_CPP_
#    define TIMEMORY_SETTINGS_SETTINGS_CPP_

#    include "timemory/settings/settings.hpp"
#    include "timemory/backends/dmp.hpp"
#    include "timemory/defines.h"
#    include "timemory/mpl/policy.hpp"
#    include "timemory/settings/macros.hpp"
#    include "timemory/settings/types.hpp"
#    include "timemory/tpls/cereal/archives.hpp"
#    include "timemory/utility/argparse.hpp"
#    include "timemory/utility/bits/signals.hpp"
#    include "timemory/utility/declaration.hpp"
#    include "timemory/utility/filepath.hpp"
#    include "timemory/utility/utility.hpp"
#    include "timemory/variadic/macros.hpp"

#    include <fstream>

namespace tim
{
//
//--------------------------------------------------------------------------------------//
//
template <typename Archive>
void
settings::serialize_settings(Archive& ar)
{
    if(settings::instance())
        ar(cereal::make_nvp("settings", *settings::instance()));
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Archive>
void
settings::serialize_settings(Archive& ar, settings& _obj)
{
    ar(cereal::make_nvp("settings", _obj));
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE
std::shared_ptr<settings>
settings::shared_instance()
{
    // do not take reference to ensure push/pop w/o template parameters do not change
    // the settings
    static auto _instance = shared_instance<TIMEMORY_API>();
    return _instance;
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE
settings*
settings::instance()
{
    // do not take reference to ensure push/pop w/o template parameters do not change
    // the settings
    static auto _instance = shared_instance();
    return _instance.get();
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE
settings::strvector_t&
settings::command_line()
{
    return instance()->get_environment();
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE
settings::strvector_t&
settings::environment()
{
    return instance()->get_environment();
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE
settings::strvector_t
settings::get_global_environment()
{
#    if defined(TIMEMORY_UNIX)
    strvector_t _environ;
    if(environ != nullptr)
    {
        int idx = 0;
        while(environ[idx] != nullptr)
            _environ.push_back(environ[idx++]);
    }
    return _environ;
#    else
    return std::vector<std::string>();
#    endif
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE
std::string
get_local_datetime(const char* dt_format, std::time_t* dt_curr)
{
    char mbstr[100];
    if(!dt_curr)
        dt_curr = settings::get_launch_time(TIMEMORY_API{});

    if(std::strftime(mbstr, sizeof(mbstr), dt_format, std::localtime(dt_curr)))
        return std::string{ mbstr };
    return std::string{};
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE
std::string
settings::tolower(std::string str)
{
    for(auto& itr : str)
        itr = ::tolower(itr);
    return str;
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE
std::string
settings::toupper(std::string str)
{
    for(auto& itr : str)
        itr = ::toupper(itr);
    return str;
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE
std::string
settings::get_global_input_prefix()
{
    auto _dir    = input_path();
    auto _prefix = input_prefix();

    return filepath::osrepr(_dir + std::string("/") + _prefix);
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE
std::string
settings::get_global_output_prefix(bool fake)
{
    static auto _settings = instance();

    auto _dir = (_settings)
                    ? _settings->get_output_path()
                    : get_env<std::string>(TIMEMORY_SETTINGS_KEY("OUTPUT_PATH"), ".");
    auto _prefix = (_settings)
                       ? _settings->get_output_prefix()
                       : get_env<std::string>(TIMEMORY_SETTINGS_KEY("OUTPUT_PREFIX"), "");
    auto _time_output = (_settings)
                            ? _settings->get_time_output()
                            : get_env<bool>(TIMEMORY_SETTINGS_KEY("TIME_OUTPUT"), false);
    auto _time_format =
        (_settings)
            ? _settings->get_time_format()
            : get_env<std::string>(TIMEMORY_SETTINGS_KEY("TIME_FORMAT"), "%F_%I.%M_%p");

    if(_time_output)
    {
        // get the statically stored launch time
        auto* _launch_time    = get_launch_time(TIMEMORY_API{});
        auto  _local_datetime = get_local_datetime(_time_format.c_str(), _launch_time);
        if(_dir.find(_local_datetime) == std::string::npos)
        {
            if(_dir.length() > 0 && _dir[_dir.length() - 1] != '/')
                _dir += "/";
            _dir += _local_datetime;
        }
    }

    // always return zero if fake. if makedir failed, don't prefix with directory
    auto ret = (fake) ? 0 : makedir(_dir);
    return (ret == 0) ? filepath::osrepr(_dir + std::string("/") + _prefix)
                      : filepath::osrepr(std::string("./") + _prefix);
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE
void
settings::store_command_line(int argc, char** argv)
{
    auto& _cmdline = command_line();
    _cmdline.clear();
    for(int i = 0; i < argc; ++i)
        _cmdline.push_back(std::string(argv[i]));
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE
std::string
settings::compose_output_filename(const std::string& _tag, std::string _ext,
                                  bool _mpi_init, const int32_t _mpi_rank, bool fake,
                                  std::string _explicit)
{
    // if there isn't an explicit prefix, get the <OUTPUT_PATH>/<OUTPUT_PREFIX>
    auto _prefix = (!_explicit.empty()) ? _explicit : get_global_output_prefix(fake);

    // return on empty
    if(_prefix.empty())
        return "";

    auto only_ascii = [](char c) { return !isascii(c); };

    _prefix.erase(std::remove_if(_prefix.begin(), _prefix.end(), only_ascii),
                  _prefix.end());

    // if explicit prefix is provided, then make the directory
    if(!_explicit.empty())
    {
        auto ret = makedir(_prefix);
        if(ret != 0)
            _prefix = filepath::osrepr(std::string("./"));
    }

    // add the mpi rank if not root
    auto _rank_suffix = (_mpi_init && _mpi_rank >= 0)
                            ? (std::string("_") + std::to_string(_mpi_rank))
                            : std::string("");

    // add period before extension
    if(_ext.find('.') != 0)
        _ext = std::string(".") + _ext;
    auto plast = static_cast<intmax_t>(_prefix.length()) - 1;
    // add dash if not empty, not ends in '/', and last char is alphanumeric
    if(!_prefix.empty() && _prefix[plast] != '/' && isalnum(_prefix[plast]))
        _prefix += "-";
    // create the path
    std::string fpath  = _prefix + _tag + _rank_suffix + _ext;
    using strpairvec_t = std::vector<std::pair<std::string, std::string>>;
    for(auto&& itr : strpairvec_t{ { "--", "-" }, { "__", "_" }, { "//", "/" } })
    {
        auto pos = std::string::npos;
        while((pos = fpath.find(itr.first)) != std::string::npos)
            fpath.replace(pos, itr.first.length(), itr.second);
    }
    return filepath::osrepr(fpath);
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE
std::string
settings::compose_input_filename(const std::string& _tag, std::string _ext,
                                 bool _mpi_init, const int32_t _mpi_rank,
                                 std::string _explicit)
{
    if(settings::input_path().empty())
        settings::input_path() = settings::output_path();

    if(settings::input_prefix().empty())
        settings::input_prefix() = settings::output_prefix();

    auto _prefix = (_explicit.length() > 0) ? _explicit : get_global_input_prefix();

    auto only_ascii = [](char c) { return !isascii(c); };

    _prefix.erase(std::remove_if(_prefix.begin(), _prefix.end(), only_ascii),
                  _prefix.end());

    if(_explicit.length() > 0)
        _prefix = filepath::osrepr(std::string("./"));

    auto _rank_suffix = (_mpi_init && _mpi_rank >= 0)
                            ? (std::string("_") + std::to_string(_mpi_rank))
                            : std::string("");
    if(_ext.find('.') != 0)
        _ext = std::string(".") + _ext;
    auto plast = _prefix.length() - 1;
    if(_prefix.length() > 0 && _prefix[plast] != '/' && isalnum(_prefix[plast]))
        _prefix += "_";
    auto fpath = utility::path(_prefix + _tag + _rank_suffix + _ext);
    while(fpath.find("//") != std::string::npos)
        fpath.replace(fpath.find("//"), 2, "/");
    return std::move(fpath);
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE
void
settings::parse(std::shared_ptr<settings> _settings)
{
    if(_settings)
        parse(_settings.get());
}
//
//--------------------------------------------------------------------------------------//
//
// function to parse the environment for settings
//
// Nearly all variables will parse env when first access but this allows provides a
// way to reparse the environment so that default settings (possibly from previous
// invocation) can be overwritten
//
TIMEMORY_SETTINGS_INLINE
void
settings::parse(settings* _settings)
{
    if(!_settings)
    {
        PRINT_HERE("%s", "nullptr to tim::settings");
        return;
    }

    if(_settings->get_suppress_parsing())
    {
        static auto _once = false;
        if(!_once)
        {
            PRINT_HERE("%s", "settings parsing has been suppressed");
            _once = true;
        }
        return;
    }

    for(const auto& itr : *_settings)
    {
        itr.second->parse();
    }
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE
settings::settings()
: m_data(data_type{})
{
    // PRINT_HERE("%s", "");
    initialize();
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE
settings::settings(const settings& rhs)
: m_data(data_type{})
, m_order(rhs.m_order)
, m_command_line(rhs.m_command_line)
, m_environment(rhs.m_environment)
{
    for(auto& itr : rhs.m_data)
        m_data.emplace(itr.first, itr.second->clone());
    for(auto& itr : m_order)
    {
        if(m_data.find(itr) == m_data.end())
        {
            auto ritr = rhs.m_data.find(itr);
            if(ritr == rhs.m_data.end())
            {
                TIMEMORY_EXCEPTION(string_t("Error! Missing ordered entry: ") + itr)
            }
            else
            {
                m_data.emplace(itr, ritr->second->clone());
            }
        }
    }
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE
settings&
settings::operator=(const settings& rhs)
{
    // PRINT_HERE("%s", "");
    if(this == &rhs)
        return *this;

    for(auto& itr : rhs.m_data)
        m_data[itr.first] = itr.second->clone();
    m_order        = rhs.m_order;
    m_command_line = rhs.m_command_line;
    m_environment  = rhs.m_environment;
    for(auto& itr : m_order)
    {
        if(m_data.find(itr) == m_data.end())
        {
            auto ritr = rhs.m_data.find(itr);
            if(ritr == rhs.m_data.end())
            {
                TIMEMORY_EXCEPTION(string_t("Error! Missing ordered entry: ") + itr)
            }
            else
            {
                m_data.emplace(itr, ritr->second->clone());
            }
        }
    }
    return *this;
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE
void
settings::initialize_core()
{
    // PRINT_HERE("%s", "");
    auto homedir = get_env<string_t>("HOME");

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        string_t, config_file, TIMEMORY_SETTINGS_KEY("CONFIG_FILE"),
        "Configuration file for timemory",
        TIMEMORY_JOIN(';', TIMEMORY_JOIN('/', homedir, ".timemory.cfg"),
                      TIMEMORY_JOIN('/', homedir, ".timemory.json"),
                      TIMEMORY_JOIN('/', homedir, ".config", "timemory.cfg"),
                      TIMEMORY_JOIN('/', homedir, ".config", "timemory.json")),
        strvector_t({ "-C", "--timemory-config" }));

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        bool, suppress_config, TIMEMORY_SETTINGS_KEY("SUPPRESS_CONFIG"),
        "Disable processing of setting configuration files", false,
        strvector_t({ "--timemory-suppress-config", "--timemory-no-config" }));

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        bool, suppress_parsing, TIMEMORY_SETTINGS_KEY("SUPPRESS_PARSING"),
        "Disable parsing environment", false,
        strvector_t({ "--timemory-suppress-parsing" }), -1, 1);

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        bool, enabled, TIMEMORY_SETTINGS_KEY("ENABLED"), "Activation state of timemory",
        TIMEMORY_DEFAULT_ENABLED, strvector_t({ "--timemory-enabled" }), -1, 1);

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(int, verbose, TIMEMORY_SETTINGS_KEY("VERBOSE"),
                                      "Verbosity level", 0,
                                      strvector_t({ "--timemory-verbose" }), 1);

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(bool, debug, TIMEMORY_SETTINGS_KEY("DEBUG"),
                                      "Enable debug output", false,
                                      strvector_t({ "--timemory-debug" }), -1, 1);

    TIMEMORY_SETTINGS_REFERENCE_ARG_IMPL(
        bool, flat_profile, TIMEMORY_SETTINGS_KEY("FLAT_PROFILE"),
        "Set the label hierarchy mode to default to flat",
        scope::get_fields()[scope::flat::value],
        strvector_t({ "--timemory-flat-profile" }), -1, 1);

    TIMEMORY_SETTINGS_REFERENCE_ARG_IMPL(
        bool, timeline_profile, TIMEMORY_SETTINGS_KEY("TIMELINE_PROFILE"),
        "Set the label hierarchy mode to default to timeline",
        scope::get_fields()[scope::timeline::value],
        strvector_t({ "--timemory-timeline-profile" }), -1, 1);

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        uint16_t, max_depth, TIMEMORY_SETTINGS_KEY("MAX_DEPTH"),
        "Set the maximum depth of label hierarchy reporting",
        std::numeric_limits<uint16_t>::max(), strvector_t({ "--timemory-max-depth" }), 1);
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE
void
settings::initialize_components()
{
    // PRINT_HERE("%s", "");
    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        string_t, global_components, TIMEMORY_SETTINGS_KEY("GLOBAL_COMPONENTS"),
        "A specification of components which is used by multiple variadic bundlers and "
        "user_bundles as the fall-back set of components if their specific variable is "
        "not set. E.g. user_mpip_bundle will use this if TIMEMORY_MPIP_COMPONENTS is not "
        "specified",
        "", strvector_t({ "--timemory-global-components" }));

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        string_t, ompt_components, TIMEMORY_SETTINGS_KEY("OMPT_COMPONENTS"),
        "A specification of components which will be added "
        "to structures containing the 'user_ompt_bundle'. Priority: TRACE_COMPONENTS -> "
        "PROFILER_COMPONENTS -> COMPONENTS -> GLOBAL_COMPONENTS",
        "", strvector_t({ "--timemory-ompt-components" }));

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        string_t, mpip_components, TIMEMORY_SETTINGS_KEY("MPIP_COMPONENTS"),
        "A specification of components which will be added "
        "to structures containing the 'user_mpip_bundle'. Priority: TRACE_COMPONENTS -> "
        "PROFILER_COMPONENTS -> COMPONENTS -> GLOBAL_COMPONENTS",
        "", strvector_t({ "--timemory-mpip-components" }));

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        string_t, ncclp_components, TIMEMORY_SETTINGS_KEY("NCCLP_COMPONENTS"),
        "A specification of components which will be added "
        "to structures containing the 'user_ncclp_bundle'. Priority: MPIP_COMPONENTS -> "
        "TRACE_COMPONENTS -> PROFILER_COMPONENTS -> COMPONENTS -> GLOBAL_COMPONENTS",
        "", strvector_t({ "--timemory-ncclp-components" }));

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        string_t, trace_components, TIMEMORY_SETTINGS_KEY("TRACE_COMPONENTS"),
        "A specification of components which will be used by the interfaces which are "
        "designed for full profiling. These components will be subjected to throttling. "
        "Priority: COMPONENTS -> GLOBAL_COMPONENTS",
        "", strvector_t({ "--timemory-trace-components" }));

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        string_t, profiler_components, TIMEMORY_SETTINGS_KEY("PROFILER_COMPONENTS"),
        "A specification of components which will be used by the interfaces which are "
        "designed for full python profiling. This specification will be overridden by a "
        "trace_components specification. Priority: COMPONENTS -> GLOBAL_COMPONENTS",
        "", strvector_t({ "--timemory-profiler-components" }));

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        string_t, kokkos_components, TIMEMORY_SETTINGS_KEY("KOKKOS_COMPONENTS"),
        "A specification of components which will be used by the interfaces which are "
        "designed for kokkos profiling. Priority: TRACE_COMPONENTS -> "
        "PROFILER_COMPONENTS -> COMPONENTS -> GLOBAL_COMPONENTS",
        "", strvector_t({ "--timemory-kokkos-components" }));

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        string_t, components, TIMEMORY_SETTINGS_KEY("COMPONENTS"),
        "A specification of components which is used by the library interface. This "
        "falls back to TIMEMORY_GLOBAL_COMPONENTS.",
        "", strvector_t({ "--timemory-components" }));
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE
void
settings::initialize_io()
{
    // PRINT_HERE("%s", "");
    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(bool, auto_output,
                                      TIMEMORY_SETTINGS_KEY("AUTO_OUTPUT"),
                                      "Generate output at application termination", true,
                                      strvector_t({ "--timemory-auto-output" }), -1, 1);

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        bool, cout_output, TIMEMORY_SETTINGS_KEY("COUT_OUTPUT"), "Write output to stdout",
        true, strvector_t({ "--timemory-cout-output" }), -1, 1);

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        bool, file_output, TIMEMORY_SETTINGS_KEY("FILE_OUTPUT"), "Write output to files",
        true, strvector_t({ "--timemory-file-output" }), -1, 1);

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(bool, text_output,
                                      TIMEMORY_SETTINGS_KEY("TEXT_OUTPUT"),
                                      "Write text output files", true,
                                      strvector_t({ "--timemory-text-output" }), -1, 1);

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(bool, json_output,
                                      TIMEMORY_SETTINGS_KEY("JSON_OUTPUT"),
                                      "Write json output files", true,
                                      strvector_t({ "--timemory-json-output" }), -1, 1);

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(bool, tree_output,
                                      TIMEMORY_SETTINGS_KEY("TREE_OUTPUT"),
                                      "Write hierarchical json output files", true,
                                      strvector_t({ "--timemory-tree-output" }), -1, 1);

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(bool, dart_output,
                                      TIMEMORY_SETTINGS_KEY("DART_OUTPUT"),
                                      "Write dart measurements for CDash", false,
                                      strvector_t({ "--timemory-dart-output" }), -1, 1);

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        bool, time_output, TIMEMORY_SETTINGS_KEY("TIME_OUTPUT"),
        "Output data to subfolder w/ a timestamp (see also: TIMEMORY_TIME_FORMAT)", false,
        strvector_t({ "--timemory-time-output" }), -1, 1);

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        bool, plot_output, TIMEMORY_SETTINGS_KEY("PLOT_OUTPUT"),
        "Generate plot outputs from json outputs", TIMEMORY_DEFAULT_PLOTTING,
        strvector_t({ "--timemory-plot-output" }), -1, 1);

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        bool, diff_output, TIMEMORY_SETTINGS_KEY("DIFF_OUTPUT"),
        "Generate a difference output vs. a pre-existing output (see also: "
        "TIMEMORY_INPUT_PATH and TIMEMORY_INPUT_PREFIX)",
        false, strvector_t({ "--timemory-diff-output" }), -1, 1);

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        bool, flamegraph_output, TIMEMORY_SETTINGS_KEY("FLAMEGRAPH_OUTPUT"),
        "Write a json output for flamegraph visualization (use chrome://tracing)", true,
        strvector_t({ "--timemory-flamegraph-output" }), -1, 1);

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        bool, ctest_notes, TIMEMORY_SETTINGS_KEY("CTEST_NOTES"),
        "Write a CTestNotes.txt for each text output", false,
        strvector_t({ "--timemory-ctest-notes" }), -1, 1);

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        string_t, output_path, TIMEMORY_SETTINGS_KEY("OUTPUT_PATH"),
        "Explicitly specify the output folder for results", "timemory-output",
        strvector_t({ "--timemory-output-path" }), 1);  // folder

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(string_t, output_prefix,
                                      TIMEMORY_SETTINGS_KEY("OUTPUT_PREFIX"),
                                      "Explicitly specify a prefix for all output files",
                                      "", strvector_t({ "--timemory-output-prefix" }),
                                      1);  // file prefix

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        string_t, input_path, TIMEMORY_SETTINGS_KEY("INPUT_PATH"),
        "Explicitly specify the input folder for difference "
        "comparisons (see also: TIMEMORY_DIFF_OUTPUT)",
        "", strvector_t({ "--timemory-input-path" }), 1);  // folder

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        string_t, input_prefix, TIMEMORY_SETTINGS_KEY("INPUT_PREFIX"),
        "Explicitly specify the prefix for input files used in difference comparisons "
        "(see also: TIMEMORY_DIFF_OUTPUT)",
        "", strvector_t({ "--timemory-input-prefix" }), 1);  // file prefix

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        string_t, input_extensions, TIMEMORY_SETTINGS_KEY("INPUT_EXTENSIONS"),
        "File extensions used when searching for input files used in difference "
        "comparisons (see also: TIMEMORY_DIFF_OUTPUT)",
        "json,xml", strvector_t({ "--timemory-input-extensions" }));  // extensions
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE
void
settings::initialize_format()
{
    // PRINT_HERE("%s", "");
    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        string_t, time_format, TIMEMORY_SETTINGS_KEY("TIME_FORMAT"),
        "Customize the folder generation when TIMEMORY_TIME_OUTPUT is enabled (see also: "
        "strftime)",
        "%F_%I.%M_%p", strvector_t({ "--timemory-time-format" }), 1);

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(int16_t, precision,
                                      TIMEMORY_SETTINGS_KEY("PRECISION"),
                                      "Set the global output precision for components",
                                      -1, strvector_t({ "--timemory-precision" }), 1);

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(int16_t, width, TIMEMORY_SETTINGS_KEY("WIDTH"),
                                      "Set the global output width for components", -1,
                                      strvector_t({ "--timemory-width" }), 1);

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(int32_t, max_width,
                                      TIMEMORY_SETTINGS_KEY("MAX_WIDTH"),
                                      "Set the maximum width for component label outputs",
                                      120, strvector_t({ "--timemory-max-width" }), 1);

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        bool, scientific, TIMEMORY_SETTINGS_KEY("SCIENTIFIC"),
        "Set the global numerical reporting to scientific format", false,
        strvector_t({ "--timemory-scientific" }), -1, 1);

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        int16_t, timing_precision, TIMEMORY_SETTINGS_KEY("TIMING_PRECISION"),
        "Set the precision for components with 'is_timing_category' type-trait", -1);

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        int16_t, timing_width, TIMEMORY_SETTINGS_KEY("TIMING_WIDTH"),
        "Set the output width for components with 'is_timing_category' type-trait", -1);

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        string_t, timing_units, TIMEMORY_SETTINGS_KEY("TIMING_UNITS"),
        "Set the units for components with 'uses_timing_units' type-trait", "",
        strvector_t({ "--timemory-timing-units" }), 1);

    TIMEMORY_SETTINGS_MEMBER_IMPL(bool, timing_scientific,
                                  TIMEMORY_SETTINGS_KEY("TIMING_SCIENTIFIC"),
                                  "Set the numerical reporting format for components "
                                  "with 'is_timing_category' type-trait",
                                  false);

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        int16_t, memory_precision, TIMEMORY_SETTINGS_KEY("MEMORY_PRECISION"),
        "Set the precision for components with 'is_memory_category' type-trait", -1);

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        int16_t, memory_width, TIMEMORY_SETTINGS_KEY("MEMORY_WIDTH"),
        "Set the output width for components with 'is_memory_category' type-trait", -1);

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        string_t, memory_units, TIMEMORY_SETTINGS_KEY("MEMORY_UNITS"),
        "Set the units for components with 'uses_memory_units' type-trait", "",
        strvector_t({ "--timemory-memory-units" }), 1);

    TIMEMORY_SETTINGS_MEMBER_IMPL(bool, memory_scientific,
                                  TIMEMORY_SETTINGS_KEY("MEMORY_SCIENTIFIC"),
                                  "Set the numerical reporting format for components "
                                  "with 'is_memory_category' type-trait",
                                  false);

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        int64_t, separator_frequency, TIMEMORY_SETTINGS_KEY("SEPARATOR_FREQ"),
        "Frequency of dashed separator lines in text output", 0);
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE
void
settings::initialize_parallel()
{
    // PRINT_HERE("%s", "");
    TIMEMORY_SETTINGS_MEMBER_IMPL(
        size_t, max_thread_bookmarks, TIMEMORY_SETTINGS_KEY("MAX_THREAD_BOOKMARKS"),
        "Maximum number of times a worker thread bookmarks the call-graph location w.r.t."
        " the master thread. Higher values tend to increase the finalization merge time",
        50);

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        bool, collapse_threads, TIMEMORY_SETTINGS_KEY("COLLAPSE_THREADS"),
        "Enable/disable combining thread-specific data", true,
        strvector_t({ "--timemory-collapse-threads" }), -1, 1);

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        bool, collapse_processes, TIMEMORY_SETTINGS_KEY("COLLAPSE_PROCESSES"),
        "Enable/disable combining process-specific data", true,
        strvector_t({ "--timemory-collapse-processes" }), -1, 1);

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        bool, cpu_affinity, TIMEMORY_SETTINGS_KEY("CPU_AFFINITY"),
        "Enable pinning threads to CPUs (Linux-only)", false,
        strvector_t({ "--timemory-cpu-affinity" }), -1, 1);

    TIMEMORY_SETTINGS_REFERENCE_IMPL(
        process::id_t, target_pid, TIMEMORY_SETTINGS_KEY("TARGET_PID"),
        "Process ID for the components which require this", process::get_target_id());

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        bool, mpi_init, TIMEMORY_SETTINGS_KEY("MPI_INIT"),
        "Enable/disable timemory calling MPI_Init / MPI_Init_thread during certain "
        "timemory_init(...) invocations",
        false, strvector_t({ "--timemory-mpi-init" }), -1, 1);

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        bool, mpi_finalize, TIMEMORY_SETTINGS_KEY("MPI_FINALIZE"),
        "Enable/disable timemory calling MPI_Finalize during "
        "timemory_finalize(...) invocations",
        false, strvector_t({ "--timemory-mpi-finalize" }), -1, 1);

    TIMEMORY_SETTINGS_REFERENCE_ARG_IMPL(
        bool, mpi_thread, TIMEMORY_SETTINGS_KEY("MPI_THREAD"),
        "Call MPI_Init_thread instead of MPI_Init (see also: TIMEMORY_MPI_INIT)",
        mpi::use_mpi_thread(), strvector_t({ "--timemory-mpi-thread" }), -1, 1);

    TIMEMORY_SETTINGS_REFERENCE_ARG_IMPL(
        string_t, mpi_thread_type, TIMEMORY_SETTINGS_KEY("MPI_THREAD_TYPE"),
        "MPI_Init_thread mode: 'single', 'serialized', "
        "'funneled', or 'multiple' (see also: "
        "TIMEMORY_MPI_INIT and TIMEMORY_MPI_THREAD)",
        mpi::use_mpi_thread_type(), strvector_t({ "--timemory-mpi-thread-type" }), 1);

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        bool, upcxx_init, TIMEMORY_SETTINGS_KEY("UPCXX_INIT"),
        "Enable/disable timemory calling upcxx::init() during certain "
        "timemory_init(...) invocations",
        false, strvector_t({ "--timemory-upcxx-init" }), -1, 1);

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        bool, upcxx_finalize, TIMEMORY_SETTINGS_KEY("UPCXX_FINALIZE"),
        "Enable/disable timemory calling upcxx::finalize() during "
        "timemory_finalize()",
        false, strvector_t({ "--timemory-upcxx-finalize" }), -1, 1);

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        int32_t, node_count, TIMEMORY_SETTINGS_KEY("NODE_COUNT"),
        "Total number of nodes used in application. Setting this value > 1 will result "
        "in aggregating N processes into groups of N / NODE_COUNT",
        0, strvector_t({ "--timemory-node-count" }), 1);
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE
void
settings::initialize_tpls()
{
    // PRINT_HERE("%s", "");
    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        bool, papi_threading, TIMEMORY_SETTINGS_KEY("PAPI_THREADING"),
        "Enable multithreading support when using PAPI", true,
        strvector_t({ "--timemory-papi-threading" }), -1, 1);

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        bool, papi_multiplexing, TIMEMORY_SETTINGS_KEY("PAPI_MULTIPLEXING"),
        "Enable multiplexing when using PAPI", false,
        strvector_t({ "--timemory-papi-multiplexing" }), -1, 1);

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        bool, papi_fail_on_error, TIMEMORY_SETTINGS_KEY("PAPI_FAIL_ON_ERROR"),
        "Configure PAPI errors to trigger a runtime error", false,
        strvector_t({ "--timemory-papi-fail-on-error" }), -1, 1);

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        bool, papi_quiet, TIMEMORY_SETTINGS_KEY("PAPI_QUIET"),
        "Configure suppression of reporting PAPI errors/warnings", false,
        strvector_t({ "--timemory-papi-quiet" }), -1, 1);

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        string_t, papi_events, TIMEMORY_SETTINGS_KEY("PAPI_EVENTS"),
        "PAPI presets and events to collect (see also: papi_avail)", "",
        strvector_t({ "--timemory-papi-events" }));

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        bool, papi_attach, TIMEMORY_SETTINGS_KEY("PAPI_ATTACH"),
        "Configure PAPI to attach to another process (see also: TIMEMORY_TARGET_PID)",
        false);

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        int, papi_overflow, TIMEMORY_SETTINGS_KEY("PAPI_OVERFLOW"),
        "Value at which PAPI hw counters trigger an overflow callback", 0,
        strvector_t({ "--timemory-papi-overflow" }), 1);

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        uint64_t, cuda_event_batch_size, TIMEMORY_SETTINGS_KEY("CUDA_EVENT_BATCH_SIZE"),
        "Batch size for create cudaEvent_t in cuda_event components", 5);

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        bool, nvtx_marker_device_sync, TIMEMORY_SETTINGS_KEY("NVTX_MARKER_DEVICE_SYNC"),
        "Use cudaDeviceSync when stopping NVTX marker (vs. cudaStreamSychronize)", true);

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        int32_t, cupti_activity_level, TIMEMORY_SETTINGS_KEY("CUPTI_ACTIVITY_LEVEL"),
        "Default group of kinds tracked via CUpti Activity API", 1,
        strvector_t({ "--timemory-cupti-activity-level" }), 1);

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(string_t, cupti_activity_kinds,
                                      TIMEMORY_SETTINGS_KEY("CUPTI_ACTIVITY_KINDS"),
                                      "Specific cupti activity kinds to track", "",
                                      strvector_t({ "--timemory-cupti-activity-kinds" }));

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        string_t, cupti_events, TIMEMORY_SETTINGS_KEY("CUPTI_EVENTS"),
        "Hardware counter event types to collect on NVIDIA GPUs", "",
        strvector_t({ "--timemory-cupti-events" }));

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        string_t, cupti_metrics, TIMEMORY_SETTINGS_KEY("CUPTI_METRICS"),
        "Hardware counter metric types to collect on NVIDIA GPUs", "",
        strvector_t({ "--timemory-cupti-metrics" }));

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(int, cupti_device,
                                      TIMEMORY_SETTINGS_KEY("CUPTI_DEVICE"),
                                      "Target device for CUPTI data collection", 0,
                                      strvector_t({ "--timemory-cupti-device" }), 1);

    insert<int>("TIMEMORY_CUPTI_PCSAMPLING_PERIOD", "cupti_pcsampling_period",
                "The period for PC sampling. Must be >= 5 and <= 31", 8,
                strvector_t{ "--timemory-cupti-pcsampling-period" });

    insert<bool>(
        "TIMEMORY_CUPTI_PCSAMPLING_PER_LINE", "cupti_pcsampling_per_line",
        "Report the PC samples per-line or collapse into one entry for entire function",
        false, strvector_t{ "--timemory-cupti-pcsampling-per-line" });

    insert<bool>(
        "TIMEMORY_CUPTI_PCSAMPLING_REGION_TOTALS", "cupti_pcsampling_region_totals",
        "When enabled, region markers will report total samples from all child functions",
        true, strvector_t{ "--timemory-cupti-pcsampling-region-totals" });

    insert<bool>("TIMEMORY_CUPTI_PCSAMPLING_SERIALIZED", "cupti_pcsampling_serialized",
                 "Serialize all the kernel functions", false,
                 strvector_t{ "--timemory-cupti-pcsampling-serialize" });

    insert<size_t>("TIMEMORY_CUPTI_PCSAMPLING_NUM_COLLECT",
                   "cupti_pcsampling_num_collect", "Number of PCs to be collected",
                   size_t{ 100 },
                   strvector_t{ "--timemory-cupti-pcsampling-num-collect" });

    insert<std::string>("TIMEMORY_CUPTI_PCSAMPLING_STALL_REASONS",
                        "cupti_pcsampling_stall_reasons",
                        "The PC sampling stall reasons to count", std::string{},
                        strvector_t{ "--timemory-cupti-pcsampling-stall-reasons" });

    TIMEMORY_SETTINGS_MEMBER_IMPL(string_t, craypat_categories,
                                  TIMEMORY_SETTINGS_KEY("CRAYPAT"),
                                  "Configure the CrayPAT categories to collect",
                                  get_env<std::string>("PAT_RT_PERFCTR", ""))

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        string_t, python_exe, TIMEMORY_SETTINGS_KEY("PYTHON_EXE"),
        "Configure the python executable to use", TIMEMORY_PYTHON_PLOTTER,
        strvector_t({ "--timemory-python-exe" }));
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE
void
settings::initialize_roofline()
{
    // PRINT_HERE("%s", "");
    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        string_t, roofline_mode, TIMEMORY_SETTINGS_KEY("ROOFLINE_MODE"),
        "Configure the roofline collection mode. Options: 'op' 'ai'.", "op",
        strvector_t({ "--timemory-roofline-mode" }), 1, 1, strvector_t({ "op", "ai" }));

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        string_t, cpu_roofline_mode, TIMEMORY_SETTINGS_KEY("ROOFLINE_MODE_CPU"),
        "Configure the roofline collection mode for CPU specifically. Options: 'op', "
        "'ai'",
        "op", strvector_t({ "--timemory-cpu-roofline-mode" }), 1, 1,
        strvector_t({ "op", "ai" }));

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        string_t, gpu_roofline_mode, TIMEMORY_SETTINGS_KEY("ROOFLINE_MODE_GPU"),
        "Configure the roofline collection mode for GPU specifically. Options: 'op' "
        "'ai'.",
        static_cast<tsettings<string_t>*>(
            m_data[TIMEMORY_SETTINGS_KEY("ROOFLINE_MODE")].get())
            ->get(),
        strvector_t({ "--timemory-gpu-roofline-mode" }), 1, 1,
        strvector_t({ "op", "ai" }));

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        string_t, cpu_roofline_events, TIMEMORY_SETTINGS_KEY("ROOFLINE_EVENTS_CPU"),
        "Configure custom hw counters to add to the cpu roofline", "");

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        string_t, gpu_roofline_events, TIMEMORY_SETTINGS_KEY("ROOFLINE_EVENTS_GPU"),
        "Configure custom hw counters to add to the gpu roofline", "");

    TIMEMORY_SETTINGS_MEMBER_IMPL(bool, roofline_type_labels,
                                  TIMEMORY_SETTINGS_KEY("ROOFLINE_TYPE_LABELS"),
                                  "Configure roofline labels/descriptions/output-files "
                                  "encode the list of data types",
                                  false);

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        bool, roofline_type_labels_cpu, TIMEMORY_SETTINGS_KEY("ROOFLINE_TYPE_LABELS_CPU"),
        "Configure labels, etc. for the roofline components "
        "for CPU (see also: TIMEMORY_ROOFLINE_TYPE_LABELS)",
        static_cast<tsettings<bool>*>(
            m_data[TIMEMORY_SETTINGS_KEY("ROOFLINE_TYPE_LABELS")].get())
            ->get());

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        bool, roofline_type_labels_gpu, TIMEMORY_SETTINGS_KEY("ROOFLINE_TYPE_LABELS_GPU"),
        "Configure labels, etc. for the roofline components "
        "for GPU (see also: TIMEMORY_ROOFLINE_TYPE_LABELS)",
        static_cast<tsettings<bool>*>(
            m_data[TIMEMORY_SETTINGS_KEY("ROOFLINE_TYPE_LABELS")].get())
            ->get());

    TIMEMORY_SETTINGS_MEMBER_IMPL(bool, instruction_roofline,
                                  TIMEMORY_SETTINGS_KEY("INSTRUCTION_ROOFLINE"),
                                  "Configure the roofline to include the hw counters "
                                  "required for generating an instruction roofline",
                                  false);
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE void
settings::initialize_miscellaneous()
{
    // PRINT_HERE("%s", "");
    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        bool, add_secondary, TIMEMORY_SETTINGS_KEY("ADD_SECONDARY"),
        "Enable/disable components adding secondary (child) entries when available. E.g. "
        "suppress individual CUDA kernels, etc. when using Cupti components",
        true, strvector_t({ "--timemory-add-secondary" }), -1, 1);

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        size_t, throttle_count, TIMEMORY_SETTINGS_KEY("THROTTLE_COUNT"),
        "Minimum number of laps before checking whether a key should be throttled", 10000,
        strvector_t({ "--timemory-throttle-count" }), 1);

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        size_t, throttle_value, TIMEMORY_SETTINGS_KEY("THROTTLE_VALUE"),
        "Average call time in nanoseconds when # laps > throttle_count that triggers "
        "throttling",
        10000, strvector_t({ "--timemory-throttle-value" }), 1);

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        bool, enable_signal_handler, TIMEMORY_SETTINGS_KEY("ENABLE_SIGNAL_HANDLER"),
        "Enable signals in timemory_init", false,
        strvector_t({ "--timemory-enable-signal-handler" }), -1, 1)

    TIMEMORY_SETTINGS_REFERENCE_ARG_IMPL(
        bool, allow_signal_handler, TIMEMORY_SETTINGS_KEY("ALLOW_SIGNAL_HANDLER"),
        "Allow signal handling to be activated", signal_settings::allow(),
        strvector_t({ "--timemory-allow-signal-handler" }), -1, 1);

    TIMEMORY_SETTINGS_REFERENCE_IMPL(
        bool, enable_all_signals, TIMEMORY_SETTINGS_KEY("ENABLE_ALL_SIGNALS"),
        "Enable catching all signals", signal_settings::enable_all());

    TIMEMORY_SETTINGS_REFERENCE_IMPL(
        bool, disable_all_signals, TIMEMORY_SETTINGS_KEY("DISABLE_ALL_SIGNALS"),
        "Disable catching any signals", signal_settings::disable_all());

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        bool, destructor_report, TIMEMORY_SETTINGS_KEY("DESTRUCTOR_REPORT"),
        "Configure default setting for auto_{list,tuple,hybrid} to write to stdout during"
        " destruction of the bundle",
        false, strvector_t({ "--timemory-destructor-report" }), -1, 1);

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        bool, stack_clearing, TIMEMORY_SETTINGS_KEY("STACK_CLEARING"),
        "Enable/disable stopping any markers still running during finalization", true,
        strvector_t({ "--timemory-stack-clearing" }), -1, 1);

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        bool, banner, TIMEMORY_SETTINGS_KEY("BANNER"),
        "Notify about tim::manager creation and destruction",
        (get_env<bool>(TIMEMORY_SETTINGS_KEY("LIBRARY_CTOR"), false)));

    TIMEMORY_SETTINGS_HIDDEN_MEMBER_ARG_IMPL(
        std::string, TIMEMORY_SETTINGS_KEY("NETWORK_INTERFACE"),
        "Default network interface", std::string{},
        strvector_t({ "--timemory-network-interface" }), -1, 1);
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE
void
settings::initialize_ert()
{
    // PRINT_HERE("%s", "");
    TIMEMORY_SETTINGS_MEMBER_IMPL(uint64_t, ert_num_threads,
                                  TIMEMORY_SETTINGS_KEY("ERT_NUM_THREADS"),
                                  "Number of threads to use when running ERT", 0);

    TIMEMORY_SETTINGS_MEMBER_IMPL(uint64_t, ert_num_threads_cpu,
                                  TIMEMORY_SETTINGS_KEY("ERT_NUM_THREADS_CPU"),
                                  "Number of threads to use when running ERT on CPU",
                                  std::thread::hardware_concurrency());

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        uint64_t, ert_num_threads_gpu, TIMEMORY_SETTINGS_KEY("ERT_NUM_THREADS_GPU"),
        "Number of threads which launch kernels when running ERT on the GPU", 1);

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        uint64_t, ert_num_streams, TIMEMORY_SETTINGS_KEY("ERT_NUM_STREAMS"),
        "Number of streams to use when launching kernels in ERT on the GPU", 1);

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        uint64_t, ert_grid_size, TIMEMORY_SETTINGS_KEY("ERT_GRID_SIZE"),
        "Configure the grid size (number of blocks) for ERT on GPU (0 == auto-compute)",
        0);

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        uint64_t, ert_block_size, TIMEMORY_SETTINGS_KEY("ERT_BLOCK_SIZE"),
        "Configure the block size (number of threads per block) for ERT on GPU", 1024);

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        uint64_t, ert_alignment, TIMEMORY_SETTINGS_KEY("ERT_ALIGNMENT"),
        "Configure the alignment (in bits) when running ERT on CPU (0 == 8 * sizeof(T))",
        0);

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        uint64_t, ert_min_working_size, TIMEMORY_SETTINGS_KEY("ERT_MIN_WORKING_SIZE"),
        "Configure the minimum working size when running ERT (0 == device specific)", 0);

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        uint64_t, ert_min_working_size_cpu,
        TIMEMORY_SETTINGS_KEY("ERT_MIN_WORKING_SIZE_CPU"),
        "Configure the minimum working size when running ERT on CPU", 64);

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        uint64_t, ert_min_working_size_gpu,
        TIMEMORY_SETTINGS_KEY("ERT_MIN_WORKING_SIZE_GPU"),
        "Configure the minimum working size when running ERT on GPU", 10 * 1000 * 1000);

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        uint64_t, ert_max_data_size, TIMEMORY_SETTINGS_KEY("ERT_MAX_DATA_SIZE"),
        "Configure the max data size when running ERT on CPU", 0);

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        uint64_t, ert_max_data_size_cpu, TIMEMORY_SETTINGS_KEY("ERT_MAX_DATA_SIZE_CPU"),
        "Configure the max data size when running ERT on CPU", 0);

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        uint64_t, ert_max_data_size_gpu, TIMEMORY_SETTINGS_KEY("ERT_MAX_DATA_SIZE_GPU"),
        "Configure the max data size when running ERT on GPU", 500 * 1000 * 1000);

    TIMEMORY_SETTINGS_MEMBER_IMPL(
        string_t, ert_skip_ops, TIMEMORY_SETTINGS_KEY("ERT_SKIP_OPS"),
        "Skip these number of ops (i.e. ERT_FLOPS) when were set at compile time", "");
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE
void
settings::initialize_dart()
{
    // PRINT_HERE("%s", "");
    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        string_t, dart_type, TIMEMORY_SETTINGS_KEY("DART_TYPE"),
        "Only echo this measurement type (see also: TIMEMORY_DART_OUTPUT)", "",
        strvector_t({ "--timemory-dart-type" }));

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        uint64_t, dart_count, TIMEMORY_SETTINGS_KEY("DART_COUNT"),
        "Only echo this number of dart tags (see also: TIMEMORY_DART_OUTPUT)", 1,
        strvector_t({ "--timemory-dart-count" }), 1);

    TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(
        bool, dart_label, TIMEMORY_SETTINGS_KEY("DART_LABEL"),
        "Echo the category instead of the label (see also: TIMEMORY_DART_OUTPUT)", true,
        strvector_t({ "--timemory-dart-label" }), -1, 1);
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE
void
settings::initialize()
{
    // m_data.clear();
    if(m_data.empty())
        m_data.reserve(160);

    initialize_core();
    initialize_components();
    initialize_io();
    initialize_format();
    initialize_parallel();
    initialize_tpls();
    initialize_roofline();
    initialize_miscellaneous();
    initialize_ert();
    initialize_dart();
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE
bool
settings::read(const string_t& inp)
{
    std::ifstream ifs(inp);
    if(!ifs)
    {
        TIMEMORY_EXCEPTION(string_t("Error reading configuration file: ") + inp)
    }
    return read(ifs, inp);
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_INLINE
bool
settings::read(std::istream& ifs, std::string inp)
{
    if(inp.find(".json") != std::string::npos || inp == "json")
    {
        using policy_type = policy::input_archive<cereal::JSONInputArchive, TIMEMORY_API>;
        auto ia           = policy_type::get(ifs);
        try
        {
            ia->setNextName("timemory");
            ia->startNode();
            {
                try
                {
                    ia->setNextName("metadata");
                    ia->startNode();
                    // settings
                    (*ia)(cereal::make_nvp("settings", *this));
                    ia->finishNode();
                } catch(...)
                {
                    // settings
                    (*ia)(cereal::make_nvp("settings", *this));
                }
            }
            ia->finishNode();
        } catch(tim::cereal::Exception& e)
        {
            PRINT_HERE("Exception reading %s :: %s", inp.c_str(), e.what());
#    if defined(TIMEMORY_INTERNAL_TESTING)
            TIMEMORY_CONDITIONAL_DEMANGLED_BACKTRACE(true, 8);
#    endif
            return false;
        }
        return true;
    }
#    if defined(TIMEMORY_USE_XML)
    else if(inp.find(".xml") != std::string::npos || inp == "xml")
    {
        using policy_type = policy::input_archive<cereal::XMLInputArchive, TIMEMORY_API>;
        auto ia           = policy_type::get(ifs);
        ia->setNextName("timemory");
        ia->startNode();
        {
            try
            {
                ia->setNextName("metadata");
                ia->startNode();
                // settings
                (*ia)(cereal::make_nvp("settings", *this));
                ia->finishNode();
            } catch(...)
            {
                // settings
                (*ia)(cereal::make_nvp("settings", *this));
            }
        }
        ia->finishNode();
        return true;
    }
#    endif
    else
    {
        if(inp.empty())
            inp = "text";

        std::string line = "";

        auto is_comment = [](const std::string& s) {
            if(s.empty())
                return true;
            for(const auto& itr : s)
            {
                if(std::isprint(itr))
                    return (itr == '#') ? true : false;
            }
            // if there were no printable characters, treat as comment
            return true;
        };

        int expected = 0;
        int valid    = 0;
        while(ifs)
        {
            std::getline(ifs, line);
            if(!ifs)
                continue;
            if(is_comment(line))
                continue;
            ++expected;
            // tokenize the string
            auto delim = tim::delimit(line, "\n=,; ");
            if(delim.size() > 0)
            {
                string_t key = delim.front();
                string_t val = "";
                // combine into another string separated by commas
                for(size_t i = 1; i < delim.size(); ++i)
                    val += "," + delim.at(i);
                // if there was any fields, remove the leading comma
                if(val.length() > 0)
                    val = val.substr(1);
                auto incr = valid;
                for(auto itr : *this)
                {
                    if(itr.second->matches(key))
                    {
                        if(get_debug() || get_verbose() > 0)
                            fprintf(stderr, "[timemory::settings]['%s']> %-30s :: %s\n",
                                    inp.c_str(), key.c_str(), val.c_str());
                        ++valid;
                        itr.second->parse(val);
                    }
                }

                if(incr == valid)
                {
                    fprintf(stderr,
                            "[timemory::settings]['%s']> WARNING! Unknown setting "
                            "ignored: '%s' (value = '%s')\n",
                            inp.c_str(), key.c_str(), val.c_str());
                }
            }
        }
        return (expected == valid);
    }
    return false;
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_MEMBER_DEF(string_t, config_file, TIMEMORY_SETTINGS_KEY("CONFIG_FILE"))
TIMEMORY_SETTINGS_MEMBER_DEF(bool, suppress_parsing,
                             TIMEMORY_SETTINGS_KEY("SUPPRESS_PARSING"))
TIMEMORY_SETTINGS_MEMBER_DEF(bool, suppress_config,
                             TIMEMORY_SETTINGS_KEY("SUPPRESS_CONFIG"))
TIMEMORY_SETTINGS_MEMBER_DEF(bool, enabled, TIMEMORY_SETTINGS_KEY("ENABLED"))
TIMEMORY_SETTINGS_MEMBER_DEF(bool, auto_output, TIMEMORY_SETTINGS_KEY("AUTO_OUTPUT"))
TIMEMORY_SETTINGS_MEMBER_DEF(bool, cout_output, TIMEMORY_SETTINGS_KEY("COUT_OUTPUT"))
TIMEMORY_SETTINGS_MEMBER_DEF(bool, file_output, TIMEMORY_SETTINGS_KEY("FILE_OUTPUT"))
TIMEMORY_SETTINGS_MEMBER_DEF(bool, text_output, TIMEMORY_SETTINGS_KEY("TEXT_OUTPUT"))
TIMEMORY_SETTINGS_MEMBER_DEF(bool, json_output, TIMEMORY_SETTINGS_KEY("JSON_OUTPUT"))
TIMEMORY_SETTINGS_MEMBER_DEF(bool, tree_output, TIMEMORY_SETTINGS_KEY("TREE_OUTPUT"))
TIMEMORY_SETTINGS_MEMBER_DEF(bool, dart_output, TIMEMORY_SETTINGS_KEY("DART_OUTPUT"))
TIMEMORY_SETTINGS_MEMBER_DEF(bool, time_output, TIMEMORY_SETTINGS_KEY("TIME_OUTPUT"))
TIMEMORY_SETTINGS_MEMBER_DEF(bool, plot_output, TIMEMORY_SETTINGS_KEY("PLOT_OUTPUT"))
TIMEMORY_SETTINGS_MEMBER_DEF(bool, diff_output, TIMEMORY_SETTINGS_KEY("DIFF_OUTPUT"))
TIMEMORY_SETTINGS_MEMBER_DEF(bool, flamegraph_output,
                             TIMEMORY_SETTINGS_KEY("FLAMEGRAPH_OUTPUT"))
TIMEMORY_SETTINGS_MEMBER_DEF(bool, ctest_notes, TIMEMORY_SETTINGS_KEY("CTEST_NOTES"))
TIMEMORY_SETTINGS_MEMBER_DEF(int, verbose, TIMEMORY_SETTINGS_KEY("VERBOSE"))
TIMEMORY_SETTINGS_MEMBER_DEF(bool, debug, TIMEMORY_SETTINGS_KEY("DEBUG"))
TIMEMORY_SETTINGS_MEMBER_DEF(bool, banner, TIMEMORY_SETTINGS_KEY("BANNER"))
TIMEMORY_SETTINGS_MEMBER_DEF(bool, collapse_threads,
                             TIMEMORY_SETTINGS_KEY("COLLAPSE_THREADS"))
TIMEMORY_SETTINGS_MEMBER_DEF(bool, collapse_processes,
                             TIMEMORY_SETTINGS_KEY("COLLAPSE_PROCESSES"))
TIMEMORY_SETTINGS_MEMBER_DEF(uint16_t, max_depth, TIMEMORY_SETTINGS_KEY("MAX_DEPTH"))
TIMEMORY_SETTINGS_MEMBER_DEF(string_t, time_format, TIMEMORY_SETTINGS_KEY("TIME_FORMAT"))
TIMEMORY_SETTINGS_MEMBER_DEF(int16_t, precision, TIMEMORY_SETTINGS_KEY("PRECISION"))
TIMEMORY_SETTINGS_MEMBER_DEF(int16_t, width, TIMEMORY_SETTINGS_KEY("WIDTH"))
TIMEMORY_SETTINGS_MEMBER_DEF(int32_t, max_width, TIMEMORY_SETTINGS_KEY("MAX_WIDTH"))
TIMEMORY_SETTINGS_MEMBER_DEF(bool, scientific, TIMEMORY_SETTINGS_KEY("SCIENTIFIC"))
TIMEMORY_SETTINGS_MEMBER_DEF(int16_t, timing_precision,
                             TIMEMORY_SETTINGS_KEY("TIMING_PRECISION"))
TIMEMORY_SETTINGS_MEMBER_DEF(int16_t, timing_width, TIMEMORY_SETTINGS_KEY("TIMING_WIDTH"))
TIMEMORY_SETTINGS_MEMBER_DEF(string_t, timing_units,
                             TIMEMORY_SETTINGS_KEY("TIMING_UNITS"))
TIMEMORY_SETTINGS_MEMBER_DEF(bool, timing_scientific,
                             TIMEMORY_SETTINGS_KEY("TIMING_SCIENTIFIC"))
TIMEMORY_SETTINGS_MEMBER_DEF(int16_t, memory_precision,
                             TIMEMORY_SETTINGS_KEY("MEMORY_PRECISION"))
TIMEMORY_SETTINGS_MEMBER_DEF(int16_t, memory_width, TIMEMORY_SETTINGS_KEY("MEMORY_WIDTH"))
TIMEMORY_SETTINGS_MEMBER_DEF(string_t, memory_units,
                             TIMEMORY_SETTINGS_KEY("MEMORY_UNITS"))
TIMEMORY_SETTINGS_MEMBER_DEF(bool, memory_scientific,
                             TIMEMORY_SETTINGS_KEY("MEMORY_SCIENTIFIC"))
TIMEMORY_SETTINGS_MEMBER_DEF(string_t, output_path, TIMEMORY_SETTINGS_KEY("OUTPUT_PATH"))
TIMEMORY_SETTINGS_MEMBER_DEF(string_t, output_prefix,
                             TIMEMORY_SETTINGS_KEY("OUTPUT_PREFIX"))
TIMEMORY_SETTINGS_MEMBER_DEF(string_t, input_path, TIMEMORY_SETTINGS_KEY("INPUT_PATH"))
TIMEMORY_SETTINGS_MEMBER_DEF(string_t, input_prefix,
                             TIMEMORY_SETTINGS_KEY("INPUT_PREFIX"))
TIMEMORY_SETTINGS_MEMBER_DEF(string_t, input_extensions,
                             TIMEMORY_SETTINGS_KEY("INPUT_EXTENSIONS"))
TIMEMORY_SETTINGS_MEMBER_DEF(string_t, dart_type, TIMEMORY_SETTINGS_KEY("DART_TYPE"))
TIMEMORY_SETTINGS_MEMBER_DEF(uint64_t, dart_count, TIMEMORY_SETTINGS_KEY("DART_COUNT"))
TIMEMORY_SETTINGS_MEMBER_DEF(bool, dart_label, TIMEMORY_SETTINGS_KEY("DART_LABEL"))
TIMEMORY_SETTINGS_MEMBER_DEF(size_t, max_thread_bookmarks,
                             TIMEMORY_SETTINGS_KEY("MAX_THREAD_BOOKMARKS"))
TIMEMORY_SETTINGS_MEMBER_DEF(bool, cpu_affinity, TIMEMORY_SETTINGS_KEY("CPU_AFFINITY"))
TIMEMORY_SETTINGS_MEMBER_DEF(bool, stack_clearing,
                             TIMEMORY_SETTINGS_KEY("STACK_CLEARING"))
TIMEMORY_SETTINGS_MEMBER_DEF(bool, add_secondary, TIMEMORY_SETTINGS_KEY("ADD_SECONDARY"))
TIMEMORY_SETTINGS_MEMBER_DEF(size_t, throttle_count,
                             TIMEMORY_SETTINGS_KEY("THROTTLE_COUNT"))
TIMEMORY_SETTINGS_MEMBER_DEF(size_t, throttle_value,
                             TIMEMORY_SETTINGS_KEY("THROTTLE_VALUE"))
TIMEMORY_SETTINGS_MEMBER_DEF(string_t, global_components,
                             TIMEMORY_SETTINGS_KEY("GLOBAL_COMPONENTS"))
TIMEMORY_SETTINGS_MEMBER_DEF(string_t, tuple_components,
                             TIMEMORY_SETTINGS_KEY("TUPLE_COMPONENTS"))
TIMEMORY_SETTINGS_MEMBER_DEF(string_t, list_components,
                             TIMEMORY_SETTINGS_KEY("LIST_COMPONENTS"))
TIMEMORY_SETTINGS_MEMBER_DEF(string_t, ompt_components,
                             TIMEMORY_SETTINGS_KEY("OMPT_COMPONENTS"))
TIMEMORY_SETTINGS_MEMBER_DEF(string_t, mpip_components,
                             TIMEMORY_SETTINGS_KEY("MPIP_COMPONENTS"))
TIMEMORY_SETTINGS_MEMBER_DEF(string_t, ncclp_components,
                             TIMEMORY_SETTINGS_KEY("NCCLP_COMPONENTS"))
TIMEMORY_SETTINGS_MEMBER_DEF(string_t, trace_components,
                             TIMEMORY_SETTINGS_KEY("TRACE_COMPONENTS"))
TIMEMORY_SETTINGS_MEMBER_DEF(string_t, profiler_components,
                             TIMEMORY_SETTINGS_KEY("PROFILER_COMPONENTS"))
TIMEMORY_SETTINGS_MEMBER_DEF(string_t, kokkos_components,
                             TIMEMORY_SETTINGS_KEY("KOKKOS_COMPONENTS"))
TIMEMORY_SETTINGS_MEMBER_DEF(string_t, components, TIMEMORY_SETTINGS_KEY("COMPONENTS"))
TIMEMORY_SETTINGS_MEMBER_DEF(bool, mpi_init, TIMEMORY_SETTINGS_KEY("MPI_INIT"))
TIMEMORY_SETTINGS_MEMBER_DEF(bool, mpi_finalize, TIMEMORY_SETTINGS_KEY("MPI_FINALIZE"))
TIMEMORY_SETTINGS_MEMBER_DEF(bool, mpi_thread, TIMEMORY_SETTINGS_KEY("MPI_THREAD"))
TIMEMORY_SETTINGS_MEMBER_DEF(string_t, mpi_thread_type,
                             TIMEMORY_SETTINGS_KEY("MPI_THREAD_TYPE"))
TIMEMORY_SETTINGS_MEMBER_DEF(bool, upcxx_init, TIMEMORY_SETTINGS_KEY("UPCXX_INIT"))
TIMEMORY_SETTINGS_MEMBER_DEF(bool, upcxx_finalize,
                             TIMEMORY_SETTINGS_KEY("UPCXX_FINALIZE"))
TIMEMORY_SETTINGS_MEMBER_DEF(bool, papi_threading,
                             TIMEMORY_SETTINGS_KEY("PAPI_THREADING"))
TIMEMORY_SETTINGS_MEMBER_DEF(bool, papi_multiplexing,
                             TIMEMORY_SETTINGS_KEY("PAPI_MULTIPLEXING"))
TIMEMORY_SETTINGS_MEMBER_DEF(bool, papi_fail_on_error,
                             TIMEMORY_SETTINGS_KEY("PAPI_FAIL_ON_ERROR"))
TIMEMORY_SETTINGS_MEMBER_DEF(bool, papi_quiet, TIMEMORY_SETTINGS_KEY("PAPI_QUIET"))
TIMEMORY_SETTINGS_MEMBER_DEF(string_t, papi_events, TIMEMORY_SETTINGS_KEY("PAPI_EVENTS"))
TIMEMORY_SETTINGS_MEMBER_DEF(bool, papi_attach, TIMEMORY_SETTINGS_KEY("PAPI_ATTACH"))
TIMEMORY_SETTINGS_MEMBER_DEF(int, papi_overflow, TIMEMORY_SETTINGS_KEY("PAPI_OVERFLOW"))
TIMEMORY_SETTINGS_MEMBER_DEF(uint64_t, cuda_event_batch_size,
                             TIMEMORY_SETTINGS_KEY("CUDA_EVENT_BATCH_SIZE"))
TIMEMORY_SETTINGS_MEMBER_DEF(bool, nvtx_marker_device_sync,
                             TIMEMORY_SETTINGS_KEY("NVTX_MARKER_DEVICE_SYNC"))
TIMEMORY_SETTINGS_MEMBER_DEF(int32_t, cupti_activity_level,
                             TIMEMORY_SETTINGS_KEY("CUPTI_ACTIVITY_LEVEL"))
TIMEMORY_SETTINGS_MEMBER_DEF(string_t, cupti_activity_kinds,
                             TIMEMORY_SETTINGS_KEY("CUPTI_ACTIVITY_KINDS"))
TIMEMORY_SETTINGS_MEMBER_DEF(string_t, cupti_events,
                             TIMEMORY_SETTINGS_KEY("CUPTI_EVENTS"))
TIMEMORY_SETTINGS_MEMBER_DEF(string_t, cupti_metrics,
                             TIMEMORY_SETTINGS_KEY("CUPTI_METRICS"))
TIMEMORY_SETTINGS_MEMBER_DEF(int, cupti_device, TIMEMORY_SETTINGS_KEY("CUPTI_DEVICE"))
TIMEMORY_SETTINGS_MEMBER_DEF(string_t, roofline_mode,
                             TIMEMORY_SETTINGS_KEY("ROOFLINE_MODE"))
TIMEMORY_SETTINGS_MEMBER_DEF(string_t, cpu_roofline_mode,
                             TIMEMORY_SETTINGS_KEY("ROOFLINE_MODE_CPU"))
TIMEMORY_SETTINGS_MEMBER_DEF(string_t, gpu_roofline_mode,
                             TIMEMORY_SETTINGS_KEY("ROOFLINE_MODE_GPU"))
TIMEMORY_SETTINGS_MEMBER_DEF(string_t, cpu_roofline_events,
                             TIMEMORY_SETTINGS_KEY("ROOFLINE_EVENTS_CPU"))
TIMEMORY_SETTINGS_MEMBER_DEF(string_t, gpu_roofline_events,
                             TIMEMORY_SETTINGS_KEY("ROOFLINE_EVENTS_GPU"))
TIMEMORY_SETTINGS_MEMBER_DEF(bool, roofline_type_labels,
                             TIMEMORY_SETTINGS_KEY("ROOFLINE_TYPE_LABELS"))
TIMEMORY_SETTINGS_MEMBER_DEF(bool, roofline_type_labels_cpu,
                             TIMEMORY_SETTINGS_KEY("ROOFLINE_TYPE_LABELS_CPU"))
TIMEMORY_SETTINGS_MEMBER_DEF(bool, roofline_type_labels_gpu,
                             TIMEMORY_SETTINGS_KEY("ROOFLINE_TYPE_LABELS_GPU"))
TIMEMORY_SETTINGS_MEMBER_DEF(bool, instruction_roofline,
                             TIMEMORY_SETTINGS_KEY("INSTRUCTION_ROOFLINE"))
TIMEMORY_SETTINGS_MEMBER_DEF(uint64_t, ert_num_threads,
                             TIMEMORY_SETTINGS_KEY("ERT_NUM_THREADS"))
TIMEMORY_SETTINGS_MEMBER_DEF(uint64_t, ert_num_threads_cpu,
                             TIMEMORY_SETTINGS_KEY("ERT_NUM_THREADS_CPU"))
TIMEMORY_SETTINGS_MEMBER_DEF(uint64_t, ert_num_threads_gpu,
                             TIMEMORY_SETTINGS_KEY("ERT_NUM_THREADS_GPU"))
TIMEMORY_SETTINGS_MEMBER_DEF(uint64_t, ert_num_streams,
                             TIMEMORY_SETTINGS_KEY("ERT_NUM_STREAMS"))
TIMEMORY_SETTINGS_MEMBER_DEF(uint64_t, ert_grid_size,
                             TIMEMORY_SETTINGS_KEY("ERT_GRID_SIZE"))
TIMEMORY_SETTINGS_MEMBER_DEF(uint64_t, ert_block_size,
                             TIMEMORY_SETTINGS_KEY("ERT_BLOCK_SIZE"))
TIMEMORY_SETTINGS_MEMBER_DEF(uint64_t, ert_alignment,
                             TIMEMORY_SETTINGS_KEY("ERT_ALIGNMENT"))
TIMEMORY_SETTINGS_MEMBER_DEF(uint64_t, ert_min_working_size,
                             TIMEMORY_SETTINGS_KEY("ERT_MIN_WORKING_SIZE"))
TIMEMORY_SETTINGS_MEMBER_DEF(uint64_t, ert_min_working_size_cpu,
                             TIMEMORY_SETTINGS_KEY("ERT_MIN_WORKING_SIZE_CPU"))
TIMEMORY_SETTINGS_MEMBER_DEF(uint64_t, ert_min_working_size_gpu,
                             TIMEMORY_SETTINGS_KEY("ERT_MIN_WORKING_SIZE_GPU"))
TIMEMORY_SETTINGS_MEMBER_DEF(uint64_t, ert_max_data_size,
                             TIMEMORY_SETTINGS_KEY("ERT_MAX_DATA_SIZE"))
TIMEMORY_SETTINGS_MEMBER_DEF(uint64_t, ert_max_data_size_cpu,
                             TIMEMORY_SETTINGS_KEY("ERT_MAX_DATA_SIZE_CPU"))
TIMEMORY_SETTINGS_MEMBER_DEF(uint64_t, ert_max_data_size_gpu,
                             TIMEMORY_SETTINGS_KEY("ERT_MAX_DATA_SIZE_GPU"))
TIMEMORY_SETTINGS_MEMBER_DEF(string_t, ert_skip_ops,
                             TIMEMORY_SETTINGS_KEY("ERT_SKIP_OPS"))
TIMEMORY_SETTINGS_MEMBER_DEF(string_t, craypat_categories,
                             TIMEMORY_SETTINGS_KEY("CRAYPAT"))
TIMEMORY_SETTINGS_MEMBER_DEF(int32_t, node_count, TIMEMORY_SETTINGS_KEY("NODE_COUNT"))
TIMEMORY_SETTINGS_MEMBER_DEF(bool, destructor_report,
                             TIMEMORY_SETTINGS_KEY("DESTRUCTOR_REPORT"))
TIMEMORY_SETTINGS_MEMBER_DEF(string_t, python_exe, TIMEMORY_SETTINGS_KEY("PYTHON_EXE"))
// stream
TIMEMORY_SETTINGS_MEMBER_DEF(int64_t, separator_frequency,
                             TIMEMORY_SETTINGS_KEY("SEPARATOR_FREQ"))
// signals
TIMEMORY_SETTINGS_MEMBER_DEF(bool, enable_signal_handler,
                             TIMEMORY_SETTINGS_KEY("ENABLE_SIGNAL_HANDLER"))
TIMEMORY_SETTINGS_REFERENCE_DEF(bool, allow_signal_handler,
                                TIMEMORY_SETTINGS_KEY("ALLOW_SIGNAL_HANDLER"))
TIMEMORY_SETTINGS_REFERENCE_DEF(bool, enable_all_signals,
                                TIMEMORY_SETTINGS_KEY("ENABLE_ALL_SIGNALS"))
TIMEMORY_SETTINGS_REFERENCE_DEF(bool, disable_all_signals,
                                TIMEMORY_SETTINGS_KEY("DISABLE_ALL_SIGNALS"))
// miscellaneous ref
TIMEMORY_SETTINGS_REFERENCE_DEF(bool, flat_profile, TIMEMORY_SETTINGS_KEY("FLAT_PROFILE"))
TIMEMORY_SETTINGS_REFERENCE_DEF(bool, timeline_profile,
                                TIMEMORY_SETTINGS_KEY("TIMELINE_PROFILE"))
TIMEMORY_SETTINGS_REFERENCE_DEF(process::id_t, target_pid,
                                TIMEMORY_SETTINGS_KEY("TARGET_PID"))
//
//--------------------------------------------------------------------------------------//
//
}  // namespace tim

#    include "timemory/tpls/cereal/archives.hpp"

TIMEMORY_SETTINGS_EXTERN_TEMPLATE(TIMEMORY_API)

#endif  // TIMEMORY_SETTINGS_SETTINGS_CPP_

// #include "timemory/settings/extern.hpp"

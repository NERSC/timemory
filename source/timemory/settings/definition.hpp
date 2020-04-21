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
 * \file timemory/settings/definition.hpp
 * \brief The definitions for the types in settings
 */

#pragma once

#include "timemory/settings/declaration.hpp"
#include "timemory/settings/macros.hpp"
#include "timemory/settings/types.hpp"
#include "timemory/types.hpp"
#include "timemory/units.hpp"
#include "timemory/utility/filepath.hpp"
#include "timemory/utility/serializer.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

namespace tim
{
//
//--------------------------------------------------------------------------------------//
//
//                              settings
//
//--------------------------------------------------------------------------------------//
//
#if !defined(TIMEMORY_USE_EXTERN) && !defined(TIMEMORY_USE_SETTINGS_EXTERN)
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_LINKAGE(std::string)
get_local_datetime(const char* dt_format)
{
    std::stringstream ss;
    std::time_t       t = std::time(nullptr);
    char              mbstr[100];
    if(std::strftime(mbstr, sizeof(mbstr), dt_format, std::localtime(&t)))
    {
        ss << mbstr;
    }
    return ss.str();
}
//
//----------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_LINKAGE(std::string)
settings::tolower(std::string str)
{
    for(auto& itr : str)
        itr = ::tolower(itr);
    return str;
}
//
//----------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_LINKAGE(std::string)
settings::toupper(std::string str)
{
    for(auto& itr : str)
        itr = ::toupper(itr);
    return str;
}
//
//----------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_LINKAGE(std::string)
settings::get_input_prefix()
{
    static auto& _dir    = input_path();
    static auto& _prefix = input_prefix();

    return filepath::osrepr(_dir + std::string("/") + _prefix);
}
//
//----------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_LINKAGE(std::string)
settings::get_output_prefix(bool fake)
{
    static auto& _dir         = output_path();
    static auto& _prefix      = output_prefix();
    static auto& _time_output = time_output();
    static auto& _time_format = time_format();

    if(_time_output)
    {
        // ensure that all output files use same local datetime
        static auto _local_datetime = get_local_datetime(_time_format.c_str());
        if(_dir.find(_local_datetime) == std::string::npos)
        {
            if(_dir.length() > 0 && _dir[_dir.length() - 1] != '/')
                _dir += "/";
            _dir += _local_datetime;
        }
    }

    if(!fake && (debug() || verbose() > 2))
        PRINT_HERE("creating output directory: '%s'", _dir.c_str());

    if(fake)
        tim::consume_parameters(get_input_prefix());

    auto ret = (fake) ? 0 : makedir(_dir);
    return (ret == 0) ? filepath::osrepr(_dir + std::string("/") + _prefix)
                      : filepath::osrepr(std::string("./") + _prefix);
}
//
//----------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_LINKAGE(void)
settings::store_command_line(int argc, char** argv)
{
    auto& _cmdline = command_line();
    _cmdline.clear();
    for(int i = 0; i < argc; ++i)
        _cmdline.push_back(std::string(argv[i]));
}
//
//----------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_LINKAGE(std::string)
settings::compose_output_filename(const std::string& _tag, std::string _ext,
                                  bool _mpi_init, const int32_t _mpi_rank, bool fake,
                                  std::string _explicit)
{
    auto _prefix = (_explicit.length() > 0) ? _explicit : get_output_prefix(fake);

    // if just caching this static variable return
    if(fake)
        return "";

    auto only_ascii = [](char c) { return !isascii(c); };

    _prefix.erase(std::remove_if(_prefix.begin(), _prefix.end(), only_ascii),
                  _prefix.end());

    if(_explicit.length() > 0)
    {
        auto ret = makedir(_prefix);
        if(ret != 0)
            _prefix = filepath::osrepr(std::string("./"));
    }

    auto _rank_suffix = (_mpi_init && _mpi_rank >= 0)
                            ? (std::string("_") + std::to_string(_mpi_rank))
                            : std::string("");
    if(_ext.find('.') != 0)
        _ext = std::string(".") + _ext;
    auto plast = _prefix.length() - 1;
    if(_prefix.length() > 0 && _prefix[plast] != '/' && isalnum(_prefix[plast]))
        _prefix += "_";
    auto fpath = path_t(_prefix + _tag + _rank_suffix + _ext);
    while(fpath.find("//") != std::string::npos)
        fpath.replace(fpath.find("//"), 2, "/");
    return std::move(fpath);
}
//
//----------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_LINKAGE(std::string)
settings::compose_input_filename(const std::string& _tag, std::string _ext,
                                 bool _mpi_init, const int32_t _mpi_rank,
                                 std::string _explicit)
{
    if(settings::input_path().empty())
        settings::input_path() = settings::output_path();

    if(settings::input_prefix().empty())
        settings::input_prefix() = settings::output_prefix();

    auto _prefix = (_explicit.length() > 0) ? _explicit : get_input_prefix();

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
    auto fpath = path_t(_prefix + _tag + _rank_suffix + _ext);
    while(fpath.find("//") != std::string::npos)
        fpath.replace(fpath.find("//"), 2, "/");
    return std::move(fpath);
}
//
//----------------------------------------------------------------------------------//
//
// function to parse the environment for settings
//
// Nearly all variables will parse env when first access but this allows provides a
// way to reparse the environment so that default settings (possibly from previous
// invocation) can be overwritten
//
TIMEMORY_SETTINGS_LINKAGE(void)
settings::parse()
{
    if(suppress_parsing())
        return;

    for(auto& itr : get_parse_callbacks())
    {
        if(settings::debug() && settings::verbose() > 0)
            std::cerr << "Executing parse callback for: " << itr.first << std::endl;
        itr.second();
    }
}
//
//----------------------------------------------------------------------------------//
//
#endif  // !defined(TIMEMORY_USE_EXTERN) || defined(TIMEMORY_SETTINGS_SOURCE)
//
//--------------------------------------------------------------------------------------//
//
}  // namespace tim
//
//--------------------------------------------------------------------------------------//

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

#include "timemory/backends/process.hpp"
#include "timemory/macros/os.hpp"
#include "timemory/mpl/apply.hpp"
#include "timemory/tpls/cereal/cereal/cereal.hpp"
#include "timemory/utility/delimit.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/variadic/macros.hpp"

#include <array>
#include <cstddef>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

namespace tim
{
namespace procfs
{
struct maps
{
    TIMEMORY_DEFAULT_OBJECT(maps)

    maps(const std::vector<std::string>& _line);

    size_t              start_address = 0;
    size_t              end_address   = 0;
    std::array<char, 4> permissions   = {};
    size_t              offset        = 0;
    std::string         device        = {};
    size_t              inode         = {};
    std::string         pathname      = {};

    bool is_empty() const
    {
        // sum will be zero if default initialized
        return ((start_address + end_address + offset + inode + device.length() +
                 pathname.length()) == 0);
    }

    template <typename ArchiveT>
    void serialize(ArchiveT& ar, const unsigned)
    {
        auto _perms = std::string(permissions.size(), '-');
        for(size_t i = 0; i < permissions.size(); ++i)
            _perms.at(i) = permissions.at(i);

        ar(cereal::make_nvp("start_address", TIMEMORY_JOIN("", std::hex, start_address)),
           cereal::make_nvp("end_address", TIMEMORY_JOIN("", std::hex, end_address)),
           cereal::make_nvp("permissions", _perms),
           cereal::make_nvp("offset", TIMEMORY_JOIN("", std::hex, offset)),
           cereal::make_nvp("device", device), cereal::make_nvp("inode", inode),
           cereal::make_nvp("pathname", pathname));
    }
};

inline maps::maps(const std::vector<std::string>& _delim)
{
    auto _addr_str = _delim.front();
    auto _addr     = delimit(_delim.front(), "-");
    start_address  = std::stoull(_addr.front(), nullptr, 16);
    end_address    = std::stoull(_addr.back(), nullptr, 16);
    auto _perm     = _delim.at(1);
    for(size_t i = 0; i < permissions.size(); ++i)
        permissions.at(i) = _perm.at(i);
    offset = std::stoull(_delim.at(2), nullptr, 16);
    device = _delim.at(3);
    inode  = std::stoull(_delim.at(4));
    if(_delim.size() > 5)
        pathname = _delim.at(5);
}

inline auto
read_maps(pid_t _pid = process::get_target_id())
{
    auto          _data  = std::vector<maps>{};
    auto          _fname = TIMEMORY_JOIN('/', "/proc", _pid, "maps");
    std::ifstream ifs{ _fname };
    if(!ifs)
    {
        fprintf(stderr, "Failure opening %s\n", _fname.c_str());
    }
    else
    {
        while(ifs)
        {
            std::string _line = {};
            if(std::getline(ifs, _line) && !_line.empty())
            {
                auto _delim = delimit(_line, " \t\n\r");
                if(_delim.size() >= 4)
                    _data.emplace_back(_delim);
                else
                {
                    fprintf(stderr, "Discarding '%s'...\n", _line.c_str());
                }
            }
        }
    }
    return _data;
}

inline auto&
get_maps(pid_t _pid, bool _update = false)
{
    static auto                  _data  = std::unordered_map<pid_t, std::vector<maps>>{};
    static auto                  _mutex = std::mutex{};
    std::scoped_lock<std::mutex> _lk{ _mutex };

    auto itr = _data.find(_pid);
    if(itr == _data.end())
        _data.emplace(_pid, read_maps(_pid));
    else if(_update)
        _data.at(_pid) = read_maps(_pid);

    return _data.at(_pid);
}

inline auto
find_map(uint64_t _addr, pid_t _pid = process::get_target_id())
{
    auto _search_map = [_addr, _pid]() {
        // make sure it is copied
        auto _v = get_maps(_pid);
        for(const auto& itr : _v)
        {
            if(_addr >= itr.start_address && _addr <= itr.end_address)
                return itr;
        }
        return maps{};
    };

    auto _v = _search_map();

    // if not found, update
    if(_v.is_empty())
        get_maps(_pid, true);

    return _search_map();
}
}  // namespace procfs
}  // namespace tim

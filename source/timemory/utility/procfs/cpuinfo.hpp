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

#include "timemory/utility/macros.hpp"

#include <cstddef>
#include <fstream>
#include <memory>
#include <thread>
#include <vector>

namespace tim
{
namespace procfs
{
namespace cpuinfo
{
struct freq
{
    TIMEMORY_DEFAULT_OBJECT(freq)

    auto operator()(size_t _idx) const;
         operator bool() const;

    static auto                 get(size_t _idx);
    static size_t               size() { return get_offsets().size(); }
    static std::vector<size_t>& get_offsets();
    static auto&                get_ifstream()
    {
        static thread_local auto _v = []() {
            auto _ifs =
                std::make_unique<std::ifstream>("/proc/cpuinfo", std::ifstream::binary);
            if(!_ifs->is_open())
                _ifs = std::unique_ptr<std::ifstream>{};
            return _ifs;
        }();
        return _v;
    }
};

inline freq::operator bool() const { return (get_ifstream() != nullptr); }

inline auto
freq::get(size_t _idx)
{
    auto&  _ifs     = get_ifstream();
    auto&  _offsets = get_offsets();
    double _freq    = 0.0;
    _ifs->seekg(_offsets.at(_idx), _ifs->beg);
    (*_ifs) >> _freq;
    return _freq;
}

auto
freq::operator()(size_t _idx) const
{
    return freq::get(_idx % size());
}

inline std::vector<size_t>&
freq::get_offsets()
{
    static auto _v = []() {
        auto                _ncpu = std::thread::hardware_concurrency();
        std::vector<size_t> _cpu_mhz_pos{};
        std::ifstream       _ifs{ "/proc/cpuinfo" };
        if(_ifs)
        {
            for(size_t i = 0; i < _ncpu; ++i)
            {
                short       _n = 0;
                std::string _st{};
                while(_ifs && _ifs.good())
                {
                    std::string _s{};
                    _ifs >> _s;
                    if(!_ifs.good() || !_ifs)
                        break;

                    if(_s == "cpu" || _s == "MHz" || _s == ":")
                    {
                        ++_n;
                        _st += _s + " ";
                    }
                    else
                    {
                        _n  = 0;
                        _st = {};
                    }

                    if(_n == 3)
                    {
                        size_t _pos = _ifs.tellg();
                        _cpu_mhz_pos.emplace_back(_pos + 1);
                        _ifs >> _s;
                        if(!_ifs.good() || !_ifs)
                            break;
                        break;
                    }
                }
            }
        }

        _ifs.close();
        return _cpu_mhz_pos;
    }();
    return _v;
}
}  // namespace cpuinfo
}  // namespace procfs
}  // namespace tim

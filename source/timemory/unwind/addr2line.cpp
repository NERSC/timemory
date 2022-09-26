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

#include "timemory/unwind/addr2line.hpp"

#include "timemory/log/macros.hpp"

#include <linux/limits.h>
#include <thread>

#if defined(TIMEMORY_USE_BFD)
#    define PACKAGE "timemory-bfd"
#    include <bfd.h>
#endif

namespace tim
{
namespace unwind
{
#if defined(TIMEMORY_USE_BFD)

void
find_address_in_section(bfd* _inp, asection* section, void* data)
{
    if(!data)
        return;

    auto* _info = static_cast<addr2line_info*>(data);
    if(*_info)
        return;

    if((bfd_section_flags(section) & SEC_ALLOC) == 0)
        return;

    bfd_vma       pc    = _info->address;
    bfd_vma       vma   = bfd_section_vma(section);
    bfd_size_type size  = bfd_section_size(section);
    auto**        _syms = reinterpret_cast<asymbol**>(_info->input->syms);

    if(!_syms)
        return;

    if(pc < vma || pc >= vma + size)
        return;

    {
        unsigned int _line = 0;
        const char*  _file = nullptr;
        const char*  _func = nullptr;
        _info->found       = (bfd_find_nearest_line_discriminator(
                            _inp, section, _syms, pc - vma, &_file, &_func, &_line,
                            &_info->discriminator) != 0);
        if(_info->found)
            _info->add_lineinfo(_func, _file, _line);
    }

    while(_info->found)
    {
        unsigned int _line = 0;
        const char*  _file = nullptr;
        const char*  _func = nullptr;
        if(bfd_find_inliner_info(_inp, &_file, &_func, &_line) != 0)
            _info->add_lineinfo(_func, _file, _line);
        else
            break;
    }
}

addr2line_info
addr2line(std::shared_ptr<bfd_file> _file, const std::vector<uint64_t>& _addresses)
{
    addr2line_info _info{ std::move(_file) };

    if(_info.input && _info.input->is_good())
    {
        bfd* _data = static_cast<bfd*>(_info.input->data);
        if(!_data)
            return _info;

        for(auto itr : _addresses)
        {
            _info.address = itr;
            bfd_map_over_sections(_data, find_address_in_section, &_info);
            if(!_info.lines.empty())
                break;
        }
    }

    return _info;
}

void
addr2line_info::add_lineinfo(const char* _func, const char* _file, unsigned int _line)
{
    auto _get_realpath = [](const char* _v) {
        char _rv[PATH_MAX];
        if(realpath(_v, _rv) == nullptr)
            return std::string{};
        return std::string{ _rv };
    };

    size_t _len = 0;
    _len += (_func) ? strlen(_func) : 0;
    _len += (_file) ? strlen(_file) : 0;
    if(_len > 0 && _file)
    {
        auto _lineinfo = addr2line_info::lineinfo{};
        _lineinfo.line = _line;
        if(_func)
            _lineinfo.name = _func;
        if(_file)
            _lineinfo.location = _get_realpath(_file);
        lines.emplace_back(_lineinfo);
    }
}

#else

addr2line_info
addr2line(std::shared_ptr<bfd_file> _file, const std::vector<uint64_t>&)
{
    return addr2line_info{ std::move(_file) };
}

void
addr2line_info::add_lineinfo(const char*, const char*, unsigned int)
{}

#endif

addr2line_info::addr2line_info(std::shared_ptr<bfd_file> _v)
: input{ std::move(_v) }
{}

addr2line_info::operator bool() const { return found && !lines.empty(); }

addr2line_info
addr2line(std::shared_ptr<bfd_file> _file, unsigned long address)
{
    return addr2line(std::move(_file), std::vector<uint64_t>{ address });
}

addr2line_info
addr2line(const std::string& dso_name, unsigned long address)
{
    auto _file = std::make_shared<bfd_file>(dso_name);

    return addr2line(_file, address);
}
}  // namespace unwind
}  // namespace tim

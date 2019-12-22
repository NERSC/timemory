//  MIT License
//
//  Copyright (c) 2019, The Regents of the University of California,
// through Lawrence Berkeley National Laboratory (subject to receipt of any
// required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to
//  deal in the Software without restriction, including without limitation the
//  rights to use, copy, modify, merge, publish, distribute, sublicense, and
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in
//  all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
//  IN THE SOFTWARE.

#include "available.hpp"
#include "timemory/timemory.hpp"

#include "timemory/components/base.hpp"
#include "timemory/components/caliper.hpp"
#include "timemory/components/cuda/event.hpp"
#include "timemory/components/cuda/nvtx_marker.hpp"
#include "timemory/components/cuda/profiler.hpp"
#include "timemory/components/cupti/activity.hpp"
#include "timemory/components/cupti/counters.hpp"
#include "timemory/components/derived/malloc_gotcha.hpp"
#include "timemory/components/general.hpp"
#include "timemory/components/gotcha.hpp"
#include "timemory/components/likwid.hpp"
#include "timemory/components/papi/array.hpp"
#include "timemory/components/papi/tuple.hpp"
#include "timemory/components/placeholder.hpp"
#include "timemory/components/properties.hpp"
#include "timemory/components/roofline/cpu.hpp"
#include "timemory/components/roofline/gpu.hpp"
#include "timemory/components/rusage.hpp"
#include "timemory/components/skeletons.hpp"
#include "timemory/components/tau.hpp"
#include "timemory/components/timing.hpp"
#include "timemory/components/types.hpp"
#include "timemory/components/user_bundle.hpp"
#include "timemory/components/vtune/event.hpp"
#include "timemory/components/vtune/frame.hpp"

#include <array>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <sstream>
#include <tuple>
#include <vector>

using namespace tim;
using string_t       = std::string;
using stringstream_t = std::stringstream;
using str_vec_t      = std::vector<string_t>;
using str_set_t      = tim::component::idset_t;
using info_type      = std::tuple<string_t, bool, str_vec_t>;

template <typename _Tp, size_t _N>
using array_t = std::array<_Tp, _N>;

char global_delim = '|';
bool markdown     = false;
int  padding      = 4;

//--------------------------------------------------------------------------------------//

template <typename Type>
struct get_availability
{
    using this_type  = get_availability<Type>;
    using value_type = typename Type::value_type;

    static info_type get_info()
    {
        bool     is_available = trait::is_available<Type>::value;
        bool     file_output  = generates_output<Type>::value;
        auto     name         = demangle<Type>();
        auto     label        = (file_output) ? Type::label() : std::string("");
        auto     description  = Type::description();
        auto     data_type    = demangle<value_type>();
        string_t enum_type    = component::properties<Type>::enum_string();
        string_t id_type      = component::properties<Type>::id();
        auto     ids_set      = component::properties<Type>::ids();
        auto     itr          = ids_set.begin();
        string_t db           = (markdown) ? "`\"" : "\"";
        string_t de           = (markdown) ? "\"`" : "\"";
        string_t ids_str      = TIMEMORY_JOIN("", TIMEMORY_JOIN("", db, *itr++, de));
        for(; itr != ids_set.end(); ++itr)
            ids_str = TIMEMORY_JOIN("  ", ids_str, TIMEMORY_JOIN("", db, *itr, de));

        return info_type{ name, is_available,
                          str_vec_t{ label, enum_type, id_type, ids_str, description,
                                     data_type } };
    }

    explicit get_availability(info_type& _info) { _info = this_type::get_info(); }
};

//--------------------------------------------------------------------------------------//

template <typename... _Types>
struct get_availability<component_list<_Types...>>
{
    using this_type = get_availability<component_list<_Types...>>;

    static constexpr size_t size() { return sizeof...(_Types); }
    static constexpr size_t nelem = sizeof...(_Types);

    using info_vec_t  = array_t<info_type, nelem>;
    using avail_types = std::tuple<get_availability<_Types>...>;

    static info_vec_t get_info()
    {
        info_vec_t _info;
        apply<void>::access<avail_types>(_info);
        return _info;
    }
};

//--------------------------------------------------------------------------------------//

string_t
remove(string_t inp, const std::set<string_t>& entries)
{
    for(const auto& itr : entries)
    {
        auto idx = inp.find(itr);
        while(idx != string_t::npos)
        {
            inp.erase(idx, itr.length());
            idx = inp.find(itr);
        }
    }
    return inp;
}

//--------------------------------------------------------------------------------------//

template <typename _Tp>
void
write_entry(std::ostream& os, const _Tp& _entry, int64_t _w, bool center, bool mark)
{
    stringstream_t ssentry, ssbeg, ss;
    ssentry << std::boolalpha << _entry;
    auto _sentry = remove(ssentry.str(), { "tim::", "component::" });
    auto wbeg    = (_w / 2) - (_sentry.length() / 2) - 1;
    if(!center)
        wbeg = 1;
    ssbeg << std::setw(wbeg) << "";
    if(mark && markdown)
        ssbeg << '`' << _sentry << '`';
    else
        ssbeg << _sentry;

    ss << std::left << std::setw(_w - 1) << ssbeg.str() << global_delim;
    os << ss.str();
}

//--------------------------------------------------------------------------------------//

template <typename _IntArray, size_t _N>
string_t
banner(_IntArray _breaks, std::array<bool, _N> _use, char filler = '-', char delim = '|')
{
    stringstream_t ss;
    ss.fill(filler);
    int64_t _remain = 0;
    for(size_t i = 0; i < _breaks.size(); ++i)
    {
        if(_use.at(i))
            _remain += _breaks.at(i);
    }
    auto _total = _remain;
    ss << delim;
    _remain -= 1;
    _breaks.at(0) -= 1;
    for(size_t i = 0; i < _breaks.size(); ++i)
    {
        if(!_use.at(i))
            continue;
        ss << std::setw(_breaks.at(i) - 1) << "" << delim;
        _remain -= _breaks.at(i);
    }
    ss << "\n";
    if(_remain != 0)
        printf("[banner]> non-zero remainder: %i with total: %i\n", (int) _remain,
               (int) _total);
    return ss.str();
}

//--------------------------------------------------------------------------------------//

static constexpr size_t num_component_options = 6;
static constexpr size_t num_settings_options  = 0;

template <size_t _N = num_component_options>
void
write_component_info(std::ostream&, const array_t<bool, _N>&, const array_t<bool, _N>&,
                     const array_t<string_t, _N>&);

template <size_t _N = num_settings_options>
void
write_settings_info(std::ostream&, const array_t<bool, _N>& = array_t<bool, _N>{},
                    const array_t<bool, _N>&     = array_t<bool, _N>{},
                    const array_t<string_t, _N>& = array_t<string_t, _N>{});

//--------------------------------------------------------------------------------------//

void
usage()
{
    std::vector<std::array<std::string, 4>> _options = {
        { "", "", "", "" },
        { "-h", "--help", "", "This menu" },
        { "", "", "", "" },
        { "-a", "--alias", "", "Display the C++ tim::component namesp alias" },
        { "-e", "--enum", "", "Display the enumeration ID" },
        { "-v", "--value", "", "Output the value type for the component" },
        { "-s", "--string", "", "Display all acceptable string identifiers" },
        { "-f", "--filename", "", "Output the filename for the component" },
        { "-d", "--description", "", "Output the description for the component" },
        { "", "", "", "" },
        { "-O", "--output", "<FILE>", "Write results to file" },
        { "-S", "--settings", "", "Display the runtime settings" },
        { "-C", "--components", "", "Only display the components data" },
        { "-M", "--markdown", "", "Write data in markdown" },
        { "", "", "", "" },
    };

    for(const auto& itr : _options)
    {
        std::cout << "\t";
        for(size_t i = 0; i < itr.size(); ++i)
        {
            auto len = itr.at(i).length();

            if(i > 2 && len > 0)
                std::cout << "[";

            std::cout << itr.at(i);

            if(i == 0 && len > 0)
                std::cout << "/";
            else if(i == 2 && len > 0)
                std::cout << " -- ";
            else if(i > 2 && len > 0)
                std::cout << "]";
            else if(len > 0)
                std::cout << " ";
        }
        std::cout << "\n";
    }

    exit(EXIT_FAILURE);
}

//--------------------------------------------------------------------------------------//

enum
{
    FNAME = 0,
    ENUM  = 1,
    ALIAS = 2,
    CID   = 3,
    DESC  = 4,
    VAL   = 5
};

int
main(int argc, char** argv)
{
    array_t<bool, 6>     options  = { false, false, false, false, false, false };
    array_t<string_t, 6> fields   = {};
    array_t<bool, 6>     use_mark = {};

    fields[CID]   = "STRING_IDS";
    fields[VAL]   = "VALUE_TYPE";
    fields[DESC]  = "DESCRIPTION";
    fields[ENUM]  = "ENUMERATION";
    fields[ALIAS] = "C++ ALIAS / PYTHON ENUMERATION";
    fields[FNAME] = "FILENAME";

    use_mark[CID]   = false;
    use_mark[VAL]   = true;
    use_mark[DESC]  = false;
    use_mark[ENUM]  = true;
    use_mark[ALIAS] = true;
    use_mark[FNAME] = false;

    bool include_settings   = false;
    bool include_components = false;

    std::string file = "";
    for(int i = 1; i < argc; ++i)
    {
        string_t _arg = argv[i];
        if(_arg == "-f" || _arg == "--filename")
            options[FNAME] = !options[FNAME];
        else if(_arg == "-d" || _arg == "--description")
            options[DESC] = !options[DESC];
        else if(_arg == "-v" || _arg == "--value")
            options[VAL] = !options[VAL];
        else if(_arg == "-e" || _arg == "--enum")
            options[ENUM] = !options[ENUM];
        else if(_arg == "-a" || _arg == "--alias")
            options[ALIAS] = !options[ALIAS];
        else if(_arg == "-s" || _arg == "--string")
            options[CID] = !options[CID];
        else if(_arg == "-O" || _arg == "--output")
        {
            if(i + 1 < argc && argv[i + 1][0] != '-')
                file = argv[++i];
            else
                throw std::runtime_error("-o/--output requires a file");
        }
        else if(_arg == "-C" || _arg == "--components")
            include_components = true;
        else if(_arg == "-S" || _arg == "--settings")
            include_settings = true;
        else if(_arg == "-M" || _arg == "--markdown")
        {
            markdown = true;
            padding  = 6;
        }
        else
            usage();
    }

    if(!include_components && !include_settings)
    {
        include_components = true;
    }

    std::ostream* os = nullptr;
    std::ofstream ofs;
    if(!file.empty())
    {
        ofs.open(file.c_str());
        if(ofs)
            os = &ofs;
        else
            std::cerr << "Error opening output file: " << file << std::endl;
    }

    if(!os)
        os = &std::cout;

    if(include_components)
        write_component_info(*os, options, use_mark, fields);

    if(include_settings)
        write_settings_info(*os);

    return 0;
}

//--------------------------------------------------------------------------------------//

template <size_t _N>
void
write_component_info(std::ostream& os, const array_t<bool, _N>& options,
                     const array_t<bool, _N>& _mark, const array_t<string_t, _N>& fields)
{
    static_assert(_N >= num_component_options,
                  "Error! Too few component options + fields");

    auto _info = get_availability<complete_list_t>::get_info();

    using int_vec_t  = std::vector<int64_t>;
    using width_type = int_vec_t;
    using width_bool = std::array<bool, _N + 2>;

    width_type _widths = width_type{ 40, 12, 20, 40, 20, 20, 20, 40 };
    width_bool _wusing = width_bool{ true, true };
    for(size_t i = 0; i < options.size(); ++i)
        _wusing[i + 2] = options[i];

    int64_t pad = padding;

    {
        constexpr size_t idx = 0;
        stringstream_t   ss;
        write_entry(ss, "COMPONENT", 0, false, true);
        _widths.at(idx) = std::max<int64_t>(ss.str().length() + pad - 1, _widths.at(idx));
    }
    {
        constexpr size_t idx = 1;
        stringstream_t   ss;
        write_entry(ss, "AVAILABLE", _widths.at(1), true, false);
        _widths.at(idx) = std::max<int64_t>(ss.str().length() + pad, _widths.at(idx));
    }

    for(size_t i = 0; i < fields.size(); ++i)
    {
        constexpr size_t idx = 2;
        stringstream_t   ss;
        if(!options[i])
            continue;
        write_entry(ss, fields[i], _widths.at(i + 2), true, _mark.at(idx));
        _widths.at(idx + i) =
            std::max<int64_t>(ss.str().length() + pad, _widths.at(idx + i));
    }

    // compute the widths
    for(const auto& itr : _info)
    {
        {
            constexpr size_t idx = 0;
            stringstream_t   ss;
            write_entry(ss, std::get<idx>(itr), 0, true, true);
            _widths.at(idx) =
                std::max<int64_t>(ss.str().length() + pad - 1, _widths.at(idx));
        }

        {
            constexpr size_t idx = 1;
            stringstream_t   ss;
            write_entry(ss, std::get<idx>(itr), 0, true, false);
            _widths.at(idx) = std::max<int64_t>(ss.str().length() + pad, _widths.at(idx));
        }

        constexpr size_t idx = 2;
        for(size_t i = 0; i < std::get<2>(itr).size(); ++i)
        {
            stringstream_t ss;
            write_entry(ss, std::get<idx>(itr)[i], 0, true, _mark.at(idx));
            _widths.at(idx + i) =
                std::max<int64_t>(ss.str().length() + pad, _widths.at(idx + i));
        }
    }

    if(!markdown)
        os << banner(_widths, _wusing, '-');

    os << global_delim << ' ';
    write_entry(os, "COMPONENT", _widths.at(0) - 2, true, false);
    write_entry(os, "AVAILABLE", _widths.at(1), true, false);
    for(size_t i = 0; i < fields.size(); ++i)
    {
        if(!options[i])
            continue;
        write_entry(os, fields[i], _widths.at(i + 2), true, false);
    }

    os << "\n" << banner(_widths, _wusing, '-');

    for(const auto& itr : _info)
    {
        os << global_delim;
        write_entry(os, std::get<0>(itr), _widths.at(0) - 1, false, true);
        write_entry(os, std::get<1>(itr), _widths.at(1), true, false);
        for(size_t i = 0; i < std::get<2>(itr).size(); ++i)
        {
            if(!options[i])
                continue;
            bool center = (i > 0) ? false : true;
            write_entry(os, std::get<2>(itr).at(i), _widths.at(i + 2), center,
                        _mark.at(i));
        }
        os << "\n";
    }

    if(!markdown)
        os << banner(_widths, _wusing, '-');

    os << "\n" << std::flush;
    // os << banner(total_width) << std::flush;
}

//--------------------------------------------------------------------------------------//

template <size_t _N>
void
write_settings_info(std::ostream& os, const array_t<bool, _N>&, const array_t<bool, _N>&,
                    const array_t<string_t, _N>&)
{
    static_assert(_N >= num_settings_options, "Error! Too few settings options + fields");

    using archive_type = cereal::SettingsTextArchive;
    using array_type   = typename archive_type::array_type;
    using exclude_type = typename archive_type::exclude_type;
    using width_type   = array_t<int64_t, 4>;
    using width_bool   = array_t<bool, 4>;

    array_type   _setting_output;
    exclude_type _settings_exclude = { "TIMEMORY_ENVIRONMENT", "TIMEMORY_COMMAND_LINE" };

    cereal::SettingsTextArchive settings_archive(_setting_output, _settings_exclude);
    settings::serialize_settings(settings_archive);

    width_type _widths;
    width_bool _wusing;
    width_bool _mark = { false, true, false, false };
    for(size_t i = 0; i < _widths.size(); ++i)
    {
        _widths.at(i) = 0;
        _wusing.at(i) = true;
    }

    for(const auto& itr : _setting_output)
        for(size_t i = 0; i < itr.size(); ++i)
            _widths.at(i) =
                std::max<uint64_t>(_widths.at(i), itr.at(i).length() + padding);

    array_t<string_t, 4> _labels = { "ENVIRONMENT", "C++ STATIC ACCESSOR", "TYPE",
                                     "VALUE" };
    array_t<bool, 4>     _center = { false, false, true, true };

    if(!markdown)
        os << banner(_widths, _wusing, '-');
    os << global_delim;
    for(size_t i = 0; i < _labels.size(); ++i)
    {
        auto _w = _widths.at(i) - ((i == 0) ? 1 : 0);
        write_entry(os, _labels.at(i), _w, true, false);
    }
    os << "\n" << banner(_widths, _wusing, '-');

    for(const auto& itr : _setting_output)
    {
        os << global_delim;
        for(size_t i = 0; i < itr.size(); ++i)
        {
            auto _w = _widths.at(i) - ((i == 0) ? 1 : 0);
            write_entry(os, itr.at(i), _w, _center.at(i), _mark.at(i));
        }
        os << "\n";
    }

    if(!markdown)
        os << banner(_widths, _wusing, '-');

    os << "\n" << std::flush;
    // os << banner(total_width, '-') << std::flush;
}

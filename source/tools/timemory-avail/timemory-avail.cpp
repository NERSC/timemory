//  MIT License
//
//  Copyright (c) 2020, The Regents of the University of California,
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

#include "timemory-avail.hpp"
//
#include "timemory/components.hpp"
#include "timemory/components/definition.hpp"
#include "timemory/components/placeholder.hpp"
#include "timemory/components/properties.hpp"
#include "timemory/components/skeletons.hpp"
#include "timemory/timemory.hpp"
#include "timemory/utility/argparse.hpp"

#include <algorithm>
#include <array>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <sstream>
#include <tuple>
#include <vector>

#if defined(TIMEMORY_UNIX)
#    include <sys/ioctl.h>  // ioctl() and TIOCGWINSZ
#    include <unistd.h>     // for STDOUT_FILENO
#elif defined(TIMEMORY_WINDOWS)
#    include <windows.h>
#endif

using namespace tim;

template <typename Tp, size_t N>
using array_t        = std::array<Tp, N>;
using string_t       = std::string;
using stringstream_t = std::stringstream;
using str_vec_t      = std::vector<string_t>;
using info_type      = std::tuple<string_t, bool, str_vec_t>;

char             global_delim           = '|';
bool             markdown               = false;
bool             alphabetical           = false;
bool             all_info               = false;
bool             force_brief            = false;
bool             debug_msg              = false;
int32_t          max_width              = 0;
int32_t          num_cols               = 0;
int32_t          min_width              = 40;
int32_t          padding                = 4;
string_t         regex_key              = {};
bool             regex_hl               = false;
constexpr size_t num_component_options  = 6;
constexpr size_t num_settings_options   = 3;
constexpr size_t num_hw_counter_options = 5;

//--------------------------------------------------------------------------------------//

static std::tuple<int32_t, std::string>
get_window_columns();

template <typename IntArrayT, typename BoolArrayT>
static IntArrayT
compute_max_columns(IntArrayT _widths, BoolArrayT _using);

string_t
remove(string_t inp, const std::set<string_t>& entries);

template <typename Tp>
void
write_entry(std::ostream& os, const Tp& _entry, int64_t _w, bool center, bool mark);

template <typename IntArrayT, size_t N>
string_t
banner(IntArrayT _breaks, std::array<bool, N> _use, char filler = '-', char delim = '|');

bool
not_filtered(const std::string& line);

std::string
hl_filtered(std::string line);

template <size_t N = num_component_options>
void
write_component_info(std::ostream&, const array_t<bool, N>&, const array_t<bool, N>&,
                     const array_t<string_t, N>&);

template <size_t N = num_settings_options>
void
write_settings_info(std::ostream&, const array_t<bool, N>& = {},
                    const array_t<bool, N>& = {}, const array_t<string_t, N>& = {});

template <size_t N = num_hw_counter_options>
void
write_hw_counter_info(std::ostream&, const array_t<bool, N>& = {},
                      const array_t<bool, N>& = {}, const array_t<string_t, N>& = {});

//--------------------------------------------------------------------------------------//

template <typename Type>
struct get_availability
{
    using this_type  = get_availability<Type>;
    using metadata_t = component::metadata<Type>;
    using property_t = component::properties<Type>;

    static info_type get_info();
};

//--------------------------------------------------------------------------------------//

template <typename... Types>
struct get_availability<type_list<Types...>>
{
    static constexpr auto N = sizeof...(Types);
    using data_type         = std::array<info_type, N>;

    static data_type get_info()
    {
        return TIMEMORY_FOLD_EXPANSION(info_type, N, get_availability<Types>::get_info());
    }
};

//--------------------------------------------------------------------------------------//

enum
{
    VAL   = 0,
    ENUM  = 1,
    LANG  = 2,
    CID   = 3,
    FNAME = 4,
    DESC  = 5,
    TOTAL = 6
};

//--------------------------------------------------------------------------------------//

int
main(int argc, char** argv)
{
    array_t<bool, 6>     options  = { false, false, false, false, false, false };
    array_t<string_t, 6> fields   = {};
    array_t<bool, 6>     use_mark = {};

    std::string cols_via{};
    std::tie(num_cols, cols_via) = get_window_columns();
    std::string col_msg =
        "(default: " + std::to_string(num_cols) + " [via " + cols_via + "])";

    fields[VAL]   = "VALUE_TYPE";
    fields[ENUM]  = "ENUMERATION";
    fields[LANG]  = "C++ ALIAS / PYTHON ENUMERATION";
    fields[FNAME] = "FILENAME";
    fields[CID]   = "STRING_IDS";
    fields[DESC]  = "DESCRIPTION";

    use_mark[VAL]   = true;
    use_mark[ENUM]  = true;
    use_mark[LANG]  = true;
    use_mark[FNAME] = false;
    use_mark[CID]   = false;
    use_mark[DESC]  = false;

    bool include_settings    = false;
    bool include_components  = false;
    bool include_hw_counters = false;

    std::string file = {};

    using parser_t = tim::argparse::argument_parser;
    parser_t parser("timemory-avail");

    parser.enable_help();
    parser.set_help_width(40);
    parser.add_argument({ "--debug" }, "Enable debug messages")
        .max_count(1)
        .action([](parser_t& p) { debug_msg = p.get<bool>("debug"); });
    parser.add_argument({ "-a", "--all" }, "Print all available info")
        .max_count(1)
        .action([&](parser_t& p) {
            all_info = p.get<bool>("all");
            if(all_info)
            {
                for(auto& itr : options)
                    itr = true;
                include_components  = true;
                include_settings    = true;
                include_hw_counters = true;
            }
        });

    parser.add_argument({ "" }, "");
    parser.add_argument({ "[CATEGORIES]" }, "");
    parser
        .add_argument({ "-S", "--settings", "--print-settings" },
                      "Display the runtime settings")
        .max_count(1);
    parser
        .add_argument({ "-C", "--components", "--print-components" },
                      "Only display the components data")
        .max_count(1);
    parser
        .add_argument({ "-H", "--hw-counters", "--print-hw-counters" },
                      "Write the available hardware counters")
        .max_count(1);

    parser.add_argument({ "" }, "");
    parser.add_argument({ "[VIEW OPTIONS]" }, "");
    parser.add_argument({ "-A", "--alphabetical" }, "Sort the output alphabetically")
        .max_count(1)
        .action([](parser_t& p) { alphabetical = p.get<bool>("alphabetical"); });
    parser
        .add_argument({ "-r", "--filter" },
                      "Filter the output according to provided regex (egrep + "
                      "case-sensitive) [e.g. -r \"true\"]")
        .count(1)
        .dtype("string")
        .action([](parser_t& p) { regex_key = p.get<std::string>("filter"); });
    parser
        .add_argument({ "--hl", "--highlight" },
                      "Highlight regex matches (only available on UNIX)")
        .max_count(1)
        .action([](parser_t&) { regex_hl = true; });

    parser.add_argument({ "" }, "");
    parser.add_argument({ "[COLUMN OPTIONS]" }, "");
    parser.add_argument({ "-b", "--brief" }, "Suppress availability/value info")
        .max_count(1)
        .action([](parser_t& p) { force_brief = p.get<bool>("brief"); });
    parser.add_argument({ "-d", "--description" }, "Display the component description")
        .max_count(1);
    parser.add_argument()
        .names({ "-e", "--enum" })
        .description("Display the enumeration ID")
        .max_count(1);
    parser
        .add_argument({ "-l", "--language-types" },
                      "Display the language-based alias/accessors")
        .max_count(1);
    parser.add_argument({ "-s", "--string" }, "Display all acceptable string identifiers")
        .max_count(1);
    parser
        .add_argument({ "-v", "--value" },
                      "Display the component data storage value type")
        .max_count(1);
    parser
        .add_argument({ "-f", "--filename" },
                      "Display the output filename for the component")
        .max_count(1);

    parser.add_argument({ "" }, "");
    parser.add_argument({ "[WIDTH OPTIONS]" }, "");
    parser
        .add_argument({ "-w", "--width" },
                      "if w > 0, truncate any columns greater than this width")
        .count(1)
        .dtype("int")
        .action([](parser_t& p) { max_width = p.get<int32_t>("width"); });
    parser
        .add_argument(
            { "-c", "--columns" },
            std::string{ "if c > 0, truncate the total width of all the columns to this "
                         "value. Set '-w 0 -c 0' to remove all truncation" } +
                col_msg)
        .set_default(num_cols)
        .count(1)
        .dtype("int")
        .action([](parser_t& p) { num_cols = p.get<int32_t>("columns"); });

    parser.add_argument({ "" }, "");
    parser.add_argument({ "[OUTPUT OPTIONS]" }, "");
    parser.add_argument({ "-O", "--output" }, "Write results to file")
        .count(1)
        .dtype("filename");
    parser.add_argument({ "-M", "--markdown" }, "Write data in markdown")
        .max_count(1)
        .action([](parser_t& p) { markdown = p.get<bool>("markdown"); });

    parser.add_positional_argument("REGEX_FILTER").set_default(std::string{});

    auto err = parser.parse(argc, argv);

    if(parser.exists("help"))
    {
        parser.print_help();
        return EXIT_SUCCESS;
    }

    if(err)
    {
        std::cerr << err << std::endl;
        parser.print_help();
        return EXIT_FAILURE;
    }

    std::string _pos_regex{};
    if(parser.get_positional_count() > 0)
    {
        err = parser.get("REGEX_FILTER", _pos_regex);
        if(err)
        {
            std::cerr << err << std::endl;
            parser.print_help();
            return EXIT_FAILURE;
        }
    }

    if(regex_key.empty())
        regex_key = _pos_regex;
    else if(!_pos_regex.empty())
    {
        regex_key.append("|" + _pos_regex);
    }

    auto _parser_set_if_exists = [&parser](auto& _var, const std::string& _opt) {
        using Tp = decay_t<decltype(_var)>;
        if(parser.exists(_opt))
            _var = parser.get<Tp>(_opt);
    };

    _parser_set_if_exists(options[FNAME], "filename");
    _parser_set_if_exists(options[DESC], "description");
    _parser_set_if_exists(options[VAL], "value");
    _parser_set_if_exists(options[ENUM], "enum");
    _parser_set_if_exists(options[LANG], "language-types");
    _parser_set_if_exists(options[CID], "string");
    _parser_set_if_exists(file, "output");
    _parser_set_if_exists(include_components, "components");
    _parser_set_if_exists(include_settings, "settings");
    _parser_set_if_exists(include_hw_counters, "hw-counters");

    if(!include_components && !include_settings && !include_hw_counters)
        include_components = true;

    if(markdown || include_hw_counters)
        padding = 6;

    std::ostream* os = nullptr;
    std::ofstream ofs;
    if(!file.empty())
    {
        ofs.open(file.c_str());
        if(ofs)
        {
            os = &ofs;
        }
        else
        {
            std::cerr << "Error opening output file: " << file << std::endl;
        }
    }

    if(!os)
        os = &std::cout;

    if(include_components)
        write_component_info(*os, options, use_mark, fields);

    if(include_settings)
        write_settings_info(*os, { options[VAL], options[LANG], options[DESC] });

    if(include_hw_counters)
        write_hw_counter_info(
            *os, { true, !force_brief, options[LANG], !options[DESC], options[DESC] });

    return 0;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename I>
struct enumerated_list;

template <template <typename...> class TupT, typename... T>
struct enumerated_list<TupT<T...>, index_sequence<>>
{
    using type = type_list<T...>;
};

template <template <typename...> class TupT, size_t I, typename... T, size_t... Idx>
struct enumerated_list<TupT<T...>, index_sequence<I, Idx...>>
{
    using Tp                         = component::enumerator_t<I>;
    static constexpr bool is_nothing = concepts::is_placeholder<Tp>::value;
    using type                       = typename enumerated_list<
        tim::conditional_t<is_nothing, type_list<T...>, type_list<T..., Tp>>,
        index_sequence<Idx...>>::type;
};

//======================================================================================//
//
//                                  COMPONENT INFO
//
//======================================================================================//

template <size_t N>
void
write_component_info(std::ostream& os, const array_t<bool, N>& options,
                     const array_t<bool, N>& _mark, const array_t<string_t, N>& fields)
{
    static_assert(N >= num_component_options,
                  "Error! Too few component options + fields");

    using index_seq_t = make_index_sequence<TIMEMORY_COMPONENTS_END>;
    using enum_list_t = typename enumerated_list<tim::type_list<>, index_seq_t>::type;
    auto _info        = get_availability<enum_list_t>::get_info();

    using width_type = std::vector<int64_t>;
    using width_bool = std::array<bool, N + 2>;

    width_type _widths = width_type{ 30, 12, 20, 20, 20, 40, 20, 40 };
    width_bool _wusing = width_bool{ true, !force_brief };
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

    if(alphabetical)
    {
        std::sort(_info.begin(), _info.end(), [](const auto& lhs, const auto& rhs) {
            return std::get<0>(lhs) < std::get<0>(rhs);
        });
    }

    // compute the widths
    for(const auto& itr : _info)
    {
        {
            std::stringstream ss;
            write_entry(ss, std::get<0>(itr), _widths.at(0) - 1, false, true);
            if(!force_brief)
                write_entry(ss, std::get<1>(itr), _widths.at(1), true, false);
            for(size_t i = 0; i < std::get<2>(itr).size(); ++i)
            {
                if(!options[i])
                    continue;
                bool center = (i > 0) ? false : true;
                write_entry(ss, std::get<2>(itr).at(i), _widths.at(i + 2), center,
                            _mark.at(i));
            }
            if(!not_filtered(ss.str()))
                continue;
        }

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

    _widths = compute_max_columns(_widths, _wusing);

    if(!markdown)
        os << banner(_widths, _wusing, '-');

    os << global_delim << ' ';
    write_entry(os, "COMPONENT", _widths.at(0) - 2, true, false);
    if(!force_brief)
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
        std::stringstream ss;
        write_entry(ss, std::get<0>(itr), _widths.at(0) - 1, false, true);
        if(!force_brief)
            write_entry(ss, std::get<1>(itr), _widths.at(1), true, false);
        for(size_t i = 0; i < std::get<2>(itr).size(); ++i)
        {
            if(!options[i])
                continue;
            bool center = (i > 0) ? false : true;
            write_entry(ss, std::get<2>(itr).at(i), _widths.at(i + 2), center,
                        _mark.at(i));
        }

        if(not_filtered(ss.str()))
        {
            os << global_delim;
            os << hl_filtered(ss.str());
            os << "\n";
        }
    }

    if(!markdown)
        os << banner(_widths, _wusing, '-');

    os << "\n" << std::flush;
    // os << banner(total_width) << std::flush;
}

//======================================================================================//
//
//                                      SETTINGS
//
//======================================================================================//

template <size_t N>
void
write_settings_info(std::ostream& os, const array_t<bool, N>& opts,
                    const array_t<bool, N>&, const array_t<string_t, N>&)
{
    static_assert(N >= num_settings_options, "Error! Too few settings options + fields");

    static constexpr size_t size = 7;
    using archive_type           = cereal::SettingsTextArchive;
    using array_type             = typename archive_type::array_type;
    using unique_set             = typename archive_type::unique_set;
    using width_type             = array_t<int64_t, size>;
    using width_bool             = array_t<bool, size>;

    array_type _setting_output;
    unique_set _settings_exclude = { "TIMEMORY_ENVIRONMENT", "TIMEMORY_COMMAND_LINE",
                                     "cereal_class_version", "settings" };

    cereal::SettingsTextArchive settings_archive(_setting_output, _settings_exclude);
    settings::serialize_settings(settings_archive);

    width_type _widths = { 0, 0, 0, 0, 0, 0, 0 };
    width_bool _wusing = {
        true, !force_brief, opts[0], opts[1], opts[1], opts[1], opts[2]
    };
    width_bool _mark = { false, false, false, true, true, true, false };

    if(alphabetical)
    {
        std::sort(_setting_output.begin(), _setting_output.end(),
                  [](const auto& lhs, const auto& rhs) {
                      return (lhs.find("identifier")->second <
                              rhs.find("identifier")->second);
                  });
    }

    array_t<string_t, size> _labels = {
        "ENVIRONMENT VARIABLE", "VALUE",           "DATA TYPE",  "C++ STATIC ACCESSOR",
        "C++ MEMBER ACCESSOR",  "Python ACCESSOR", "DESCRIPTION"
    };
    array_t<string_t, size> _keys   = { "environ",         "value",
                                      "data_type",       "static_accessor",
                                      "member_accessor", "python_accessor",
                                      "description" };
    array_t<bool, size>     _center = { false, true, true, false, false, false, false };

    std::vector<array_t<string_t, size>> _results;
    for(const auto& itr : _setting_output)
    {
        array_t<string_t, size> _tmp{};
        for(size_t j = 0; j < _keys.size(); ++j)
        {
            auto eitr = itr.find(_keys.at(j));
            if(eitr != itr.end())
                _tmp.at(j) = eitr->second;
        }
        if(!_tmp.at(0).empty())
            _results.push_back(_tmp);
    }

    for(const auto& itr : _results)
    {
        std::stringstream ss;
        for(size_t i = 0; i < itr.size(); ++i)
        {
            if(!_wusing.at(i))
                continue;
            auto _w = _widths.at(i) - ((i == 0) ? 1 : 0);
            write_entry(ss, itr.at(i), _w, _center.at(i), _mark.at(i));
        }

        if(!not_filtered(ss.str()))
            continue;

        for(size_t i = 0; i < itr.size(); ++i)
        {
            _widths.at(i) =
                std::max<uint64_t>(_widths.at(i), itr.at(i).length() + padding);
        }
    }

    _widths = compute_max_columns(_widths, _wusing);

    if(!markdown)
        os << banner(_widths, _wusing, '-');

    os << global_delim;
    for(size_t i = 0; i < _labels.size(); ++i)
    {
        if(!_wusing.at(i))
            continue;
        auto _w = _widths.at(i) - ((i == 0) ? 1 : 0);
        write_entry(os, _labels.at(i), _w, true, false);
    }
    os << "\n" << banner(_widths, _wusing, '-');

    for(const auto& itr : _results)
    {
        std::stringstream ss;
        for(size_t i = 0; i < itr.size(); ++i)
        {
            if(!_wusing.at(i))
                continue;
            auto _w = _widths.at(i) - ((i == 0) ? 1 : 0);
            write_entry(ss, itr.at(i), _w, _center.at(i), _mark.at(i));
        }

        if(not_filtered(ss.str()))
        {
            os << global_delim;
            os << hl_filtered(ss.str());
            os << "\n";
        }
    }

    if(!markdown)
        os << banner(_widths, _wusing, '-');

    os << "\n" << std::flush;
    // os << banner(total_width, '-') << std::flush;
}

//======================================================================================//
//
//                                  HARDWARE COUNTERS
//
//======================================================================================//

template <size_t N>
void
write_hw_counter_info(std::ostream& os, const array_t<bool, N>& options,
                      const array_t<bool, N>&, const array_t<string_t, N>&)
{
    static_assert(N >= num_hw_counter_options,
                  "Error! Too few hw counter options + fields");

    using width_type = array_t<int64_t, N>;
    using width_bool = array_t<bool, N>;

    tim::cupti::device_t device;

#if defined(TIMEMORY_USE_CUPTI)
    TIMEMORY_CUDA_DRIVER_API_CALL(cuInit(0));
    TIMEMORY_CUDA_DRIVER_API_CALL(cuDeviceGet(&device, tim::settings::cupti_device()));
#endif

    auto _cupti_events  = tim::cupti::available_events_info(device);
    auto _cupti_metrics = tim::cupti::available_metrics_info(device);
    auto _papi_events   = tim::papi::available_events_info();

    auto _process_counters = [](auto& _events, int32_t _offset) {
        for(auto& itr : _events)
        {
            itr.offset() += _offset;
            itr.python_symbol() = "timemory.hardware_counters." + itr.python_symbol();
        }
        return static_cast<int32_t>(_events.size());
    };

    int32_t _offset = 0;
    _offset += _process_counters(_papi_events, _offset);
    _offset += _process_counters(_cupti_events, _offset);
    _offset += _process_counters(_cupti_metrics, _offset);

    using hwcounter_info_t = std::vector<tim::hardware_counters::info>;
    auto fields =
        std::vector<hwcounter_info_t>{ _papi_events, _cupti_events, _cupti_metrics };
    auto                 subcategories = std::vector<std::string>{ "CPU", "GPU", "" };
    array_t<string_t, N> _labels = { "HARDWARE COUNTER", "AVAILABLE", "PYTHON", "SUMMARY",
                                     "DESCRIPTION" };
    array_t<bool, N>     _center = { false, true, false, false, false };

    width_type _widths;
    width_bool _wusing;
    width_bool _mark = { false, true, false, false };
    _widths.fill(0);
    _wusing.fill(false);
    for(size_t i = 0; i < _widths.size(); ++i)
    {
        _widths.at(i) = _labels.at(i).length() + padding;
        _wusing.at(i) = options[i];
    }

    for(const auto& fitr : fields)
    {
        for(const auto& itr : fitr)
        {
            width_type _w = { { (int64_t) itr.symbol().length(), (int64_t) 6,
                                (int64_t) itr.python_symbol().length(),
                                (int64_t) itr.short_description().length(),
                                (int64_t) itr.long_description().length() } };
            for(auto& witr : _w)
                witr += padding;

            for(size_t i = 0; i < N; ++i)
                _widths.at(i) = std::max<uint64_t>(_widths.at(i), _w.at(i));
        }
    }

    _widths = compute_max_columns(_widths, _wusing);

    if(!markdown)
        os << banner(_widths, _wusing, '-');
    os << global_delim;

    for(size_t i = 0; i < _labels.size(); ++i)
    {
        auto _w = _widths.at(i) - ((i == 0) ? 1 : 0);
        if(options[i])
            write_entry(os, _labels.at(i), _w, true, false);
    }
    os << "\n" << banner(_widths, _wusing, '-');

    size_t nitr = 0;
    for(const auto& fitr : fields)
    {
        auto idx = nitr++;

        if(idx < subcategories.size())
        {
            if(!markdown && idx != 0)
                os << banner(_widths, _wusing, '-');
            if(subcategories.at(idx).length() > 0)
            {
                os << global_delim;
                if(options[0])
                {
                    write_entry(os, subcategories.at(idx), _widths.at(0) - 1, true,
                                _mark.at(0));
                }
                for(size_t i = 1; i < N; ++i)
                {
                    if(options[i])
                        write_entry(os, "", _widths.at(i), _center.at(i), _mark.at(i));
                }
                os << "\n";
                if(!markdown)
                    os << banner(_widths, _wusing, '-');
            }
        }
        else
        {
            if(!markdown)
                os << banner(_widths, _wusing, '-');
        }

        for(const auto& itr : fitr)
        {
            std::stringstream ss;

            if(options[0])
            {
                write_entry(ss, itr.symbol(), _widths.at(0) - 1, _center.at(0),
                            _mark.at(0));
            }
            if(options[1])
            {
                write_entry(ss, itr.available(), _widths.at(1), _center.at(1),
                            _mark.at(1));
            }

            array_t<string_t, N> _e = { { "", "", itr.python_symbol(),
                                          itr.short_description(),
                                          itr.long_description() } };
            for(size_t i = 2; i < N; ++i)
            {
                if(options[i])
                    write_entry(ss, _e.at(i), _widths.at(i), _center.at(i), _mark.at(i));
            }

            if(not_filtered(ss.str()))
            {
                os << global_delim;
                os << hl_filtered(ss.str());
                os << "\n";
            }
        }
    }

    if(!markdown)
        os << banner(_widths, _wusing, '-');

    os << "\n" << std::flush;
}

//======================================================================================//

struct unknown
{};

template <typename T, typename U = typename T::value_type>
constexpr bool
available_value_type_alias(int)
{
    return true;
}

template <typename T, typename U = unknown>
constexpr bool
available_value_type_alias(long)
{
    return false;
}

template <typename Type, bool>
struct component_value_type;

template <typename Type>
struct component_value_type<Type, true>
{
    using type = typename Type::value_type;
};

template <typename Type>
struct component_value_type<Type, false>
{
    using type = unknown;
};

template <typename Type>
using component_value_type_t =
    typename component_value_type<Type, available_value_type_alias<Type>(0)>::type;

//--------------------------------------------------------------------------------------//

template <typename Type>
info_type
get_availability<Type>::get_info()
{
    using value_type = component_value_type_t<Type>;

    auto _cleanup = [](std::string _type, const std::string& _pattern) {
        auto _pos = std::string::npos;
        while((_pos = _type.find(_pattern)) != std::string::npos)
            _type.erase(_pos, _pattern.length());
        return _type;
    };
    auto _replace = [](std::string _type, const std::string& _pattern,
                       const std::string& _with) {
        auto _pos = std::string::npos;
        while((_pos = _type.find(_pattern)) != std::string::npos)
            _type.replace(_pos, _pattern.length(), _with);
        return _type;
    };

    bool has_metadata   = metadata_t::specialized();
    bool has_properties = property_t::specialized();
    bool is_available   = trait::is_available<Type>::value;
    bool file_output    = trait::generates_output<Type>::value;
    auto name           = component::metadata<Type>::name();
    auto label          = (file_output)
                     ? ((has_metadata) ? metadata_t::label() : Type::get_label())
                     : std::string("");
    auto description =
        (has_metadata) ? metadata_t::description() : Type::get_description();
    auto     data_type = demangle<value_type>();
    string_t enum_type = property_t::enum_string();
    string_t id_type   = property_t::id();
    auto     ids_set   = property_t::ids();
    if(!has_properties)
    {
        enum_type = "";
        id_type   = "";
        ids_set.clear();
    }
    auto     itr = ids_set.begin();
    string_t db  = (markdown) ? "`\"" : "\"";
    string_t de  = (markdown) ? "\"`" : "\"";
    if(has_metadata)
        description += ". " + metadata_t::extra_description();
    while(itr->empty())
        ++itr;
    string_t ids_str = {};
    if(itr != ids_set.end())
        ids_str = TIMEMORY_JOIN("", TIMEMORY_JOIN("", db, *itr++, de));
    for(; itr != ids_set.end(); ++itr)
    {
        if(!itr->empty())
            ids_str = TIMEMORY_JOIN("  ", ids_str, TIMEMORY_JOIN("", db, *itr, de));
    }

    data_type = _replace(_cleanup(data_type, "::__1"), "> >", ">>");
    return info_type{ name, is_available,
                      str_vec_t{ data_type, enum_type, id_type, ids_str, label,
                                 description } };
}

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

std::tuple<int32_t, std::string>
get_window_columns()
{
    using return_type = std::tuple<int32_t, std::string>;
#if defined(TIMEMORY_UNIX)
    struct winsize size;
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &size);
    return return_type{ size.ws_col, "ioctl" };
#elif defined(TIMEMORY_WINDOWS)
    CONSOLE_SCREEN_BUFFER_INFO csbi;
    GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi);
    return return_type{ csbi.srWindow.Right - csbi.srWindow.Left + 1,
                        "GetConsoleScreenBufferInfo" };
#else
    return return_type{ 0, "none" };
#endif
}

//--------------------------------------------------------------------------------------//

template <typename IntArrayT, typename BoolArrayT>
IntArrayT
compute_max_columns(IntArrayT _widths, BoolArrayT _using)
{
    using value_type = typename IntArrayT::value_type;

    if(num_cols == 0)
        return _widths;

    auto _get_sum = [&]() {
        value_type _sumv = 0;
        for(size_t i = 0; i < _widths.size(); ++i)
            if(_using.at(i))
                _sumv += _widths.at(i);
        return _sumv;
    };
    auto _get_max = [&]() {
        auto       _midx = _widths.size();
        value_type _maxv = 0;
        for(size_t i = 0; i < _widths.size(); ++i)
        {
            if(_using.at(i) && _widths.at(i) > _maxv)
            {
                _midx = i;
                _maxv = _widths.at(i);
            }
        }

        if(_maxv <= min_width)
        {
            _midx = _widths.size();
            _maxv = min_width;
        }
        return std::make_pair(_midx, _maxv);
    };
    auto _decrement_max = [&]() {
        auto _midx = _get_max().first;
        if(_midx < _widths.size())
            _widths.at(_midx) -= 1;
    };

    int32_t _max_width = num_cols;
    size_t  _n         = 0;
    size_t  _nmax      = std::numeric_limits<uint16_t>::max();
    while(_n++ < _nmax)
    {
        if(debug_msg)
        {
            std::stringstream _msg;
            for(size_t i = 0; i < _widths.size(); ++i)
                _msg << ", " << ((_using.at(i)) ? _widths.at(i) : 0);
            std::cout << "[temp]>  sum_width = " << _get_sum()
                      << ", max_width = " << _max_width
                      << ", widths = " << _msg.str().substr(2) << std::endl;
        }

        if(_get_max().first == _widths.size() || _get_sum() <= _max_width)
            break;
        _decrement_max();
    }

    int32_t _maxw = _get_max().second;
    if(max_width == 0 || _maxw < max_width)
        max_width = _maxw;

    if(debug_msg)
    {
        std::stringstream _msg;
        for(size_t i = 0; i < _widths.size(); ++i)
            _msg << ", " << ((_using.at(i)) ? _widths.at(i) : 0);
        std::cout << "[final]> sum_width = " << _get_sum()
                  << ", max_width = " << _max_width
                  << ", widths = " << _msg.str().substr(2)
                  << ", column max width = " << max_width << std::endl;
    }

    return _widths;
}

//--------------------------------------------------------------------------------------//

template <typename Tp>
void
write_entry(std::ostream& os, const Tp& _entry, int64_t _w, bool center, bool mark)
{
    if(max_width > 0 && _w > max_width)
        _w = max_width;

    stringstream_t ssentry;
    stringstream_t ssbeg;
    stringstream_t ss;
    ssentry << std::boolalpha << _entry;
    auto _sentry = remove(ssentry.str(), { "tim::", "component::" });
    if(max_width > 0 && _sentry.length() > static_cast<size_t>(max_width - 2))
        _sentry = _sentry.substr(0, max_width - 5) + "...";
    auto wbeg = (_w / 2) - (_sentry.length() / 2) - 1;
    if(!center)
        wbeg = 1;
    ssbeg << std::setw(wbeg) << "";
    if(mark && markdown)
    {
        ssbeg << '`' << _sentry << '`';
    }
    else
    {
        ssbeg << _sentry;
    }

    ss << std::left << std::setw(_w - 1) << ssbeg.str() << global_delim;
    os << ss.str();
}

//--------------------------------------------------------------------------------------//

template <typename IntArrayT, size_t N>
string_t
banner(IntArrayT _breaks, std::array<bool, N> _use, char filler, char delim)
{
    _breaks = compute_max_columns(_breaks, _use);

    for(auto& itr : _breaks)
    {
        if(max_width > 0 && itr > max_width)
            itr = max_width;
    }

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
    {
        printf("[banner]> non-zero remainder: %i with total: %i\n", (int) _remain,
               (int) _total);
    }
    return ss.str();
}

//--------------------------------------------------------------------------------------//

bool
not_filtered(const std::string& line)
{
    bool _display = regex_key.empty();
    if(!_display)
    {
        const auto _rc = regex_const::egrep | regex_const::optimize;
        const auto _re = std::regex(regex_key, _rc);
        _display       = std::regex_search(line, _re);
    }
    return _display;
}

//--------------------------------------------------------------------------------------//

std::string
hl_filtered(std::string line)
{
#if defined(TIMEMORY_UNIX)
    if(regex_hl)
    {
        const auto _rc = regex_const::egrep | regex_const::optimize;
        const auto _re = std::regex(regex_key, _rc);
        line           = std::regex_replace(line, _re, "\33[01;04;36;40m$&\33[0m");
    }
#endif
    return line;
}

//--------------------------------------------------------------------------------------//

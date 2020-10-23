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

using namespace tim;

using string_t       = std::string;
using stringstream_t = std::stringstream;
using str_vec_t      = std::vector<string_t>;
using info_type      = std::tuple<string_t, bool, str_vec_t>;

template <typename Tp, size_t N>
using array_t = std::array<Tp, N>;

char global_delim = '|';
bool markdown     = false;
bool alphabetical = false;
bool all_info     = false;
bool force_brief  = false;
int  padding      = 4;

//--------------------------------------------------------------------------------------//

struct unknown
{};

//--------------------------------------------------------------------------------------//

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
    typename component_value_type<Type, tim::trait::is_available<Type>::value>::type;

//--------------------------------------------------------------------------------------//

template <typename Type>
struct get_availability
{
    using this_type  = get_availability<Type>;
    using value_type = component_value_type_t<Type>;

    static info_type get_info()
    {
        bool     is_available = trait::is_available<Type>::value;
        bool     file_output  = trait::generates_output<Type>::value;
        auto     name         = demangle<Type>();
        auto     label        = (file_output) ? Type::get_label() : std::string("");
        auto     description  = Type::get_description();
        auto     data_type    = demangle<value_type>();
        string_t enum_type    = component::properties<Type>::enum_string();
        string_t id_type      = component::properties<Type>::id();
        auto     ids_set      = component::properties<Type>::ids();
        auto     itr          = ids_set.begin();
        string_t db           = (markdown) ? "`\"" : "\"";
        string_t de           = (markdown) ? "\"`" : "\"";
        while(itr->empty())
            ++itr;
        string_t ids_str = "";
        if(itr != ids_set.end())
            ids_str = TIMEMORY_JOIN("", TIMEMORY_JOIN("", db, *itr++, de));
        for(; itr != ids_set.end(); ++itr)
        {
            if(!itr->empty())
                ids_str = TIMEMORY_JOIN("  ", ids_str, TIMEMORY_JOIN("", db, *itr, de));
        }

        return info_type{ name, is_available,
                          str_vec_t{ data_type, enum_type, id_type, ids_str, label,
                                     description } };
    }

    explicit get_availability(info_type& _info) { _info = this_type::get_info(); }
};

//--------------------------------------------------------------------------------------//

template <typename... Types>
struct get_availability<component_list<Types...>>
{
    using this_type = get_availability<component_list<Types...>>;

    static constexpr size_t size() { return sizeof...(Types); }
    static constexpr size_t nelem = sizeof...(Types);

    using info_vec_t  = array_t<info_type, nelem>;
    using avail_types = std::tuple<get_availability<Types>...>;

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

template <typename Tp>
void
write_entry(std::ostream& os, const Tp& _entry, int64_t _w, bool center, bool mark)
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

template <typename IntArrayT, size_t N>
string_t
banner(IntArrayT _breaks, std::array<bool, N> _use, char filler = '-', char delim = '|')
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

static constexpr size_t num_component_options  = 6;
static constexpr size_t num_settings_options   = 3;
static constexpr size_t num_hw_counter_options = 5;

template <size_t N = num_component_options>
void
write_component_info(std::ostream&, const array_t<bool, N>&, const array_t<bool, N>&,
                     const array_t<string_t, N>&);

template <size_t N = num_settings_options>
void
write_settings_info(std::ostream&, const array_t<bool, N>& = array_t<bool, N>{},
                    const array_t<bool, N>&     = array_t<bool, N>{},
                    const array_t<string_t, N>& = array_t<string_t, N>{});

template <size_t N = num_hw_counter_options>
void
write_hw_counter_info(std::ostream&, const array_t<bool, N>& = array_t<bool, N>{},
                      const array_t<bool, N>&     = array_t<bool, N>{},
                      const array_t<string_t, N>& = array_t<string_t, N>{});

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

    std::string file = "";

    tim::argparse::argument_parser parser("timemory-avail");

    parser.enable_help();
    parser.add_argument({ "-a", "--all" }, "Print all available info");
    parser.add_argument({ "-A", "--alphabetical" }, "Sort the output alphabetically");
    parser
        .add_argument({ "-d", "--description" },
                      "Output the description for the component")
        .count(0);
    parser.add_argument()
        .names({ "-e", "--enum" })
        .description("Display the enumeration ID")
        .count(0);
    parser.add_argument({ "-f", "--filename" }, "Output the filename for the component")
        .count(0);
    parser
        .add_argument({ "-l", "--language-types" },
                      "Display the language-based alias/accessors")
        .count(0);
    parser.add_argument({ "-s", "--string" }, "Display all acceptable string identifiers")
        .count(0);
    parser.add_argument({ "-v", "--value" }, "Output the value type for the component")
        .count(0);
    parser.add_argument({ "-b", "--brief" }, "Suppress availibility/value info").count(0);
    parser.add_argument({ "-S", "--settings" }, "Display the runtime settings").count(0);
    parser.add_argument({ "-C", "--components" }, "Only display the components data")
        .count(0);
    parser.add_argument({ "-M", "--markdown" }, "Write data in markdown").count(0);
    parser.add_argument({ "-H", "--hw-counters" },
                        "Write the available hardware counters");
    parser.add_argument({ "-O", "--output" }, "Write results to file").count(1);

    auto err = parser.parse(argc, argv);
    if(err)
        std::cerr << err << std::endl;

    if(err || parser.exists("help"))
    {
        parser.print_help();
        return EXIT_FAILURE;
    }

    if(parser.exists("all"))
        all_info = true;

    if(parser.exists("alphabetical"))
        alphabetical = true;

    if(parser.exists("filename"))
        options[FNAME] = !options[FNAME];

    if(parser.exists("description"))
        options[DESC] = !options[DESC];

    if(parser.exists("value"))
        options[VAL] = !options[VAL];

    if(parser.exists("enum"))
        options[ENUM] = !options[ENUM];

    if(parser.exists("brief"))
        force_brief = true;

    if(parser.exists("language-types"))
        options[LANG] = !options[LANG];

    if(parser.exists("string"))
        options[CID] = !options[CID];

    if(parser.exists("output"))
        file = parser.get<std::string>("output");

    if(parser.exists("components"))
        include_components = true;

    if(parser.exists("settings"))
    {
        include_settings = true;
    }

    if(parser.exists("markdown"))
    {
        markdown = true;
        padding  = 6;
    }

    if(parser.exists("hw-counters"))
    {
        include_hw_counters = true;
        padding             = 6;
    }

    if(!include_components && !include_settings && !include_hw_counters)
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

    if(all_info)
    {
        for(auto& itr : options)
            itr = true;
        include_components  = true;
        include_settings    = true;
        include_hw_counters = true;
    }

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

template <int I>
using make_int_sequence = std::make_integer_sequence<int, I>;

template <int... Ints>
using int_sequence = std::integer_sequence<int, Ints...>;

template <typename T, typename I>
struct enumerated_list;

template <template <typename...> class TupT, typename... T>
struct enumerated_list<TupT<T...>, int_sequence<>>
{
    using type = tim::component_list<T...>;
};

using tim::component::nothing;
using tim::component::placeholder;

template <template <typename...> class TupT, int I, typename... T, int... Idx>
struct enumerated_list<TupT<T...>, int_sequence<I, Idx...>>
{
    using Tp                         = component::enumerator_t<I>;
    static constexpr bool is_nothing = std::is_same<Tp, placeholder<nothing>>::value;
    using type                       = typename enumerated_list<
        tim::conditional_t<(is_nothing), tim::component_list<T...>,
                           tim::component_list<T..., Tp>>,
        int_sequence<Idx...>>::type;
};

//--------------------------------------------------------------------------------------//

template <size_t N>
void
write_component_info(std::ostream& os, const array_t<bool, N>& options,
                     const array_t<bool, N>& _mark, const array_t<string_t, N>& fields)
{
    static_assert(N >= num_component_options,
                  "Error! Too few component options + fields");

    using seq_t       = make_int_sequence<TIMEMORY_COMPONENTS_END>;
    using enum_list_t = typename enumerated_list<tim::type_list<>, seq_t>::type;
    auto _info        = get_availability<enum_list_t>::get_info();

    using int_vec_t  = std::vector<int64_t>;
    using width_type = int_vec_t;
    using width_bool = std::array<bool, N + 2>;

    width_type _widths = width_type{ 40, 12, 20, 20, 20, 40, 20, 40 };
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
        os << global_delim;
        write_entry(os, std::get<0>(itr), _widths.at(0) - 1, false, true);
        if(!force_brief)
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

template <size_t N>
void
write_settings_info(std::ostream& os, const array_t<bool, N>& opts,
                    const array_t<bool, N>&, const array_t<string_t, N>&)
{
    static_assert(N >= num_settings_options, "Error! Too few settings options + fields");

    using archive_type = cereal::SettingsTextArchive;
    using array_type   = typename archive_type::array_type;
    using unique_set   = typename archive_type::unique_set;
    using width_type   = array_t<int64_t, 5>;
    using width_bool   = array_t<bool, 5>;

    array_type _setting_output;
    unique_set _settings_exclude = { "TIMEMORY_ENVIRONMENT", "TIMEMORY_COMMAND_LINE",
                                     "cereal_class_version", "settings" };

    cereal::SettingsTextArchive settings_archive(_setting_output, _settings_exclude);
    settings::serialize_settings(settings_archive);

    width_type _widths = { 0, 0, 0, 0, 0 };
    width_bool _wusing = { true, !force_brief, opts[0], opts[1], opts[2] };
    width_bool _mark   = { false, false, false, true, false };

    if(alphabetical)
    {
        std::sort(_setting_output.begin(), _setting_output.end(),
                  [](const auto& lhs, const auto& rhs) {
                      return (lhs.find("identifier")->second <
                              rhs.find("identifier")->second);
                  });
    }

    array_t<string_t, 5> _labels = { "ENVIRONMENT VARIABLE", "VALUE", "DATA TYPE",
                                     "C++ STATIC ACCESSOR", "DESCRIPTION" };
    array_t<string_t, 5> _keys   = { "environ", "value", "data_type", "accessor",
                                   "description" };
    array_t<bool, 5>     _center = { false, true, true, false, false };

    std::vector<array_t<string_t, 5>> _results;
    for(const auto& itr : _setting_output)
    {
        array_t<string_t, 5> _tmp{};
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
        for(size_t i = 0; i < itr.size(); ++i)
            _widths.at(i) =
                std::max<uint64_t>(_widths.at(i), itr.at(i).length() + padding);

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
        os << global_delim;
        for(size_t i = 0; i < itr.size(); ++i)
        {
            if(!_wusing.at(i))
                continue;
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

//--------------------------------------------------------------------------------------//

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
                    write_entry(os, subcategories.at(idx), _widths.at(0) - 1, true,
                                _mark.at(0));
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
            os << global_delim;

            if(options[0])
                write_entry(os, itr.symbol(), _widths.at(0) - 1, _center.at(0),
                            _mark.at(0));
            if(options[1])
                write_entry(os, itr.available(), _widths.at(1), _center.at(1),
                            _mark.at(1));

            array_t<string_t, N> _e = { { "", "", itr.python_symbol(),
                                          itr.short_description(),
                                          itr.long_description() } };
            for(size_t i = 2; i < N; ++i)
            {
                if(options[i])
                    write_entry(os, _e.at(i), _widths.at(i), _center.at(i), _mark.at(i));
            }

            os << "\n";
        }
    }

    if(!markdown)
        os << banner(_widths, _wusing, '-');

    os << "\n" << std::flush;
}

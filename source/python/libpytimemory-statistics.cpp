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

#if !defined(TIMEMORY_PYSTATISTICS_SOURCE)
#    define TIMEMORY_PYSTATISTICS_SOURCE
#endif

#include "libpytimemory-components.hpp"
#include "timemory/components/types.hpp"
#include "timemory/data/statistics.hpp"

namespace pystatistics
{
//
template <typename Tp>
struct valid_statistics_type
{
    using invalid_types = tim::type_list<tim::true_type, tim::false_type, tim::null_type,
                                         void, bool, std::tuple<>, tim::type_list<>>;
    static constexpr bool value =
        tim::trait::is_available<Tp>::value && !tim::is_one_of<Tp, invalid_types>::value;
};
//
template <size_t Idx>
struct statistics_type
{
    using type = typename tim::trait::statistics<tim::component::enumerator_t<Idx>>::type;
};
//
template <size_t Idx>
using statistics_type_t = typename statistics_type<Idx>::type;
//
static inline std::string
get_class_name(std::string id)
{
    auto pos = std::string::npos;
    for(auto&& itr : { " ", ":", ",", "<", ">" })
        while((pos = id.find(itr)) != std::string::npos)
            id = id.replace(pos, 1, "_");

    while((pos = id.find("__")) != std::string::npos)
        id = id.replace(pos, 2, "_");

    while((pos = id.find("std_1_")) != std::string::npos)
        id = id.replace(pos, 6, "std_");

    while(id.back() == '_')
        id = id.substr(0, id.length() - 1);

    return id;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
auto
construct(py::module& _pymod, int, tim::enable_if_t<valid_statistics_type<Tp>::value> = 0)
    -> decltype(tim::demangle<Tp>(), void())
{
    using statistics_type = tim::statistics<Tp>;
    using pystatistics_type = py::class_<statistics_type>;

    static std::set<std::string> _created{};
    auto                         _type = tim::demangle<Tp>();
    auto              _name = get_class_name(std::string("Statistics_") + _type);
    std::stringstream _desc;
    _desc << "Statistics class for " << _type << " data type";

    if(_created.count(_name) > 0)
        return;
    _created.insert(_name);

    // PRINT_HERE("Constructing %s", _name.c_str());

    pystatistics_type _pystat(_pymod, _name.c_str(), _desc.str().c_str());
    statistics_type::construct(tim::project::python{}, _pystat);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
void
construct(py::module&, long)
{}
//
//--------------------------------------------------------------------------------------//
//
template <size_t... Idx>
constexpr auto construct(std::index_sequence<Idx...>)
{
    return tim::type_list<statistics_type_t<Idx>...>{};
}
//
//--------------------------------------------------------------------------------------//
//
template <typename... Tp>
auto
construct(py::module& _pymod, tim::type_list<Tp...>)
{
    TIMEMORY_FOLD_EXPRESSION(construct<Tp>(_pymod, 0));
}
//
//--------------------------------------------------------------------------------------//
//
void
generate(py::module& _pymod)
{
    auto _types = construct(std::make_index_sequence<TIMEMORY_NATIVE_COMPONENTS_END>{});
    // std::cerr << "Types: " << tim::demangle<decltype(_types)>() << std::endl;
    construct(_pymod, _types);
    (void) _types;
    (void) _pymod;
}
//
}  // namespace pystatistics

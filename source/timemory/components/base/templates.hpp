//  MIT License
//
//  Copyright (c) 2020, The Regents of the University of California,
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

/**
 * \headerfile "timemory/components/base/templates.hpp"
 * \brief Defines the template functions for the static polymorphic base for the
 * components
 *
 */

#pragma once

#include "timemory/components/base/declaration.hpp"
#include "timemory/components/base/types.hpp"
#include "timemory/mpl/math.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/operations/types.hpp"
#include "timemory/operations/types/add_secondary.hpp"
#include "timemory/operations/types/add_statistics.hpp"
#include "timemory/operations/types/base_printer.hpp"
#include "timemory/operations/types/serialization.hpp"
#include "timemory/storage/declaration.hpp"
#include "timemory/units.hpp"
#include "timemory/utility/serializer.hpp"

#include <cassert>

//======================================================================================//
//
namespace tim
{
namespace component
{
//
//======================================================================================//
//
//                              NON-VOID BASE
//
//======================================================================================//
//
template <typename Tp, typename Value>
template <typename Archive, typename Up,
          enable_if_t<!(trait::custom_serialization<Up>::value), int>>
void
base<Tp, Value>::CEREAL_LOAD_FUNCTION_NAME(Archive& ar, const unsigned int)
{
    // clang-format off
        ar(cereal::make_nvp("is_transient", is_transient),
           cereal::make_nvp("laps", laps),
           cereal::make_nvp("value", value),
           cereal::make_nvp("accum", accum),
           cereal::make_nvp("last", last));
    // clang-format on
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename Value>
template <typename Archive, typename Up,
          enable_if_t<!(trait::custom_serialization<Up>::value), int>>
void
base<Tp, Value>::CEREAL_SAVE_FUNCTION_NAME(Archive& ar, const unsigned int version) const
{
    operation::serialization<Type>(static_cast<const Type&>(*this), ar, version);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename Value>
template <typename Vp, typename Up, enable_if_t<(trait::sampler<Up>::value), int>>
void
base<Tp, Value>::add_sample(Vp&& _obj)
{
    auto _storage = static_cast<storage_type*>(get_storage());
    assert(_storage != nullptr);
    if(_storage)
        _storage->add_sample(std::forward<Vp>(_obj));
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename Value>
template <typename Up, enable_if_t<(trait::base_has_accum<Up>::value), int>>
const Value&
base<Tp, Value>::load() const
{
    return (is_transient) ? accum : value;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename Value>
template <typename Up, enable_if_t<!(trait::base_has_accum<Up>::value), int>>
const Value&
base<Tp, Value>::load() const
{
    return value;
}
//
//--------------------------------------------------------------------------------------//
//
//          Units, display units, width, precision, labels, display labels
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename Value>
template <typename Up, typename Unit,
          enable_if_t<(std::is_same<Unit, int64_t>::value), int>>
int64_t
base<Tp, Value>::unit()
{
    if(timing_units_v)
        return units::sec;
    else if(memory_units_v)
        return units::megabyte;
    else if(percent_units_v)
        return 1;

    return 1;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename Value>
template <typename Up, typename Unit,
          enable_if_t<(std::is_same<Unit, std::string>::value), int>>
std::string
base<Tp, Value>::display_unit()
{
    if(timing_units_v)
        return units::time_repr(unit());
    else if(memory_units_v)
        return units::mem_repr(unit());
    else if(percent_units_v)
        return "%";

    return "";
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename Value>
template <typename Up, typename Unit,
          enable_if_t<(std::is_same<Unit, int64_t>::value), int>>
int64_t
base<Tp, Value>::get_unit()
{
    static int64_t _instance = []() {
        auto _value = Type::unit();
        if(timing_units_v && settings::timing_units().length() > 0)
            _value = std::get<1>(units::get_timing_unit(settings::timing_units()));
        else if(memory_units_v && settings::memory_units().length() > 0)
            _value = std::get<1>(units::get_memory_unit(settings::memory_units()));
        return _value;
    }();
    return _instance;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename Value>
template <typename Up, typename Unit,
          enable_if_t<(std::is_same<Unit, std::string>::value), int>>
std::string
base<Tp, Value>::get_display_unit()
{
    static std::string _instance = Type::display_unit();

    if(timing_units_v && settings::timing_units().length() > 0)
        _instance = std::get<0>(units::get_timing_unit(settings::timing_units()));
    else if(memory_units_v && settings::memory_units().length() > 0)
        _instance = std::get<0>(units::get_memory_unit(settings::memory_units()));

    return _instance;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename Value>
template <typename Up, typename Vp, enable_if_t<(implements_storage<Up, Vp>::value), int>>
void
base<Tp, Value>::print(std::ostream& os) const
{
    operation::base_printer<Up>(os, static_cast<const Up&>(*this));
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename Value>
template <typename Up, typename Vp,
          enable_if_t<!(implements_storage<Up, Vp>::value), int>>
void
base<Tp, Value>::print(std::ostream&) const
{}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace component
//
//----------------------------------------------------------------------------------//
//
}  // namespace tim
//
//======================================================================================//

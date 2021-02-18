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

#include "timemory/mpl/concepts.hpp"
#include "timemory/utility/types.hpp"

#include <ostream>

namespace tim
{
namespace quirk
{
//
template <typename... Types>
struct config
{
    using type       = type_list<Types...>;
    using value_type = void;

    friend std::ostream& operator<<(std::ostream& _os, const config&) { return _os; }
};

/// \struct tim::quirk::config
/// \brief a variadic type which holds zero or more quirks that are passed to the
/// constructor of a component bundler.
/// \code{.cpp}
/// namespace quirk = tim::quirk;
/// using foo_t = tim::component_tuple<wall_clock>;
///
/// foo_t f("example", quirk::config<quirk::auto_start, quirk::flat_scope>{});
/// ...
/// f.stop();
/// \endcode
template <typename... Types>
struct config;

template <typename T>
struct is_config : false_type
{};

template <typename... Types>
struct is_config<config<Types...>> : true_type
{};

/// \struct tim::quirk::auto_start
/// \brief A dummy type intended to be included in a \ref tim::quirk::config object
/// and passed to the constructor of a component bundler or included as template parameter
/// of the component bundle. It will cause non-auto bundlers
/// to invoke start() during construction. If included as a template parameter of the
/// bundler, it will have no effect.
struct auto_start : concepts::quirk_type
{};

/// \struct tim::quirk::auto_stop
/// \brief A dummy type intended to be included in a \ref tim::quirk::config object
/// and passed to the constructor of a component bundler or included as template parameter
/// of the component bundle. It will cause non-auto bundlers
/// to invoke stop() during destruction. If included as a template parameter of the
/// bundler, it will have no effect.
struct auto_stop : concepts::quirk_type
{};

/// \struct tim::quirk::explicit_start
/// \brief A dummy type intended to be included in a \ref tim::quirk::config object
/// and passed to the constructor of a component bundler or included as template parameter
/// of the component bundle. It will cause auto bundlers
/// to suppress calling start during construction. If included as a template parameter of
/// the bundler, it will have no effect.
struct explicit_start : concepts::quirk_type
{};

/// \struct tim::quirk::explicit_stop
/// \brief A dummy type intended to be included in a \ref tim::quirk::config object
/// and passed to the constructor of a component bundler or included as template parameter
/// of the component bundle. It will cause auto bundlers
/// to suppress calling stop during destruction. If included as a template parameter of
/// the bundler, it will have no effect.
struct explicit_stop : concepts::quirk_type
{};

/// \struct tim::quirk::explicit_push
/// \brief A dummy type intended to be included in a \ref tim::quirk::config object
/// and included as template parameter of the component bundler. It will suppress the
/// implicit `push()` within `start()` for the bundlers with this characteristic
struct explicit_push : concepts::quirk_type
{};

/// \struct tim::quirk::explicit_pop
/// \brief A dummy type intended to be included in a \ref tim::quirk::config object
/// and included as template parameter of the component bundler. It will suppress the
/// implicit `pop()` within `stop()` for the bundlers with this characteristic
struct explicit_pop : concepts::quirk_type
{};

/// \struct tim::quirk::exit_report
/// \brief A dummy type intended to be included in a \ref tim::quirk::config object
/// and passed to the constructor of a component bundler or included as template parameter
/// of the component bundle. It will cause bundlers
/// to write itself to stdout during destruction. If included as a template parameter of
/// the bundler, it will have no effect.
struct exit_report : concepts::quirk_type
{};

/// \struct tim::quirk::no_init
/// \brief A dummy type intended to be included in a \ref tim::quirk::config object
/// and passed to the constructor of a component bundler or included as template parameter
/// of the component bundle. It will cause bundlers
/// to suppress calling any routines related to initializing storage during construction.
struct no_init : concepts::quirk_type
{};

/// \struct tim::quirk::no_store
/// \brief A dummy type intended to be included in a \ref tim::quirk::config object
/// and passed to the constructor of a component bundler or included as template parameter
/// of the component bundle. It will cause bundlers
/// to suppress any implicit entries into the component storage. This
/// behavior is the default for tim::lightweight_bundle and is meaningless in that
/// context.
struct no_store : concepts::quirk_type
{};

/// \struct tim::quirk::tree_scope
/// \brief A dummy type intended to be included in a \ref tim::quirk::config object
/// and passed to the constructor of a component bundler or included as template parameter
/// of the component bundle. It will cause bundlers to ignore the global settings and
/// enforce hierarchical storage.
struct tree_scope
: scope::tree
, concepts::quirk_type
{};

/// \struct tim::quirk::flat_scope
/// \brief A dummy type intended to be included in a \ref tim::quirk::config object
/// and passed to the constructor of a component bundler or included as template parameter
/// of the component bundle. It will cause bundlers to ignore the global settings and
/// enforce flat storage.
struct flat_scope
: scope::flat
, concepts::quirk_type
{};

/// \struct tim::quirk::timeline_scope
/// \brief A dummy type intended to be included in a \ref tim::quirk::config object
/// and passed to the constructor of a component bundler or included as template parameter
/// of the component bundle. It will cause bundlers to ignore the global settings and
/// enforce timeline storage.
struct timeline_scope
: scope::timeline
, concepts::quirk_type
{};
//

/// \struct tim::quirk::unsafe
/// \brief A dummy type intended to be included in a \ref tim::quirk::config object
/// and passed to the constructor of a component bundler, included as template parameter
/// of the component bundle, or as the first argument of bundler member function. When
/// present, this argument instructs to skip any safety checks. Example checks include:
/// checking whether a component is in a stop state before startings, checking whether a
/// component has been started before stopping, checking whether push/pop has been applied
/// before applying the inverse
struct unsafe : concepts::quirk_type
{};
//
}  // namespace quirk
}  // namespace tim

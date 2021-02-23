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

#ifndef TIMEMORY_OPERATIONS_TYPES_SERIALIZATION_CPP_
#define TIMEMORY_OPERATIONS_TYPES_SERIALIZATION_CPP_ 1

#if !defined(TIMEMORY_OPERATIONS_TYPES_SERIALIZATION_HPP_)
#    include "timemory/operations/types/serialization.hpp"
#    include "timemory/tpls/cereal/archives.hpp"
#endif

namespace tim
{
namespace operation
{
namespace internal
{
template <typename Tp>
serialization<Tp, true>::serialization(const Tp&                         obj,
                                       cereal::MinimalJSONOutputArchive& ar,
                                       const unsigned int                version, ...)
{
    impl(obj, ar, version);
}

template <typename Tp>
void
serialization<Tp, true>::operator()(const Tp& obj, cereal::MinimalJSONOutputArchive& ar,
                                    const unsigned int version, ...) const
{
    impl(obj, ar, version);
}

template <typename Tp>
void
serialization<Tp, true>::operator()(cereal::MinimalJSONOutputArchive& ar, metadata,
                                    ...) const
{
    impl(ar, metadata{});
}

template <typename Tp>
void
serialization<Tp, true>::operator()(cereal::MinimalJSONOutputArchive& ar,
                                    const basic_tree_vector_type&     data, ...) const
{
    impl(ar, data);
}

template <typename Tp>
void
serialization<Tp, true>::operator()(cereal::MinimalJSONOutputArchive&          ar,
                                    const std::vector<basic_tree_vector_type>& data,
                                    ...) const
{
    impl(ar, data);
}

template <typename Tp>
void
serialization<Tp, true>::operator()(cereal::MinimalJSONOutputArchive& ar,
                                    const basic_tree_map_type&        data, ...) const
{
    impl(ar, data);
}

template <typename Tp>
void
serialization<Tp, true>::operator()(cereal::MinimalJSONOutputArchive& ar,
                                    const result_type&                data, ...) const
{
    impl(ar, data);
}

template <typename Tp>
void
serialization<Tp, true>::operator()(cereal::MinimalJSONOutputArchive& ar,
                                    const distrib_type&               data, ...) const
{
    impl(ar, data);
}

template <typename Tp>
serialization<Tp, true>::serialization(const Tp& obj, cereal::PrettyJSONOutputArchive& ar,
                                       const unsigned int version, ...)
{
    impl(obj, ar, version);
}

template <typename Tp>
void
serialization<Tp, true>::operator()(const Tp& obj, cereal::PrettyJSONOutputArchive& ar,
                                    const unsigned int version, ...) const
{
    impl(obj, ar, version);
}

template <typename Tp>
void
serialization<Tp, true>::operator()(cereal::PrettyJSONOutputArchive& ar, metadata,
                                    ...) const
{
    impl(ar, metadata{});
}

template <typename Tp>
void
serialization<Tp, true>::operator()(cereal::PrettyJSONOutputArchive& ar,
                                    const basic_tree_vector_type&    data, ...) const
{
    impl(ar, data);
}

template <typename Tp>
void
serialization<Tp, true>::operator()(cereal::PrettyJSONOutputArchive&           ar,
                                    const std::vector<basic_tree_vector_type>& data,
                                    ...) const
{
    impl(ar, data);
}

template <typename Tp>
void
serialization<Tp, true>::operator()(cereal::PrettyJSONOutputArchive& ar,
                                    const basic_tree_map_type&       data, ...) const
{
    impl(ar, data);
}

template <typename Tp>
void
serialization<Tp, true>::operator()(cereal::PrettyJSONOutputArchive& ar,
                                    const result_type&               data, ...) const
{
    impl(ar, data);
}

template <typename Tp>
void
serialization<Tp, true>::operator()(cereal::PrettyJSONOutputArchive& ar,
                                    const distrib_type&              data, ...) const
{
    impl(ar, data);
}

template <typename Tp>
void
serialization<Tp, true>::operator()(cereal::JSONInputArchive& ar,
                                    basic_tree_vector_type&   data, ...) const
{
    impl(ar, data);
}

template <typename Tp>
void
serialization<Tp, true>::operator()(cereal::JSONInputArchive&            ar,
                                    std::vector<basic_tree_vector_type>& data, ...) const
{
    impl(ar, data);
}

template <typename Tp>
void
serialization<Tp, true>::operator()(cereal::JSONInputArchive& ar, result_type& data,
                                    ...) const
{
    impl(ar, data);
}

template <typename Tp>
void
serialization<Tp, true>::operator()(cereal::JSONInputArchive& ar, distrib_type& data,
                                    ...) const
{
    impl(ar, data);
}
//
}  // namespace internal
}  // namespace operation
}  // namespace tim

#endif

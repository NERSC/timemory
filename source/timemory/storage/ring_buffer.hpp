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

#include <iostream>
#include <sstream>

#include <stdexcept>

namespace tim
{
namespace base
{
/// \struct tim::base::ring_buffer
/// \brief Ring buffer implementation, using mmap as backend.
struct ring_buffer
{
    ring_buffer() = default;
    explicit ring_buffer(size_t _size) { init(_size); }

    ~ring_buffer();

    ring_buffer(const ring_buffer&) = delete;
    ring_buffer(ring_buffer&&)      = default;

    ring_buffer& operator=(const ring_buffer&) = delete;
    ring_buffer& operator=(ring_buffer&&) = default;

    /// Returns whether the buffer has been allocated
    bool is_initialized() const { return m_init; }

    /// Get the total number of bytes supported
    size_t capacity() const { return m_size; }

    /// Creates new ring buffer.
    void init(size_t size);

    /// Destroy ring buffer.
    void destroy();

    /// Write data to buffer.
    template <typename Tp>
    size_t write(Tp* in, std::enable_if_t<std::is_class<Tp>::value, int> = 0);

    template <typename Tp>
    size_t write(Tp* in, std::enable_if_t<!std::is_class<Tp>::value, int> = 0);

    /// Read data from buffer.
    template <typename Tp>
    size_t read(Tp* out, std::enable_if_t<std::is_class<Tp>::value, int> = 0) const;

    template <typename Tp>
    size_t read(Tp* out, std::enable_if_t<!std::is_class<Tp>::value, int> = 0) const;

    /// Returns number of bytes currently held by the buffer.
    size_t count() const { return m_write_count - m_read_count; }

    /// Returns how many bytes are availiable in the buffer.
    size_t free() const { return m_size - count(); }

    /// Returns if the buffer is empty.
    bool is_empty() const { return count() == 0; }

    /// Returns if the buffer is full.
    bool is_full() const { return count() == m_size; }

    size_t rewind(size_t n) const;

private:
    /// Returns the current write pointer.
    void* write_ptr() const
    {
        return static_cast<char*>(m_ptr) + (m_write_count % m_size);
    }

    /// Returns the current read pointer.
    void* read_ptr() const { return static_cast<char*>(m_ptr) + (m_read_count % m_size); }

private:
    bool           m_init        = false;
    int            m_fd          = 0;
    void*          m_ptr         = nullptr;
    size_t         m_size        = 0;
    mutable size_t m_read_count  = 0;
    size_t         m_write_count = 0;
};
//
template <typename Tp>
size_t
ring_buffer::write(Tp* in, std::enable_if_t<std::is_class<Tp>::value, int>)
{
    if(in == nullptr)
        return 0;

    auto _length = sizeof(Tp);

    // Make sure we don't put in more than there's room for, by writing no
    // more than there is free.
    if(_length > free())
        _length = free();

    // Copy in.
    new(write_ptr()) Tp{ *in };
    // memcpy(write_ptr(), in, _length);

    // Update write count
    m_write_count += _length;

    return _length;
}
//
template <typename Tp>
size_t
ring_buffer::write(Tp* in, std::enable_if_t<!std::is_class<Tp>::value, int>)
{
    if(in == nullptr)
        return 0;

    auto _length = sizeof(Tp);

    // Make sure we don't put in more than there's room for, by writing no
    // more than there is free.
    if(_length > free())
        _length = free();

    // Copy in.
    memcpy(write_ptr(), in, _length);

    // Update write count
    m_write_count += _length;

    return _length;
}
//
template <typename Tp>
size_t
ring_buffer::read(Tp* out, std::enable_if_t<std::is_class<Tp>::value, int>) const
{
    if(is_empty() || out == nullptr)
        return 0;

    auto _length = sizeof(Tp);

    // Make sure we do not read out more than there is actually in the buffer.
    if(_length > count())
        _length = count();

    // Copy out for BYTE, nothing magic here.
    *out = *(reinterpret_cast<Tp*>(read_ptr()));

    // Update read count.
    m_read_count += _length;

    return _length;
}
//
template <typename Tp>
size_t
ring_buffer::read(Tp* out, std::enable_if_t<!std::is_class<Tp>::value, int>) const
{
    if(is_empty() || out == nullptr)
        return 0;

    auto _length = sizeof(Tp);

    using Up = typename std::remove_const<Tp>::type;

    // Make sure we do not read out more than there is actually in the buffer.
    if(_length > count())
        _length = count();

    assert(out != nullptr);
    // Copy out for BYTE, nothing magic here.
    Up* _out = const_cast<Up*>(out);
    memcpy(_out, read_ptr(), _length);

    // Update read count.
    m_read_count += _length;

    return _length;
}
//
size_t
ring_buffer::rewind(size_t n) const
{
    if(n > m_read_count)
        n = m_read_count;
    m_read_count -= n;
    return n;
}
//
}  // namespace base
//
namespace data_storage
{
template <typename Tp>
struct ring_buffer : private base::ring_buffer
{
    using base_type = base::ring_buffer;

    ring_buffer()  = default;
    ~ring_buffer() = default;

    explicit ring_buffer(size_t _size)
    : base_type{ _size * sizeof(Tp) }
    {}

    ring_buffer(const ring_buffer&) = delete;
    ring_buffer(ring_buffer&&)      = default;

    ring_buffer& operator=(const ring_buffer&) = delete;
    ring_buffer& operator=(ring_buffer&&) = default;

    /// Returns whether the buffer has been allocated
    bool is_initialized() const { return base_type::is_initialized(); }

    /// Get the total number of bytes supported
    size_t capacity() const { return base_type::capacity(); }

    /// Creates new ring buffer.
    void init(size_t _size) { base_type::init(_size * sizeof(Tp)); }

    /// Destroy ring buffer.
    void destroy() { base_type::destroy(); }

    /// Write data to buffer.
    size_t data_size() { return sizeof(Tp); }

    /// Write data to buffer.
    size_t write(Tp* in) { return base_type::write<Tp>(in); }

    /// Read data from buffer.
    size_t read(Tp* out) const { return base_type::read<Tp>(out); }

    /// Returns number of bytes currently held by the buffer.
    size_t count() const { return base_type::count() / sizeof(Tp); }

    /// Returns how many bytes are availiable in the buffer.
    size_t free() const { return base_type::free() / sizeof(Tp); }

    /// Returns if the buffer is empty.
    bool is_empty() const { return base_type::is_empty(); }

    /// Returns if the buffer is full.
    bool is_full() const { return base_type::is_full(); }

    /// Rewinds the read pointer
    size_t rewind(size_t n) const { return base_type::rewind(n); }

    template <typename... Args>
    auto emplace(Args&&... args)
    {
        Tp _obj{ std::forward<Args>(args)... };
        return write(&_obj);
    }

    friend std::ostream& operator<<(std::ostream& os, const ring_buffer& obj)
    {
        std::stringstream ss;
        ss << std::boolalpha << "is_initialized: " << obj.is_initialized()
           << ", capacity: " << obj.capacity() << ", count: " << obj.count()
           << ", free: " << obj.free() << ", is_empty: " << obj.is_empty()
           << ", is_full: " << obj.is_full() << ", capacity: " << obj.capacity();
        os << ss.str();
        return os;
    }
};
}  // namespace data_storage
//
}  // namespace tim

#if !defined(TIMEMORY_COMMON_SOURCE) && !defined(TIMEMORY_USE_COMMON_EXTERN)
#    include "timemory/storage/ring_buffer.cpp"
#endif

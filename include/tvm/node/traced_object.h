/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#ifndef TVM_NODE_TRACED_OBJECT_H_
#define TVM_NODE_TRACED_OBJECT_H_

#include <tvm/node/object_path.h>
#include <tvm/node/reflection.h>
#include <tvm/runtime/object.h>

#include <string>
#include <utility>

namespace tvm {

template <typename RefT>
class TracedObject;
template <typename K, typename V>
class TracedMap;
template <typename T>
class TracedArray;
template <typename T>
class TracedOptional;
template <typename T>
class TracedBasicValue;

namespace detail {

template <typename T>
struct TypedAttrGetter : public AttrVisitor {
  const char* desired_key;
  const T* found_attr;

  explicit TypedAttrGetter(const char* desired_key)
      : desired_key(desired_key), found_attr(nullptr) {}

  void Visit(const char* key, double* value) final { DoVisit(key, value); }
  void Visit(const char* key, int64_t* value) final { DoVisit(key, value); }
  void Visit(const char* key, uint64_t* value) final { DoVisit(key, value); }
  void Visit(const char* key, int* value) final { DoVisit(key, value); }
  void Visit(const char* key, bool* value) final { DoVisit(key, value); }
  void Visit(const char* key, void** value) final { DoVisit(key, value); }
  void Visit(const char* key, DataType* value) final { DoVisit(key, value); }
  void Visit(const char* key, std::string* value) final { DoVisit(key, value); }
  void Visit(const char* key, runtime::NDArray* value) final { DoVisit(key, value); }
  void Visit(const char* key, ObjectRef* value) final { DoVisit(key, value); }

 private:
  void DoVisit(const char* key, const T* value) {
    if (!strcmp(desired_key, key)) {
      found_attr = value;
    }
  }

  template <typename U>
  void DoVisit(const char* key, const U*) {
    if (!strcmp(desired_key, key)) {
      LOG(FATAL) << "Attribute '" << key << "' is present but has wrong type";
    }
  }
};

template <typename T, bool IsObject = std::is_base_of<ObjectRef, T>::value,
          bool IsEnum = std::is_enum<T>::value>
struct GetTypedAttr;

template <typename T>
struct GetTypedAttr<T, true, false> {
  T operator()(const ObjectRef& object, const char* attr_key) const {
    TypedAttrGetter<ObjectRef> visitor(attr_key);
    ReflectionVTable::Global()->VisitAttrs(const_cast<Object*>(object.get()), &visitor);
    ICHECK(visitor.found_attr != nullptr) << "No such attribute '" << attr_key << "'";
    return Downcast<T>(*visitor.found_attr);
  }
};

template <typename T>
struct GetTypedAttr<T, false, false> {
  const T& operator()(const ObjectRef& object, const char* attr_key) const {
    TypedAttrGetter<T> visitor(attr_key);
    ReflectionVTable::Global()->VisitAttrs(const_cast<Object*>(object.get()), &visitor);
    ICHECK(visitor.found_attr != nullptr) << "No such attribute '" << attr_key << "'";
    return *visitor.found_attr;
  }
};

template <typename T>
struct GetTypedAttr<T, false, true> {
  const T& operator()(const ObjectRef& object, const char* attr_key) const {
    static_assert(std::is_same<int, typename std::underlying_type<T>::type>::value);

    TypedAttrGetter<int> visitor(attr_key);
    ReflectionVTable::Global()->VisitAttrs(const_cast<Object*>(object.get()), &visitor);
    ICHECK(visitor.found_attr != nullptr) << "No such attribute '" << attr_key << "'";
    return *reinterpret_cast<const T*>(visitor.found_attr);
  }
};

template <typename T, bool IsObject = std::is_base_of<ObjectRef, T>::value>
struct TracedObjectWrapperSelector;

template <typename T>
struct TracedObjectWrapperSelector<T, false> {
  using Type = TracedBasicValue<T>;
};

template <typename T>
struct TracedObjectWrapperSelector<T, true> {
  using Type = TracedObject<T>;
};

template <typename K, typename V>
struct TracedObjectWrapperSelector<Map<K, V>, true> {
  using Type = TracedMap<K, V>;
};

template <typename T>
struct TracedObjectWrapperSelector<Array<T>, true> {
  using Type = TracedArray<T>;
};

template <typename T>
struct TracedObjectWrapperSelector<Optional<T>, true> {
  using Type = TracedOptional<T>;
};

}  // namespace detail

template <typename RefT>
class TracedObject {
 public:
  explicit TracedObject(const RefT& object_ref, ObjectPath path)
      : ref_(object_ref), path_(std::move(path)) {}

  template <typename DerivedRef>
  TracedObject(const TracedObject<DerivedRef>& derived)
      : ref_(derived.Get()), path_(derived.GetPath()) {}

  template <typename T>
  typename detail::TracedObjectWrapperSelector<T>::Type GetAttr(const char* attr_key) const {
    using WrapperType = typename detail::TracedObjectWrapperSelector<T>::Type;
    return WrapperType(detail::GetTypedAttr<T>()(ref_, attr_key), path_->Attr(attr_key));
  }

  const RefT& Get() const { return ref_; }

  template <typename RefU>
  bool IsInstance() const {
    return ref_->template IsInstance<typename RefU::ContainerType>();
  }

  bool defined() const { return ref_.defined(); }

  template <typename U>
  TracedObject<U> Downcast() const {
    return TracedObject<U>(tvm::runtime::Downcast<U>(ref_), path_);
  }

  template <typename RefU>
  TracedOptional<RefU> TryDowncast() const {
    if (ref_->template IsInstance<typename RefU::ContainerType>()) {
      return Downcast<RefU>();
    } else {
      return TracedOptional<RefU>(NullOpt, path_);
    }
  }

  const ObjectPath& GetPath() const { return path_; }

 private:
  RefT ref_;
  ObjectPath path_;
};

template <typename K, typename V>
class TracedMapIterator {
 public:
  using WrappedV = typename detail::TracedObjectWrapperSelector<V>::Type;
  using MapIter = typename Map<K, V>::iterator;

  using iterator_category = std::bidirectional_iterator_tag;
  using difference_type = ptrdiff_t;
  using value_type = const std::pair<K, WrappedV>;
  using pointer = value_type*;
  using reference = value_type;

  explicit TracedMapIterator(MapIter iter, ObjectPath map_path)
      : iter_(iter), map_path_(std::move(map_path)) {}

  bool operator==(const TracedMapIterator& other) const { return iter_ == other.iter_; }

  bool operator!=(const TracedMapIterator& other) const { return iter_ != other.iter_; }

  pointer operator->() const = delete;

  reference operator*() const {
    auto kv = *iter_;
    return std::make_pair(kv.first, WrappedV(kv.second, map_path_->MapValue(kv.first)));
  }

  TracedMapIterator& operator++() {
    ++iter_;
    return *this;
  }

  TracedMapIterator operator++(int) {
    TracedMapIterator copy = *this;
    ++(*this);
    return copy;
  }

 private:
  MapIter iter_;
  ObjectPath map_path_;
};

template <typename K, typename V>
class TracedMap {
 public:
  using WrappedV = typename detail::TracedObjectWrapperSelector<V>::Type;

  using iterator = TracedMapIterator<K, V>;

  explicit TracedMap(Map<K, V> map, ObjectPath path)
      : map_(std::move(map)), path_(std::move(path)) {}

  WrappedV at(const K& key) const {
    auto it = map_.find(key);
    ICHECK(it != map_.end()) << "No such key in Map";
    auto kv = *it;
    return WrappedV(kv.second, path_->MapValue(kv.first));
  }

  const Map<K, V>& Get() const { return map_; }

  const ObjectPath& GetPath() const { return path_; }

  iterator begin() const { return iterator(map_.begin(), path_); }

  iterator end() const { return iterator(map_.end(), path_); }

  bool empty() const { return map_.empty(); }

 private:
  Map<K, V> map_;
  ObjectPath path_;
};

template <typename T>
class TracedArrayIterator {
 public:
  using WrappedT = typename detail::TracedObjectWrapperSelector<T>::Type;

  using difference_type = ptrdiff_t;
  using value_type = WrappedT;
  using pointer = WrappedT*;
  using reference = WrappedT&;
  using iterator_category = std::random_access_iterator_tag;

  explicit TracedArrayIterator(Array<T> array, size_t index, ObjectPath array_path)
      : array_(array), index_(index), array_path_(array_path) {}

  TracedArrayIterator& operator++() {
    ++index_;
    return *this;
  }
  TracedArrayIterator& operator--() {
    --index_;
    return *this;
  }
  TracedArrayIterator operator++(int) {
    TracedArrayIterator copy = *this;
    ++index_;
    return copy;
  }
  TracedArrayIterator operator--(int) {
    TracedArrayIterator copy = *this;
    --index_;
    return copy;
  }

  TracedArrayIterator operator+(difference_type offset) const {
    return TracedArrayIterator(array_, index_ + offset, array_path_);
  }

  TracedArrayIterator operator-(difference_type offset) const {
    return TracedArrayIterator(array_, index_ - offset, array_path_);
  }

  difference_type operator-(const TracedArrayIterator& rhs) const { return index_ - rhs.index_; }

  bool operator==(TracedArrayIterator other) const {
    return array_.get() == other.array_.get() && index_ == other.index_;
  }
  bool operator!=(TracedArrayIterator other) const { return !(*this == other); }
  value_type operator*() const { return WrappedT(array_[index_], array_path_->ArrayIndex(index_)); }

  bool empty() const { return array_.empty(); }

 private:
  Array<T> array_;
  size_t index_;
  ObjectPath array_path_;
};

template <typename T>
class TracedArray {
 public:
  using WrappedT = typename detail::TracedObjectWrapperSelector<T>::Type;

  using iterator = TracedArrayIterator<T>;

  explicit TracedArray(Array<T> array, ObjectPath path)
      : array_(std::move(array)), path_(std::move(path)) {}

  const Array<T>& Get() const { return array_; }

  const ObjectPath& GetPath() const { return path_; }

  WrappedT operator[](size_t index) const {
    return WrappedT(array_[index], path_->ArrayIndex(index));
  }

  iterator begin() const { return iterator(array_, 0, path_); }

  iterator end() const { return iterator(array_, array_.size(), path_); }

  bool empty() const { return array_.empty(); }

  size_t size() const { return array_.size(); }

 private:
  Array<T> array_;
  ObjectPath path_;
};

template <typename T>
class TracedOptional {
 public:
  using WrappedT = typename detail::TracedObjectWrapperSelector<T>::Type;

  TracedOptional(const WrappedT& value)  // NOLINT(runtime/explicit)
      : optional_(value.Get().defined() ? value.Get() : Optional<T>(NullOpt)),
        path_(value.GetPath()) {}

  explicit TracedOptional(Optional<T> optional, ObjectPath path)
      : optional_(std::move(optional)), path_(std::move(path)) {}

  const Optional<T>& Get() const { return optional_; }

  const ObjectPath& GetPath() const { return path_; }

  bool defined() const { return optional_.defined(); }

  WrappedT value() const { return WrappedT(optional_.value(), path_); }

  explicit operator bool() const { return optional_.defined(); }

 private:
  Optional<T> optional_;
  ObjectPath path_;
};

template <typename T>
class TracedBasicValue {
 public:
  explicit TracedBasicValue(const T& value, ObjectPath path)
      : value_(value), path_(std::move(path)) {}

  const T& Get() const { return value_; }

  const ObjectPath& GetPath() const { return path_; }

  template <typename F>
  typename detail::TracedObjectWrapperSelector<typename std::result_of<F(const T&)>::type>::Type
  ApplyFunc(F&& f) const {
    return MakeTraced(f(value_), path_);
  }

 private:
  T value_;
  ObjectPath path_;
};

template <typename RefT>
typename detail::TracedObjectWrapperSelector<RefT>::Type MakeTraced(const RefT& object) {
  using WrappedT = typename detail::TracedObjectWrapperSelector<RefT>::Type;
  return WrappedT(object, ObjectPath::Root());
}

template <typename RefT>
typename detail::TracedObjectWrapperSelector<RefT>::Type MakeTraced(const RefT& object,
                                                                    ObjectPath path) {
  using WrappedT = typename detail::TracedObjectWrapperSelector<RefT>::Type;
  return WrappedT(object, std::move(path));
}

}  // namespace tvm

#endif  // TVM_NODE_TRACED_OBJECT_H_

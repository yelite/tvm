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
#ifndef TVM_SCRIPT_PRINTER_FUNCTOR_H_
#define TVM_SCRIPT_PRINTER_FUNCTOR_H_

#include <tvm/node/node.h>

#include <type_traits>
#include <utility>
#include <vector>

#include "tvm/node/traced_object.h"
#include "tvm/runtime/logging.h"

namespace tvm {
namespace script {
namespace printer {

template <typename FType>
class ObjectFunctor;

template <typename R, typename... Args>
class ObjectFunctor<R(const ObjectRef& n, Args...)> {
 private:
  template <class TObjectRef, class TCallable>
  using IsDispatchFunc = typename std::enable_if_t<
      std::is_convertible<TCallable, std::function<R(TObjectRef, Args...)>>::value>;
  using TSelf = ObjectFunctor<R(const ObjectRef& n, Args...)>;
  std::unordered_map<std::string, std::vector<runtime::PackedFunc>> dispatch_table_;

 public:
  R operator()(const String& token, const ObjectRef& n, Args... args) const {
    uint32_t type_index = n->type_index();
    if (const runtime::PackedFunc* pf = GetDispatch(type_index, token)) {
      return (*pf)(n, std::forward<Args>(args)...);
    }
    if (const runtime::PackedFunc* pf = GetDispatch(type_index, "")) {
      return (*pf)(n, std::forward<Args>(args)...);
    }
    ICHECK(false) << "ObjectFunctor calls un-registered function on type: " << n->GetTypeKey()
                  << " (token: " << token << ")";
    throw;
  }

  template <typename TObjectRef, typename TCallable,
            typename = IsDispatchFunc<TObjectRef, TCallable>>
  TSelf& set_dispatch(TCallable f) {
    SetDispatch(runtime::TypedPackedFunc<R(TObjectRef, Args...)>(f),
                TObjectRef::ContainerType::RuntimeTypeIndex(),  //
                &this->dispatch_table_[""]);
    return *this;
  }

  template <typename TObjectRef, typename TCallable,
            typename = IsDispatchFunc<TObjectRef, TCallable>>
  TSelf& set_dispatch(String token, TCallable f) {
    SetDispatch(runtime::TypedPackedFunc<R(TObjectRef, Args...)>(f),
                TObjectRef::ContainerType::RuntimeTypeIndex(),  //
                &this->dispatch_table_[token]);
    return *this;
  }

 private:
  void SetDispatch(runtime::PackedFunc f, uint32_t type_index,
                   std::vector<runtime::PackedFunc>* func) {
    if (func->size() <= type_index) {
      func->resize(type_index + 1, nullptr);
    }
    runtime::PackedFunc& slot = (*func)[type_index];
    if (slot != nullptr) {
      ICHECK(false) << "Dispatch for type is already registered: "
                    << runtime::Object::TypeIndex2Key(type_index);
    }
    slot = f;
  }

  const runtime::PackedFunc* GetDispatch(uint32_t type_index, const String& token) const {
    auto it = dispatch_table_.find(token);
    if (it == dispatch_table_.end()) {
      return nullptr;
    }
    const std::vector<runtime::PackedFunc>& tab = it->second;
    if (type_index >= tab.size()) {
      return nullptr;
    }
    const PackedFunc* f = &tab[type_index];
    if (f->defined()) {
        return f;
    } else {
        return nullptr;
    }
  }
};

template <typename R, typename... Args>
class TracedObjectFunctor : private ObjectFunctor<R(const ObjectRef&, ObjectPath, Args...)> {
  template <class TObjectRef, class TCallable>
  using IsTracedObjectDispatchFunc = typename std::enable_if_t<
      std::is_convertible<TCallable, std::function<R(TracedObject<TObjectRef>, Args...)>>::value>;

  using TBase = ObjectFunctor<R(const ObjectRef&, ObjectPath, Args...)>;
  using TSelf = TracedObjectFunctor<R, Args...>;

 public:
  using TBase::operator();

  template <typename TObjectRef, typename TCallable,
            typename = IsTracedObjectDispatchFunc<TObjectRef, TCallable>>
  TSelf& set_dispatch(TCallable f) {
    TBase::template set_dispatch<TObjectRef>(
        [f](TObjectRef object, ObjectPath path, Args... args) -> R {
          return f(MakeTraced(object, path), std::forward<Args>(args)...);
        });
    return *this;
  }

  template <typename TObjectRef, typename TCallable,
            typename = IsTracedObjectDispatchFunc<TObjectRef, TCallable>>
  TSelf& set_dispatch(String token, TCallable f) {
    TBase::template set_dispatch<TObjectRef>(
        std::move(token), [f](TObjectRef object, ObjectPath path, Args... args) -> R {
          return f(MakeTraced(object, path), std::forward<Args>(args)...);
        });
    return *this;
  }
};

}  // namespace printer
}  // namespace script
}  // namespace tvm
#endif  // TVM_SCRIPT_PRINTER_FUNCTOR_H_

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
/*!
 * \brief Registry class for dynamic disptach on ObjectRef
 */
#ifndef TVM_SCRIPT_PRINTER_REGISTRY_H_
#define TVM_SCRIPT_PRINTER_REGISTRY_H_

#include <tvm/runtime/object.h>

#include <functional>

namespace tvm {
namespace script {
namespace printer {

namespace {
using runtime::Downcast;
using runtime::ObjectRef;
}  // namespace

/* !
 * A modified version of NodeFunctor which supports auto downcast
 */
template <typename R, typename... Args>
class ObjectGenericFunction {
 private:
  using InternalFunctionType = std::function<R(const ObjectRef&, Args...)>;

  template <typename ObjectRefType, typename ReturnType>
  using FunctionType = ReturnType (*)(ObjectRefType, Args...);

  std::vector<InternalFunctionType> registry_;

 public:
  bool can_dispatch(const ObjectRef& ref) const {
    uint32_t type_index = ref->type_index();
    return type_index < registry_.size() && registry_[type_index] != nullptr;
  }

  R operator()(const ObjectRef& ref, Args&&... args) const {
    ICHECK(can_dispatch(ref)) << "ObjectGenericFunction calls un-registered function on type "
                              << ref->GetTypeKey();
    return registry_[ref->type_index()](ref, std::forward<Args>(args)...);
  }

  template <typename ObjectRefType, typename ReturnType>
  ObjectGenericFunction& register_func(FunctionType<ObjectRefType, ReturnType> func) {
    using ObjectType = typename ObjectRefType::ContainerType;
    uint32_t tindex = ObjectType::RuntimeTypeIndex();
    if (registry_.size() <= tindex) {
      registry_.resize(tindex + 1, nullptr);
    }
    ICHECK(registry_[tindex] == nullptr)
        << "Handler for " << ObjectType::_type_key << " is already set";

    registry_[tindex] = [handler = std::move(func)](const ObjectRef& ref, Args&&... args) {
      R result = handler(Downcast<ObjectRefType>(ref), std::forward<Args>(args)...);
      return result;
    };
    return *this;
  }
};

#define TVM_STATIC_REGISTER_GENERIC_FUNCTION(GenericFunctionGetter, HandlerFunction) \
  TVM_STR_CONCAT(static TVM_ATTRIBUTE_UNUSED int __object_generic_function_, __COUNTER__) =                        \
      (GenericFunctionGetter().register_func(+HandlerFunction), 0)

}  // namespace printer
}  // namespace script
}  // namespace tvm

#endif

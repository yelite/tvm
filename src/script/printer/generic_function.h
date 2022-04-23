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
#ifndef TVM_SCRIPT_PRINTER_GENERIC_FUNCTION_H_
#define TVM_SCRIPT_PRINTER_GENERIC_FUNCTION_H_

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
  bool CanDispatch(const ObjectRef& ref) const {
    uint32_t type_index = ref->type_index();
    return type_index < registry_.size() && registry_[type_index] != nullptr;
  }

  std::vector<uint32_t> RegistredTypeIds() const {
    std::vector<uint32_t> result;
    for (std::size_t type_index = 0; type_index < registry_.size(); type_index++) {
      if (registry_[type_index] != nullptr) {
        result.emplace_back(type_index);
      }
    }
    return result;
  }

  std::vector<std::string> RegisteredTypeKeys() const {
    std::vector<std::string> result;
    for (uint32_t type_index : RegisteredTypeKeys()) {
      result.emplace_back(runtime::Object::TypeIndex2Key(type_index));
    }
    return result;
  }

  template <typename ObjectRefType, typename ReturnType>
  ObjectGenericFunction& RegisterFunction(FunctionType<ObjectRefType, ReturnType> func) {
    using ObjectType = typename ObjectRefType::ContainerType;
    uint32_t tindex = ObjectType::RuntimeTypeIndex();
    if (registry_.size() <= tindex) {
      registry_.resize(tindex + 1, nullptr);
    }
    ICHECK(registry_[tindex] == nullptr)
        << "Handler for " << ObjectType::_type_key << " is already set";

    registry_[tindex] = [handler = std::move(func)](const ObjectRef& ref, Args&&... args) {
      return handler(Downcast<ObjectRefType>(ref), std::forward<Args>(args)...);
    };
    return *this;
  }

  R operator()(const ObjectRef& ref, Args... args) const {
    ICHECK(CanDispatch(ref)) << "ObjectGenericFunction calls un-registered function on type "
                             << ref->GetTypeKey();
    return registry_[ref->type_index()](ref, std::move(args)...);
  }
};

#define TVM_STATIC_REGISTER_GENERIC_FUNCTION(GenericFunctionGetter, HandlerFunction) \
  TVM_STR_CONCAT(static TVM_ATTRIBUTE_UNUSED int __object_generic_function_, __COUNTER__) =                        \
      (GenericFunctionGetter().RegisterFunction(+HandlerFunction), 0)

}  // namespace printer
}  // namespace script
}  // namespace tvm

#endif

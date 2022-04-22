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
 * \brief Context and Frame represent semantic information in IR during printing
 */
#ifndef TVM_SCRIPT_PRINTER_FRAME_H_
#define TVM_SCRIPT_PRINTER_FRAME_H_

#include <tvm/runtime/container/array.h>
#include <tvm/runtime/container/map.h>
#include <tvm/runtime/container/string.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/memory.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/packed_func.h>

#include "generic_function.h"

namespace tvm {
namespace script {
namespace printer {

namespace {
using runtime::make_object;
using runtime::Object;
using runtime::ObjectRef;
}  // namespace

ObjectGenericFunction<String>& VariableNamers();
String GetVariableName(const ObjectRef& ref);

#define TVMSCRIPT_PRINTER_VARIABLE_NAMER(VariableNamer) \
  TVM_STATIC_REGISTER_GENERIC_FUNCTION(::tvm::script::printer::VariableNamers, VariableNamer)

class TranslatorFrameNode : public Object {
 public:
  Map<String, ObjectRef> variables;

  TranslatorFrameNode() = default;
  virtual ~TranslatorFrameNode() = default;

  virtual bool AddVariable(String name, ObjectRef variable);
  virtual Optional<ObjectRef> GetVariable(String name) const;

  static constexpr const char* _type_key = "script.printer.TranslatorFrame";
  TVM_DECLARE_BASE_OBJECT_INFO(TranslatorFrameNode, Object);
};

class TranslatorFrame : public ObjectRef {
 public:
  TranslatorFrame() : TranslatorFrame(make_object<TranslatorFrameNode>()) {}

  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(TranslatorFrame, ObjectRef,
                                                    TranslatorFrameNode);
};

class TranslatorContextNode : public Object {
 public:
  Array<TranslatorFrame> frames;

  TranslatorContextNode() = default;
  virtual ~TranslatorContextNode() = default;

  static constexpr const char* _type_key = "script.printer.TranslatorContext";
  TVM_DECLARE_BASE_OBJECT_INFO(TranslatorContextNode, Object);
};

template <typename FrameType>
class FrameGuard {
 public:
  FrameType frame;
  TranslatorContextNode* context;

  FrameGuard() : frame(TranslatorFrame()), context(nullptr) {}
  FrameGuard(FrameType frame, TranslatorContextNode* context) : context(context) {
    context->frames.push_back(frame);
    this->frame = std::move(frame);
  }

  ~FrameGuard() {
    if (context == nullptr) {
      return;
    }
    ICHECK_EQ(frame, context->frames.back()) << "Translator frame mismatch when popping";
    context->frames.pop_back();
  };

  FrameGuard(FrameGuard<FrameType>&& other) noexcept
      : frame(std::move(other.frame)), context(other.context) {
    other.context = nullptr;
  }

  FrameGuard(const FrameGuard<FrameType>& other) = delete;
  FrameGuard<FrameType>& operator=(const FrameGuard<FrameType>& other) = delete;
};

class TranslatorContext : public ObjectRef {
 public:
  TranslatorContext() {
    auto node = make_object<TranslatorContextNode>();
    // Add global frame to capture free variables
    node->frames.push_back(TranslatorFrame());
    data_ = std::move(node);
  }

  template <typename FrameType = TranslatorFrame>
  FrameGuard<FrameType> WithFrame();

  template <typename FrameType>
  bool HasFrame() const;

  template <typename FrameType>
  Optional<FrameType> GetFrame() const;

  template <typename FrameType>
  FrameType GetFrameOrDefault() const;

  TranslatorFrame GlobalFrame() const;

  void AddVariable(ObjectRef variable);
  Optional<ObjectRef> GetVariable(String name) const;
  bool HasVariable(String name) const { return GetVariable(std::move(name)) != nullptr; }

  void OnVariableUsed(ObjectRef variable);

  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(TranslatorContext, ObjectRef,
                                                    TranslatorContextNode);
};

template <typename FrameType>
FrameGuard<FrameType> TranslatorContext::WithFrame() {
  return FrameGuard<FrameType>(FrameType(), get());
}

template <typename FrameType>
Optional<FrameType> TranslatorContext::GetFrame() const {
  for (auto it = get()->frames.rbegin(); it != get()->frames.rend(); ++it) {
    if ((*it)->IsInstance<FrameType>()) {
      return *it;
    }
  }
  return NullOpt;
}

template <typename FrameType>
bool TranslatorContext::HasFrame() const {
  return GetFrame<FrameType>() != nullptr;
}

template <typename FrameType>
FrameType TranslatorContext::GetFrameOrDefault() const {
  Optional<FrameType> frame = GetFrame<FrameType>();
  if (frame == nullptr) {
    return FrameType();
  } else {
    return frame;
  }
};

}  // namespace printer
}  // namespace script
}  // namespace tvm

#endif

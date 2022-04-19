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
 * \brief Frame represents semantic information in IR during printing
 */
#ifndef TVM_SCRIPT_PRINTER_FRAME_H_
#define TVM_SCRIPT_PRINTER_FRAME_H_

#include <tvm/runtime/container/array.h>
#include <tvm/runtime/container/map.h>
#include <tvm/runtime/container/string.h>
#include <tvm/runtime/memory.h>
#include <tvm/runtime/object.h>

#include "registry.h"
#include "tvm/runtime/logging.h"

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

class PrinterFrameNode : public Object {
 public:
  Map<String, ObjectRef> variables;

  PrinterFrameNode() = default;
  virtual ~PrinterFrameNode() = default;

  static constexpr const char* _type_key = "script.printer.Frame";
  TVM_DECLARE_BASE_OBJECT_INFO(PrinterFrameNode, Object);
};

class PrinterFrame : public ObjectRef {
 public:
  PrinterFrame() : PrinterFrame(make_object<PrinterFrameNode>()) {}
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(PrinterFrame, ObjectRef, PrinterFrameNode);
};

class PrinterContextNode : public Object {
 public:
  Array<PrinterFrame> frames;

  PrinterContextNode() = default;
  virtual ~PrinterContextNode() = default;

  static constexpr const char* _type_key = "script.printer.Context";
  TVM_DECLARE_FINAL_OBJECT_INFO(PrinterContextNode, Object);
};

class PrinterContext : public ObjectRef {
 public:
  PrinterContext() {
    auto node = make_object<PrinterContextNode>();
    node->frames.push_back(PrinterFrame());
    data_ = std::move(node);
  }

  template <typename FrameType = PrinterFrame>  // TODO: Remove after having more subclass of frame
  FrameType AddFrame() {
    FrameType frame;
    get()->frames.push_back(frame);
    return frame;
  }
  void PopFrame() {
    Array<PrinterFrame>& frames = get()->frames;
    ICHECK_GT(frames.size(), 1);
    frames.pop_back();
  }

  void AddVariable(ObjectRef variable);
  Optional<ObjectRef> GetVariable(String name);
  bool HasVariable(String name) { return GetVariable(std::move(name)) != nullptr; }

  void OnVariableUsed(ObjectRef variable);
  Map<String, ObjectRef> GetFreeVariables();

  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(PrinterContext, ObjectRef, PrinterContextNode);

 private:
  PrinterFrame GlobalFrame();
};

}  // namespace printer
}  // namespace script
}  // namespace tvm

#endif

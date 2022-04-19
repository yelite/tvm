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

#include "context.h"

#include <tvm/tir/buffer.h>
#include <tvm/tir/var.h>

#include "registry.h"
#include "tvm/runtime/container/base.h"
#include "tvm/runtime/logging.h"

namespace tvm {
namespace script {
namespace printer {

ObjectGenericFunction<String>& VariableNamers() {
  static ObjectGenericFunction<String> f;
  return f;
}

String GetVariableName(const ObjectRef& ref) { return VariableNamers()(ref); }

void PrinterContext::AddVariable(ObjectRef variable) {
  String name = GetVariableName(variable);

  PrinterFrame top_frame = get()->frames.back();

  ICHECK(!top_frame->variables.Get(name));
  top_frame->variables.Set(name, variable);
}

Optional<ObjectRef> PrinterContext::GetVariable(String name) {
  auto self = get();
  for (auto it = self->frames.rbegin(); it != self->frames.rend(); ++it) {
    const PrinterFrame& frame = *it;
    if (frame->variables.find(name) != frame->variables.end()) {
      return frame->variables.Get(name);
    }
  }
  return NullOpt;
}

void PrinterContext::OnVariableUsed(ObjectRef variable) {
  String variable_name = GetVariableName(variable);
  Optional<ObjectRef> variable_from_frame = GetVariable(variable_name);
  if (variable_from_frame) {
    ICHECK_EQ(variable_from_frame.value(), variable);
  } else {
    GlobalFrame()->variables.Set(variable_name, variable);
  }
}

Map<String, ObjectRef> PrinterContext::GetFreeVariables() {
    return GlobalFrame()->variables;
}

PrinterFrame PrinterContext::GlobalFrame() { return get()->frames.front(); }

}  // namespace printer
}  // namespace script
}  // namespace tvm

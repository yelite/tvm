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

#include "context.h"

#include <tvm/runtime/container/base.h>
#include <tvm/runtime/logging.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/var.h>

namespace tvm {
namespace script {
namespace printer {

ObjectGenericFunction<String>& VariableNamers() {
  static ObjectGenericFunction<String> f;
  return f;
}

String GetVariableName(const ObjectRef& ref) {
  static ObjectGenericFunction<String>& f = VariableNamers();
  return f(ref);
}

bool TranslatorFrameNode::AddVariable(String name, ObjectRef variable) {
  ICHECK(variables.Get(name) == nullptr) << "Duplicate definition of variable " << name;
  variables.Set(GetVariableName(variable), variable);
  return true;
}

Optional<ObjectRef> TranslatorFrameNode::GetVariable(String name) const {
  return variables.Get(name);
}

TranslatorFrame TranslatorContext::GlobalFrame() const { return get()->frames.back(); }

void TranslatorContext::AddVariable(ObjectRef variable) {
  Array<TranslatorFrame>& frames = get()->frames;
  String variable_name = GetVariableName(variable);

  for (auto it = frames.rbegin(); it != frames.rend(); ++it) {
    if ((*it)->AddVariable(variable_name, variable)) {
      return;
    }
  }
}

Optional<ObjectRef> TranslatorContext::GetVariable(String name) const {
  TranslatorContextNode* self = get();
  for (auto it = self->frames.rbegin(); it != self->frames.rend(); ++it) {
    const TranslatorFrame& frame = *it;
    if (Optional<ObjectRef> v = frame->GetVariable(name)) {
      return v;
    }
  }
  return NullOpt;
}

void TranslatorContext::OnVariableUsed(ObjectRef variable) {
  String variable_name = GetVariableName(variable);
  if (Optional<ObjectRef> variable_from_frame = GetVariable(variable_name)) {
    ICHECK_EQ(variable_from_frame.value(), variable);
  } else {
    GlobalFrame()->AddVariable(variable_name, variable);
  }
}

}  // namespace printer
}  // namespace script
}  // namespace tvm

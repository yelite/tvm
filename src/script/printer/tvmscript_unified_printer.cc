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

#include "tvmscript_unified_printer.h"

#include <tvm/tir/op.h>
#include "context.h"
#include "doc.h"
#include "registry.h"

namespace tvm {
namespace script {
namespace printer {


ObjectGenericFunction<Doc, TVMScriptUnifiedPrinter*> DocTranslators() {
    static ObjectGenericFunction<Doc, TVMScriptUnifiedPrinter*> f;
    return f;
}

ObjectGenericFunction<TypeDoc, TVMScriptUnifiedPrinter*> VariableTypeDocTranslators() {
    static ObjectGenericFunction<TypeDoc, TVMScriptUnifiedPrinter*> f;
    return f;
}

String TVMScriptUnifiedPrinter::Print(const ObjectRef& ref) {
  auto element = ToDoc<Doc>(ref);
  Array<StmtDoc> prelude = GetPrelude();
  if (prelude.empty()) {
    return doc_printer_->Print({element});
  } else {
    return doc_printer_->Print({SeqStmtDoc(prelude), element});
  }
}

Array<StmtDoc> TVMScriptUnifiedPrinter::GetPrelude() {
  Array<StmtDoc> result;
  Map<String, ObjectRef> free_variables = context.GetFreeVariables();

  for (auto it = free_variables.begin(); it != free_variables.end(); ++it) {
      ObjectRef variable = (*it).second;
      AssignDoc declaration;
      declaration->target = IdentifierDoc(GetVariableName(variable));
      declaration->type = ToVariableTypeDoc(variable);
      result.push_back(declaration);
  }

  return result;
}

TypeDoc TVMScriptUnifiedPrinter::ToVariableTypeDoc(const ObjectRef& ref) {
  TypeDoc type_doc = VariableTypeDocTranslators()(ref, this);
  type_doc->origin_ir_node = ref;
  return type_doc;
}

String AsTVMScriptUnified(const ObjectRef& node, const String& tir_prefix) {
  auto printer = TVMScriptUnifiedPrinter(std::make_unique<PythonDocPrinter>(tir_prefix));
  return printer.Print(node);
}

TVM_REGISTER_GLOBAL("experiment.AsTVMScript").set_body_typed(AsTVMScriptUnified);

}  // namespace printer
}  // namespace script
}  // namespace tvm

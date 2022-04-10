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

namespace tvm {
namespace script {
namespace printer {

DocProducerRegistry& TVMScriptUnifiedPrinter::registry() {
  static DocProducerRegistry inst;
  return inst;
}

String TVMScriptUnifiedPrinter::Print(const ObjectRef& ref) {
  auto element = ToDoc<Doc>(ref);
  if (prelude_.empty()) {
    return doc_printer_->Print({element});
  } else {
    return doc_printer_->Print({SeqStmtDoc(prelude_), element});
  }
}

TypeDoc TVMScriptUnifiedPrinter::GetBufferTypeDoc(const tir::Buffer& buf) {
  TypeCallDoc type_doc;
  type_doc->base = ExprTypeDoc::TIRPrimitive("Buffer");

  if (buf->shape.size() > 1) {
    TupleDoc shape_doc;
    shape_doc->elements = ToExprDocArray(buf->shape);
    type_doc->args.push_back(ExprTypeDoc(shape_doc));
  } else {
    type_doc->args.push_back(ExprTypeDoc(ToExprDoc(buf->shape[0])));
  }
  type_doc->args.push_back(ExprTypeDoc(LiteralValueDoc(runtime::DLDataType2String(buf->dtype))));
  return type_doc;
}

TypeDoc TVMScriptUnifiedPrinter::GetVarTypeDoc(const tir::Var& var) {
  return ToDoc<TypeDoc>(GetType(var));
}

bool TVMScriptUnifiedPrinter::HasFreeVar(const String& name, const ObjectRef& var) {
  Optional<ObjectRef> free_var = free_vars_.Get(name);
  return free_var && free_var.value() == var;
}

String AsTVMScriptUnified(const ObjectRef& node, const String& tir_prefix) {
  auto printer = TVMScriptUnifiedPrinter(std::make_unique<PythonDocPrinter>(tir_prefix));
  return printer.Print(node);
}

TVM_REGISTER_GLOBAL("experiment.AsTVMScript").set_body_typed(AsTVMScriptUnified);

}  // namespace printer
}  // namespace script
}  // namespace tvm

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

#include <tvm/tir/op.h>

#include "doc.h"
#include "doc_printer.h"
#include "ir_docsifier.h"

namespace tvm {
namespace script {
namespace printer {

String AsTVMScript(const ObjectRef& node, const String& tir_prefix) {
  constexpr int32_t indent_spaces = 4;
  Map<String, String> ir_prefix;
  ir_prefix.Set("tir", tir_prefix);
  IRDocsifier ir_docsifier(ir_prefix);

  VarDefFrame def_frame = VarDefFrame(ir_docsifier->sym);
  auto frame_ctx = ir_docsifier->WithFrame(def_frame);

  Doc doc = ir_docsifier->AsDoc<Doc>(node);

  Array<Doc> doc_to_print;
  for (const StmtDoc& def : def_frame->stmts) {
    doc_to_print.push_back(def);
  }
  doc_to_print.push_back(doc);

  PythonDocPrinter doc_printer(indent_spaces);
  return doc_printer.Print(doc);
}

TVM_REGISTER_GLOBAL("experiment.AsTVMScript").set_body_typed(AsTVMScript);

}  // namespace printer
}  // namespace script
}  // namespace tvm

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
 * \brief Printer class to print TVMScript from Relax and TIR nodes.
 */
#ifndef TVM_SCRIPT_PRINTER_TVMSCRIPT_UNIFIED_PRINTER_H_
#define TVM_SCRIPT_PRINTER_TVMSCRIPT_UNIFIED_PRINTER_H_

#include "context.h"
#include "doc_printer.h"
#include "node_translator.h"

namespace tvm {
namespace script {
namespace printer {

class TVMScriptUnifiedPrinter {
 public:
  explicit TVMScriptUnifiedPrinter(std::unique_ptr<DocPrinter> element_printer)
      : doc_printer_(std::move(element_printer)){};

  String Print(const ObjectRef& ref);
  Doc PrintExtraVarDeclaration();

 protected:
  std::unique_ptr<DocPrinter> doc_printer_;
  NodeTranslator node_translator_;

  Array<StmtDoc> GetPrelude();
};

}  // namespace printer
}  // namespace script
}  // namespace tvm

#endif  // TVM_SCRIPT_PRINTER_TVMSCRIPT_UNIFIED_PRINTER_H_

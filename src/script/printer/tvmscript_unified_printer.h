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

namespace tvm {
namespace script {
namespace printer {

class TVMScriptUnifiedPrinter;

ObjectGenericFunction<Doc, TVMScriptUnifiedPrinter*> DocTranslators();
ObjectGenericFunction<TypeDoc, TVMScriptUnifiedPrinter*> VariableTypeDocTranslators();

#define TVMSCRIPT_PRINTER_DOC_TRANSLATOR(DocTranslator) \
  TVM_STATIC_REGISTER_GENERIC_FUNCTION(::tvm::script::printer::DocTranslators, DocTranslator)

#define TVMSCRIPT_PRINTER_VAR_TYPE_DOC_TRANSLATOR(VarTypeDocTranslator)                    \
  TVM_STATIC_REGISTER_GENERIC_FUNCTION(::tvm::script::printer::VariableTypeDocTranslators, \
                                       VarTypeDocTranslator)

class TVMScriptUnifiedPrinter {
 public:
  explicit TVMScriptUnifiedPrinter(std::unique_ptr<DocPrinter> element_printer)
      : doc_printer_(std::move(element_printer)){};

  String Print(const ObjectRef& ref);
  Doc PrintExtraVarDeclaration();

  template <typename T, typename = std::enable_if_t<std::is_base_of<Doc, T>::value>>
  T ToDoc(const ObjectRef& ref);

  template <typename DocType, typename NodeType>
  Array<DocType> ToDocArray(const Array<NodeType>& refs);

  ExprDoc ToExprDoc(const ObjectRef& ref) { return ToDoc<ExprDoc>(ref); }

  template <typename NodeType>
  Array<ExprDoc> ToExprDocArray(const Array<NodeType>& refs) {
    return ToDocArray<ExprDoc>(refs);
  }

  TypeDoc ToVariableTypeDoc(const ObjectRef& ref);

  PrinterContext context;

 protected:
  std::unique_ptr<DocPrinter> doc_printer_;

  Array<StmtDoc> GetPrelude();
};

template <typename T, typename>
T TVMScriptUnifiedPrinter::ToDoc(const ObjectRef& ref) {
  Doc doc = DocTranslators()(ref, this);
  doc->origin_ir_node = ref;
  return Downcast<T>(doc);
}

template <typename DocType, typename NodeType>
Array<DocType> TVMScriptUnifiedPrinter::ToDocArray(const Array<NodeType>& refs) {
  Array<DocType> result;
  for (auto& n : refs) {
    result.push_back(ToDoc<DocType>(n));
  }
  return result;
}

#define TVMSCRIPT_PRINTER_DOC_PRODUCER(Producer)                               \
  TVM_STR_CONCAT(TVM_REG_FUNC_VAR_DEF(TVMScriptUnifiedPrinter), __COUNTER__) = \
      TVMScriptUnifiedPrinter::registry().register_producer(+Producer)

}  // namespace printer
}  // namespace script
}  // namespace tvm

#endif  // TVM_SCRIPT_PRINTER_TVMSCRIPT_UNIFIED_PRINTER_H_

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
#ifndef TVM_SCRIPT_PRINTER_DOC_PRINTER_H_
#define TVM_SCRIPT_PRINTER_DOC_PRINTER_H_

#include "doc.h"

namespace tvm {
namespace script {
namespace printer {

class DocPrinter {
 public:
  virtual ~DocPrinter() = default;

  virtual String Print(std::initializer_list<Doc> docs);

 protected:
  virtual void PrintDoc(const Doc& doc);

  virtual void PrintDoc(const LiteralValueDoc& doc) = 0;
  virtual void PrintDoc(const ConstDoc& doc) = 0;
  virtual void PrintDoc(const IdentifierDoc& doc) = 0;
  virtual void PrintDoc(const AttrAccessDoc& doc) = 0;
  virtual void PrintDoc(const IndexDoc& doc) = 0;
  virtual void PrintDoc(const OperationDoc& doc) = 0;
  virtual void PrintDoc(const CallDoc& doc) = 0;
  virtual void PrintDoc(const TupleDoc& doc) = 0;
  virtual void PrintDoc(const StmtBlockDoc& doc) = 0;
  virtual void PrintDoc(const ScopeDoc& doc) = 0;
  virtual void PrintDoc(const ForDoc& doc) = 0;
  virtual void PrintDoc(const AssignDoc& doc) = 0;
  virtual void PrintDoc(const ExprTypeDoc& doc) = 0;
  virtual void PrintDoc(const TypeCallDoc& doc) = 0;
  virtual void PrintDoc(const FunctionDoc& doc) = 0;
  virtual void PrintDoc(const FunctionArgDoc& doc) = 0;

  using OutputStream = std::ostringstream;

  OutputStream& NewLine() {
    output_ << "\n" << std::string(indent_, ' ');
    return output_;
  }

  OutputStream output_;
  int indent_ = 0;
};

class PythonDocPrinter : public DocPrinter {
 public:
  PythonDocPrinter(String tir_prefix) : tir_prefix_(tir_prefix) {}

 protected:
  using DocPrinter::PrintDoc;

  void PrintDoc(const LiteralValueDoc& doc) final;
  void PrintDoc(const ConstDoc& doc) final;
  void PrintDoc(const IdentifierDoc& doc) final;
  void PrintDoc(const AttrAccessDoc& doc) final;
  void PrintDoc(const IndexDoc& doc) final;
  void PrintDoc(const OperationDoc& doc) final;
  void PrintDoc(const CallDoc& doc) final;
  void PrintDoc(const TupleDoc& doc) final;

  void PrintDoc(const StmtBlockDoc& doc) final;
  void PrintDoc(const ScopeDoc& doc) final;
  void PrintDoc(const ForDoc& doc) final;
  void PrintDoc(const AssignDoc& doc) final;

  void PrintDoc(const ExprTypeDoc& doc) final;
  void PrintDoc(const TypeCallDoc& doc) final;

  void PrintDoc(const FunctionDoc& doc) final;
  void PrintDoc(const FunctionArgDoc& doc) final;

 private:
  String tir_prefix_;

  int indent_spaces_ = 4;

  void IncreaseIndent() { indent_ += indent_spaces_; }

  void DecreaseIndent() { indent_ -= indent_spaces_; }

  void PrintWithIncreasedIndent(const Doc& doc) {
    IncreaseIndent();
    PrintDoc(doc);
    DecreaseIndent();
  }

  template <typename ContainerType>
  void PrintJoinedElements(const std::string& left, const ContainerType& elements,
                           const std::string& separator, const std::string& right,
                           const std::string& default_content = "") {
    output_ << left;
    bool is_first = true;
    for (auto& element : elements) {
      if (is_first) {
        is_first = false;
      } else {
        output_ << separator;
      }
      PrintObject(element);
    }
    if (is_first) {
      // This means that `elements` is empty
      output_ << default_content;
    }
    output_ << right;
  }

  template <typename... Arg>
  void PrintTIRPrimitiveCall(const std::string& primitive, Arg... args) {
    output_ << tir_prefix_ << "." << primitive;
    PrintJoinedElements("(", std::vector<typename std::common_type_t<Arg...>>{args...}, ", ", ")");
  }

  template <typename ObjType>
  std::enable_if_t<std::is_base_of<Doc, ObjType>::value, void> PrintObject(const ObjType& obj) {
    PrintDoc(obj);
  }

  template <typename ObjType>
  std::enable_if_t<!std::is_base_of<Doc, ObjType>::value, void> PrintObject(const ObjType& obj) {
    output_ << obj;
  }

  void PrintStringLiteral(const String& string);
  void PrintNumberNode(const PrimExpr& expr);
};

}  // namespace printer
}  // namespace script
}  // namespace tvm

#endif

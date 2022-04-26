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

  virtual String Print(Doc doc);
  virtual String Print(Array<Doc> docs);

 protected:
  virtual void PrintDoc(const Doc& doc);

  virtual void PrintTypedDoc(const LiteralDoc& doc) = 0;
  virtual void PrintTypedDoc(const SliceDoc& doc) = 0;
  virtual void PrintTypedDoc(const IdDoc& doc) = 0;
  virtual void PrintTypedDoc(const AttrAccessDoc& doc) = 0;
  virtual void PrintTypedDoc(const IndexDoc& doc) = 0;
  virtual void PrintTypedDoc(const OperationDoc& doc) = 0;
  virtual void PrintTypedDoc(const CallDoc& doc) = 0;
  virtual void PrintTypedDoc(const ListDoc& doc) = 0;
  virtual void PrintTypedDoc(const DictDoc& doc) = 0;
  virtual void PrintTypedDoc(const TupleDoc& doc) = 0;
  virtual void PrintTypedDoc(const StmtBlockDoc& doc) = 0;
  virtual void PrintTypedDoc(const ExprStmtDoc& doc) = 0;
  virtual void PrintTypedDoc(const ScopeDoc& doc) = 0;
  virtual void PrintTypedDoc(const ForDoc& doc) = 0;
  virtual void PrintTypedDoc(const AssignDoc& doc) = 0;
  virtual void PrintTypedDoc(const FunctionDoc& doc) = 0;

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
  PythonDocPrinter(int indent_spaces) : indent_spaces_(indent_spaces) {}

 protected:
  using DocPrinter::PrintDoc;

  void PrintTypedDoc(const LiteralDoc& doc) final;
  void PrintTypedDoc(const SliceDoc& doc) final;
  void PrintTypedDoc(const IdDoc& doc) final;
  void PrintTypedDoc(const AttrAccessDoc& doc) final;
  void PrintTypedDoc(const IndexDoc& doc) final;
  void PrintTypedDoc(const OperationDoc& doc) final;
  void PrintTypedDoc(const CallDoc& doc) final;
  void PrintTypedDoc(const ListDoc& doc) final;
  void PrintTypedDoc(const DictDoc& doc) final;
  void PrintTypedDoc(const TupleDoc& doc) final;
  void PrintTypedDoc(const StmtBlockDoc& doc) final;
  void PrintTypedDoc(const ExprStmtDoc& doc) final;
  void PrintTypedDoc(const ScopeDoc& doc) final;
  void PrintTypedDoc(const ForDoc& doc) final;
  void PrintTypedDoc(const AssignDoc& doc) final;
  void PrintTypedDoc(const FunctionDoc& doc) final;

 private:
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

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

using ByteSpan = std::pair<size_t, size_t>;

struct DocPrinterOptions {
  int indent_spaces = 4;
  bool print_line_numbers = false;
  size_t num_context_lines = std::numeric_limits<size_t>::max();
};

class DocPrinter {
 public:
  explicit DocPrinter(const DocPrinterOptions& options);
  virtual ~DocPrinter() = default;

  void Clear();
  void Append(const Doc& doc);
  void Append(const Doc& doc, const ObjectPath& path_to_highlight);
  String GetString() const;

 protected:
  void PrintDoc(const Doc& doc);

  void MarkCurrentPosition(const ObjectPath& path);

  virtual void PrintTypedDoc(const LiteralDoc& doc) = 0;
  virtual void PrintTypedDoc(const SliceDoc& doc) = 0;
  virtual void PrintTypedDoc(const IdDoc& doc) = 0;
  virtual void PrintTypedDoc(const AttrAccessDoc& doc) = 0;
  virtual void PrintTypedDoc(const IndexDoc& doc) = 0;
  virtual void PrintTypedDoc(const OperationDoc& doc) = 0;
  virtual void PrintTypedDoc(const CallDoc& doc) = 0;
  virtual void PrintTypedDoc(const LambdaDoc& doc) = 0;
  virtual void PrintTypedDoc(const ListDoc& doc) = 0;
  virtual void PrintTypedDoc(const DictDoc& doc) = 0;
  virtual void PrintTypedDoc(const TupleDoc& doc) = 0;
  virtual void PrintTypedDoc(const StmtBlockDoc& doc) = 0;
  virtual void PrintTypedDoc(const ExprStmtDoc& doc) = 0;
  virtual void PrintTypedDoc(const ScopeDoc& doc) = 0;
  virtual void PrintTypedDoc(const IfDoc& doc) = 0;
  virtual void PrintTypedDoc(const WhileDoc& doc) = 0;
  virtual void PrintTypedDoc(const ForDoc& doc) = 0;
  virtual void PrintTypedDoc(const AssignDoc& doc) = 0;
  virtual void PrintTypedDoc(const FunctionDoc& doc) = 0;

  using OutputStream = std::ostringstream;

  void IncreaseIndent() { indent_ += options_.indent_spaces; }

  void DecreaseIndent() { indent_ -= options_.indent_spaces; }

  OutputStream& NewLine() {
    output_ << "\n";
    line_starts_.push_back(output_.tellp());
    output_ << std::string(indent_, ' ');
    return output_;
  }

  OutputStream output_;

 private:
  void MarkSpan(const ByteSpan& span, const ObjectPath& path);

  DocPrinterOptions options_;
  int indent_ = 0;
  std::vector<size_t> line_starts_;
  ObjectPath path_to_highlight_;
  size_t current_path_best_match_length_;
  std::vector<ByteSpan> current_highlight_candidates_;
  std::vector<ByteSpan> highlights_;
};

class PythonDocPrinter : public DocPrinter {
 public:
  PythonDocPrinter(const DocPrinterOptions& options) : DocPrinter(options) {}

 protected:
  using DocPrinter::PrintDoc;

  void PrintTypedDoc(const LiteralDoc& doc) final;
  void PrintTypedDoc(const SliceDoc& doc) final;
  void PrintTypedDoc(const IdDoc& doc) final;
  void PrintTypedDoc(const AttrAccessDoc& doc) final;
  void PrintTypedDoc(const IndexDoc& doc) final;
  void PrintTypedDoc(const OperationDoc& doc) final;
  void PrintTypedDoc(const CallDoc& doc) final;
  void PrintTypedDoc(const LambdaDoc& doc) final;
  void PrintTypedDoc(const ListDoc& doc) final;
  void PrintTypedDoc(const DictDoc& doc) final;
  void PrintTypedDoc(const TupleDoc& doc) final;
  void PrintTypedDoc(const StmtBlockDoc& doc) final;
  void PrintTypedDoc(const ExprStmtDoc& doc) final;
  void PrintTypedDoc(const ScopeDoc& doc) final;
  void PrintTypedDoc(const IfDoc& doc) final;
  void PrintTypedDoc(const WhileDoc& doc) final;
  void PrintTypedDoc(const ForDoc& doc) final;
  void PrintTypedDoc(const AssignDoc& doc) final;
  void PrintTypedDoc(const FunctionDoc& doc) final;

 private:
  void PrintStmtArray(const Array<StmtDoc>& docs);

  void PrintStmtBlock(const Array<StmtDoc>& docs);

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
    // TODO: Replace this with configurable prefix
    output_ << "T"
            << "." << primitive;
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

  void PrintBinaryOp(OperationDocNode::Kind operation_kind, const ExprDoc& a, const ExprDoc& b);
  void PrintSpecialOp(OperationDocNode::Kind operation_kind, const Array<ExprDoc>& oprands);

  void PrintStringLiteral(const String& string);
  void PrintNumberNode(const PrimExpr& expr);
};

}  // namespace printer
}  // namespace script
}  // namespace tvm

#endif

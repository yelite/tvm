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

#include "doc_printer.h"

#include "doc.h"
#include "tvm/runtime/logging.h"
#include "tvm/tir/expr.h"

namespace tvm {
namespace script {
namespace printer {

String DocPrinter::Print(Doc doc) { return Print(Array<Doc>({doc})); }

String DocPrinter::Print(Array<Doc> docs) {
  output_.str("");
  for (const Doc& doc : docs) {
    PrintDoc(doc);
  }
  std::string result = output_.str();
  output_.str("");
  return result;
}

void DocPrinter::PrintDoc(const Doc& doc) {
  if (const auto* doc_node = doc.as<LiteralDocNode>()) {
    PrintTypedDoc(GetRef<LiteralDoc>(doc_node));
  } else if (const auto* doc_node = doc.as<SliceDocNode>()) {
    PrintTypedDoc(GetRef<SliceDoc>(doc_node));
  } else if (const auto* doc_node = doc.as<IdDocNode>()) {
    PrintTypedDoc(GetRef<IdDoc>(doc_node));
  } else if (const auto* doc_node = doc.as<AttrAccessDocNode>()) {
    PrintTypedDoc(GetRef<AttrAccessDoc>(doc_node));
  } else if (const auto* doc_node = doc.as<IndexDocNode>()) {
    PrintTypedDoc(GetRef<IndexDoc>(doc_node));
  } else if (const auto* doc_node = doc.as<OperationDocNode>()) {
    PrintTypedDoc(GetRef<OperationDoc>(doc_node));
  } else if (const auto* doc_node = doc.as<CallDocNode>()) {
    PrintTypedDoc(GetRef<CallDoc>(doc_node));
  } else if (const auto* doc_node = doc.as<ListDocNode>()) {
    PrintTypedDoc(GetRef<ListDoc>(doc_node));
  } else if (const auto* doc_node = doc.as<DictDocNode>()) {
    PrintTypedDoc(GetRef<DictDoc>(doc_node));
  } else if (const auto* doc_node = doc.as<TupleDocNode>()) {
    PrintTypedDoc(GetRef<TupleDoc>(doc_node));
  } else if (const auto* doc_node = doc.as<StmtBlockDocNode>()) {
    PrintTypedDoc(GetRef<StmtBlockDoc>(doc_node));
  } else if (const auto* doc_node = doc.as<ExprStmtDocNode>()) {
    PrintTypedDoc(GetRef<ExprStmtDoc>(doc_node));
  } else if (const auto* doc_node = doc.as<ScopeDocNode>()) {
    PrintTypedDoc(GetRef<ScopeDoc>(doc_node));
  } else if (const auto* doc_node = doc.as<ForDocNode>()) {
    PrintTypedDoc(GetRef<ForDoc>(doc_node));
  } else if (const auto* doc_node = doc.as<AssignDocNode>()) {
    PrintTypedDoc(GetRef<AssignDoc>(doc_node));
  } else if (const auto* doc_node = doc.as<FunctionDocNode>()) {
    PrintTypedDoc(GetRef<FunctionDoc>(doc_node));
  } else {
    LOG(FATAL) << "Do not know how to print " << doc->GetTypeKey();
    throw;
  }
}

void PythonDocPrinter::PrintTypedDoc(const LiteralDoc& doc) {
  const ObjectRef& value = doc->value;
  if (value->IsInstance<FloatImmNode>() || value->IsInstance<IntImmNode>()) {
    PrintNumberNode(Downcast<PrimExpr>(doc->value));
  } else if (const auto* string_obj = value.as<StringObj>()) {
    PrintStringLiteral(GetRef<String>(string_obj));
  } else if (const auto* node = doc.as<tir::StringImmNode>()) {
    PrintStringLiteral(node->value);
  } else {
    ICHECK(false) << "Unsupported literal value type " << value->GetTypeKey();
  }
}

void PythonDocPrinter::PrintTypedDoc(const SliceDoc& doc) {
  output_ << "[";
  if (doc->start != nullptr) {
    PrintDoc(doc->start.value());
  }
  output_ << ":";
  if (doc->stop != nullptr) {
    PrintDoc(doc->stop.value());
  }
  output_ << "]";
}

void PythonDocPrinter::PrintTypedDoc(const IdDoc& doc) { output_ << doc->name; }

void PythonDocPrinter::PrintTypedDoc(const AttrAccessDoc& doc) {
  PrintDoc(doc->value);
  output_ << "." << doc->attr;
}

void PythonDocPrinter::PrintTypedDoc(const IndexDoc& doc) {
  PrintDoc(doc->value);
  PrintJoinedElements("[", doc->indices, ", ", "]", "()");
}

void PythonDocPrinter::PrintTypedDoc(const OperationDoc& doc) {
  PrintDoc(doc->operands[0]);
  // TODO: unary and ternary op
  using OpKind = OperationDocNode::Kind;
  switch (doc->kind) {
    case OpKind::kUndefined:
      ICHECK(false) << "Cannot print undefined operation doc";
    case OpKind::kAdd:
      output_ << " + ";
      break;
    case OpKind::kSub:
      output_ << " - ";
      break;
    case OpKind::kMul:
      output_ << " * ";
      break;
    case OpKind::kFloorDiv:
      output_ << " // ";
      break;
    case OpKind::kFloorMod:
      output_ << " % ";
      break;
  }
  PrintDoc(doc->operands[1]);
}

void PythonDocPrinter::PrintTypedDoc(const CallDoc& doc) {
  PrintDoc(doc->callee);
  PrintJoinedElements("(", doc->args, ", ", ")");
}

void PythonDocPrinter::PrintTypedDoc(const ListDoc& doc) {
  size_t size = doc->elements.size();
  if (size == 0) {
    output_ << "list()";
  } else {
    PrintJoinedElements("[", doc->elements, ", ", "]");
  }
}

void PythonDocPrinter::PrintTypedDoc(const DictDoc& doc) {
  size_t size = doc->keys.size();
  if (size == 0) {
    output_ << "dict()";
  } else {
    output_ << "{";
    size_t idx = 0;
    for (const ExprDoc& key : doc->keys) {
      if (idx > 0) {
        output_ << ", ";
      }
      PrintDoc(key);
      output_ << ": ";
      PrintDoc(doc->values[idx]);
      idx++;
    }
    output_ << "}";
  }
}

void PythonDocPrinter::PrintTypedDoc(const TupleDoc& doc) {
  size_t size = doc->elements.size();
  if (size == 0) {
    output_ << "tuple()";
  } else if (size == 1) {
    output_ << "(";
    PrintDoc(doc->elements[0]);
    output_ << ",)";
  } else {
    PrintJoinedElements("(", doc->elements, ", ", ")");
  }
}

void PythonDocPrinter::PrintTypedDoc(const StmtBlockDoc& doc) {
  for (const StmtDoc& stmt : doc->stmts) {
    PrintDoc(stmt);
  }
}

void PythonDocPrinter::PrintTypedDoc(const ExprStmtDoc& doc) {
  PrintDoc(doc->expr);
  NewLine();
}

void PythonDocPrinter::PrintTypedDoc(const ScopeDoc& doc) {
  output_ << "with ";
  PrintDoc(doc->lhs);
  if (doc->rhs != nullptr) {
    output_ << " as ";
    PrintDoc(doc->rhs.value());
  }
  output_ << ":";
  NewLine();

  PrintWithIncreasedIndent(doc->body);
}

void PythonDocPrinter::PrintTypedDoc(const ForDoc& doc) {
  output_ << "for ";
  PrintDoc(doc->lhs);
  output_ << " in ";
  PrintDoc(doc->rhs);
  output_ << ":";
  NewLine();

  PrintWithIncreasedIndent(doc->body);
}

void PythonDocPrinter::PrintTypedDoc(const AssignDoc& doc) {
  // TODO: Assign Kind, Type Annotation
  PrintDoc(doc->lhs);
  if (doc->annotation) {
    output_ << ": ";
    PrintDoc(doc->annotation.value());
  }
  if (doc->rhs) {
    output_ << " = ";
    PrintDoc(doc->rhs.value());
  }
  NewLine();
}

void PythonDocPrinter::PrintTypedDoc(const FunctionDoc& doc) {
  for (const ExprDoc& decorator : doc->decorators) {
    output_ << "@";
    PrintDoc(decorator);
    NewLine();
  }
  output_ << "def ";
  PrintDoc(doc->name);
  PrintJoinedElements("(", doc->args, ", ", ")");
  output_ << " -> ";
  PrintDoc(doc->return_type);
  output_ << ":";
  NewLine();
  PrintWithIncreasedIndent(doc->body);
}

void PythonDocPrinter::PrintStringLiteral(const String& string) {
  // TODO: Escape and smart quote (choose ' or " automatically)
  output_ << "\"" << string << "\"";
}

void PythonDocPrinter::PrintNumberNode(const PrimExpr& expr) {
  const DataType& dtype = expr->dtype;
  std::ostringstream number_value;

  if (const auto* int_node = expr.as<IntImmNode>()) {
    number_value << int_node->value;
  } else if (const auto* float_node = expr.as<FloatImmNode>()) {
    number_value.precision(17);
    number_value << float_node->value;
  } else {
    LOG(FATAL) << "Do not know how to process " << expr->GetTypeKey() << " as literal number";
  }

  if (dtype == DataType::Int(32)) {
    output_ << number_value.str();
  } else if (dtype == DataType::Bool()) {
    output_ << (Downcast<IntImm>(expr)->value ? "True" : "False");
  } else {
    PrintTIRPrimitiveCall(runtime::DLDataType2String(dtype), number_value.str());
  }
}

}  // namespace printer
}  // namespace script
}  // namespace tvm

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

namespace tvm {
namespace script {
namespace printer {

String DocPrinter::Print(std::initializer_list<Doc> docs) {
  output_.str("");
  for (const Doc& doc : docs) {
    indent_ = 0;
    PrintDoc(doc);
    output_ << "\n";
  }
  std::string result = output_.str();
  result.erase(result.begin(), std::find_if(result.begin(), result.end(),
                                            [](unsigned char ch) { return !std::isspace(ch); }));
  output_.str("");
  return result;
}

void DocPrinter::PrintDoc(const Doc& doc) {
  if (doc->IsInstance<LiteralValueDocNode>()) {
    PrintDoc(Downcast<LiteralValueDoc>(doc));
  } else if (doc->IsInstance<ConstDocNode>()) {
    PrintDoc(Downcast<ConstDoc>(doc));
  } else if (doc->IsInstance<IdentifierDocNode>()) {
    PrintDoc(Downcast<IdentifierDoc>(doc));
  } else if (doc->IsInstance<AttrAccessDocNode>()) {
    PrintDoc(Downcast<AttrAccessDoc>(doc));
  } else if (doc->IsInstance<IndexDocNode>()) {
    PrintDoc(Downcast<IndexDoc>(doc));
  } else if (doc->IsInstance<OperationDocNode>()) {
    PrintDoc(Downcast<OperationDoc>(doc));
  } else if (doc->IsInstance<CallDocNode>()) {
    PrintDoc(Downcast<CallDoc>(doc));
  } else if (doc->IsInstance<TupleDocNode>()) {
    PrintDoc(Downcast<TupleDoc>(doc));
  } else if (doc->IsInstance<SeqStmtDocNode>()) {
    PrintDoc(Downcast<SeqStmtDoc>(doc));
  } else if (doc->IsInstance<ScopeDocNode>()) {
    PrintDoc(Downcast<ScopeDoc>(doc));
  } else if (doc->IsInstance<ForDocNode>()) {
    PrintDoc(Downcast<ForDoc>(doc));
  } else if (doc->IsInstance<AssignDocNode>()) {
    PrintDoc(Downcast<AssignDoc>(doc));
  } else if (doc->IsInstance<ExprTypeDocNode>()) {
    PrintDoc(Downcast<ExprTypeDoc>(doc));
  } else if (doc->IsInstance<TypeCallDocNode>()) {
    PrintDoc(Downcast<TypeCallDoc>(doc));
  } else if (doc->IsInstance<FunctionDocNode>()) {
    PrintDoc(Downcast<FunctionDoc>(doc));
  } else if (doc->IsInstance<FunctionArgDocNode>()) {
    PrintDoc(Downcast<FunctionArgDoc>(doc));
  } else {
    LOG(FATAL) << "Do not know how to print " << doc->GetTypeKey();
    throw;
  }
}

void PythonDocPrinter::PrintDoc(const LiteralValueDoc& doc) {
  ObjectRef& value = doc->value;
  if (value->IsInstance<FloatImmNode>() || value->IsInstance<IntImmNode>()) {
    PrintNumberNode(Downcast<PrimExpr>(doc->value));
  } else if (value->IsInstance<StringObj>()) {
    PrintStringLiteral(Downcast<String>(value));
  } else if (const auto* node = doc.as<tir::StringImmNode>()) {
    PrintStringLiteral(node->value);
  }
}

void PythonDocPrinter::PrintDoc(const ConstDoc& doc) {
  switch (doc->kind) {
    case ConstDocNode::ConstKind::TIRBuilder:
      output_ << tir_prefix_;
      break;
    case ConstDocNode::ConstKind::RelaxBuilder:
      LOG(FATAL) << "RelaxBuilder constant not supported yet";
      break;
    case ConstDocNode::ConstKind::None:
      output_ << "None";
      break;
  }
}

void PythonDocPrinter::PrintDoc(const IdentifierDoc& doc) { output_ << doc->name; }

void PythonDocPrinter::PrintDoc(const AttrAccessDoc& doc) {
  PrintDoc(doc->value);
  output_ << ".";
  PrintDoc(doc->attr);
}

void PythonDocPrinter::PrintDoc(const IndexDoc& doc) {
  PrintDoc(doc->value);
  PrintJoinedElements("[", doc->indices, ", ", "]", "()");
}

void PythonDocPrinter::PrintDoc(const OperationDoc& doc) {
  PrintDoc(doc->operands[0]);
  switch (doc->kind) {
    case OperationDocNode::OperationKind::Add:
      output_ << " + ";
      break;
    case OperationDocNode::OperationKind::Sub:
      output_ << " - ";
      break;
    case OperationDocNode::OperationKind::Mul:
      output_ << " * ";
      break;
    case OperationDocNode::OperationKind::Div:
      output_ << " / ";
      break;
    case OperationDocNode::OperationKind::FloorDiv:
      output_ << " // ";
      break;
  }
  PrintDoc(doc->operands[1]);
}

void PythonDocPrinter::PrintDoc(const CallDoc& doc) {
  PrintDoc(doc->callee);
  PrintJoinedElements("(", doc->args, ", ", ")");
}

void PythonDocPrinter::PrintDoc(const TupleDoc& doc) {
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

void PythonDocPrinter::PrintDoc(const SeqStmtDoc& doc) {
  for (const StmtDoc& stmt : doc->seq) {
    PrintDoc(stmt);
  }
}

void PythonDocPrinter::PrintDoc(const ScopeDoc& doc) {
  NewLine() << "with ";
  PrintDoc(doc->scope);
  output_ << ":";

  PrintWithIncreasedIndent(doc->body);
}

void PythonDocPrinter::PrintDoc(const ForDoc& doc) {
  NewLine() << "for ";
  PrintDoc(doc->target);
  output_ << " in ";
  PrintDoc(doc->iter);
  output_ << ":";

  PrintWithIncreasedIndent(doc->body);
}

void PythonDocPrinter::PrintDoc(const AssignDoc& doc) {
  // TODO: Assign Kind, Type Annotation
  NewLine();
  PrintDoc(doc->target);
  if (doc->type) {
    output_ << ": ";
    PrintDoc(doc->type.value());
  }
  if (doc->value) {
    output_ << " = ";
    PrintDoc(doc->value.value());
  }
}

void PythonDocPrinter::PrintDoc(const ExprTypeDoc& doc) { PrintDoc(doc->expr); }

void PythonDocPrinter::PrintDoc(const TypeCallDoc& doc) {
  PrintDoc(doc->base);
  PrintJoinedElements("[", doc->args, ", ", "]");
}

void PythonDocPrinter::PrintDoc(const FunctionDoc& doc) {
  NewLine() << "@" << tir_prefix_ << ".prim_func";
  NewLine() << "def ";
  PrintDoc(doc->name);
  PrintJoinedElements("(", doc->args, ", ", ")");
  output_ << " -> ";
  PrintDoc(doc->return_type);
  output_ << ":";
  PrintWithIncreasedIndent(doc->body);
}

void PythonDocPrinter::PrintDoc(const FunctionArgDoc& doc) {
  PrintDoc(doc->name);
  output_ << ": ";
  PrintDoc(doc->type);
}

void PythonDocPrinter::PrintStringLiteral(const String& string) {
  // TODO: Escape and smart quote (choose ' or " automatically)
  output_ << "\"" << string << "\"";
}

void PythonDocPrinter::PrintNumberNode(const PrimExpr& expr) {
  const DataType& dtype = expr->dtype;
  std::ostringstream number_value;

  if (expr->IsInstance<IntImmNode>()) {
    number_value << Downcast<IntImm>(expr)->value;
  } else if (expr->IsInstance<FloatImmNode>()) {
    number_value.precision(17);
    number_value << Downcast<FloatImm>(expr)->value;
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

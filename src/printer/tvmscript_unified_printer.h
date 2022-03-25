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
#ifndef TVM_TVMSCRIPT_UNIFIED_PRINTER_H_
#define TVM_TVMSCRIPT_UNIFIED_PRINTER_H_

#include <tvm/node/node.h>
#include <tvm/runtime/container/map.h>

#include <initializer_list>
#include <utility>

#include "tvm/relay/expr.h"
#include "tvm/runtime/container/array.h"
#include "tvm/runtime/object.h"

namespace tvm {
namespace script {

// Code Doc

class DocNode : public Object {
 public:
  ObjectRef origin_ir_node;

  DocNode() = default;
  virtual ~DocNode() = default;

  static constexpr const char* _type_key = "script.Docs.Doc";
  TVM_DECLARE_BASE_OBJECT_INFO(DocNode, Object);
};

class Doc : public ObjectRef {
 public:
  Doc() : Doc(make_object<DocNode>()){};
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(Doc, ObjectRef, DocNode);
};

// Base Expr Doc

class ExprDocNode : public DocNode {
 public:
  static constexpr const char* _type_key = "script.Docs.BaseExprDoc";
  TVM_DECLARE_BASE_OBJECT_INFO(ExprDocNode, DocNode);
};

class ExprDoc : public Doc {
 public:
  static ExprDoc TIRBuilder();
  static ExprDoc None();

  template <typename... AttrType>
  static ExprDoc TIRBuilderAttribute(AttrType&&... name);

  ExprDoc AccessAttr(String attr);

  template <typename... ArgType>
  ExprDoc CallWith(ArgType&&... args);

  template <typename... IndexType>
  ExprDoc IndexWith(IndexType&&... args);

  ExprDoc() : ExprDoc(make_object<ExprDocNode>()){};
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(ExprDoc, Doc, ExprDocNode);
};

// Literal Value Doc

class LiteralValueDocNode : public ExprDocNode {
 public:
  ObjectRef value;  // Can only be FloatImm/IntImm/StringImm/String

  static constexpr const char* _type_key = "script.Docs.LiteralValueDoc";
  TVM_DECLARE_FINAL_OBJECT_INFO(LiteralValueDocNode, ExprDocNode);

 private:
  ObjectRef value_;
};

class LiteralValueDoc : public ExprDoc {
 public:
  LiteralValueDoc() : LiteralValueDoc(make_object<LiteralValueDocNode>()){};
  LiteralValueDoc(IntImm val) : LiteralValueDoc(static_cast<ObjectRef>(val)) {}
  LiteralValueDoc(FloatImm val) : LiteralValueDoc(static_cast<ObjectRef>(val)) {}
  LiteralValueDoc(tir::StringImm val) : LiteralValueDoc(static_cast<ObjectRef>(val)) {}
  LiteralValueDoc(String val) : LiteralValueDoc(static_cast<ObjectRef>(val)) {}

  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(LiteralValueDoc, ExprDoc, LiteralValueDocNode);

 protected:
  explicit LiteralValueDoc(ObjectRef value) {
    auto node = make_object<LiteralValueDocNode>();
    node->value = std::move(value);
    data_ = std::move(node);
  }
};

// Special Constant Doc

class ConstDocNode : public ExprDocNode {
 public:
  enum class ConstKind { TIRBuilder, RelaxBuilder, None };

  ConstKind kind;

  ConstDocNode(ConstKind kind) : kind(kind) {}

  static constexpr const char* _type_key = "script.Docs.ConstDoc";
  TVM_DECLARE_FINAL_OBJECT_INFO(ConstDocNode, ExprDocNode);
};

class ConstDoc : public ExprDoc {
 public:
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(ConstDoc, ExprDoc, ConstDocNode);
};

// Identifier Doc

class IdentifierDocNode : public ExprDocNode {
 public:
  String name;

  static constexpr const char* _type_key = "script.Docs.IdentidierDoc";
  TVM_DECLARE_FINAL_OBJECT_INFO(IdentifierDocNode, ExprDocNode);
};

class IdentifierDoc : public ExprDoc {
 public:
  IdentifierDoc(String name) {
    auto node = make_object<IdentifierDocNode>();
    node->name = std::move(name);
    data_ = std::move(node);
  };

  IdentifierDoc() : IdentifierDoc(make_object<IdentifierDocNode>()){};
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(IdentifierDoc, ExprDoc, IdentifierDocNode);
};

// Attr Access Doc

class AttrAccessDocNode : public ExprDocNode {
 public:
  ExprDoc value;
  IdentifierDoc attr;

  static constexpr const char* _type_key = "script.Docs.AttrAccessDoc";
  TVM_DECLARE_FINAL_OBJECT_INFO(AttrAccessDocNode, ExprDocNode);
};

class AttrAccessDoc : public ExprDoc {
 public:
  AttrAccessDoc() : AttrAccessDoc(make_object<AttrAccessDocNode>()){};
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(AttrAccessDoc, ExprDoc, AttrAccessDocNode);
};

// Index Doc

class IndexDocNode : public ExprDocNode {
 public:
  ExprDoc value;
  Array<ExprDoc> indices;

  static constexpr const char* _type_key = "script.Docs.IndexDoc";
  TVM_DECLARE_FINAL_OBJECT_INFO(IndexDocNode, ExprDocNode);
};

class IndexDoc : public ExprDoc {
 public:
  IndexDoc() : IndexDoc(make_object<IndexDocNode>()){};
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(IndexDoc, ExprDoc, IndexDocNode);
};

// Operation Doc (Unary, Binary and Ternary operations)

class OperationDocNode : public ExprDocNode {
 public:
  enum class OperationKind { Add, Sub, Mul, Div, FloorDiv };

  OperationKind kind;
  Array<ExprDoc> operands;

  static constexpr const char* _type_key = "script.Docs.OperationDoc";
  TVM_DECLARE_FINAL_OBJECT_INFO(OperationDocNode, ExprDocNode);
};

class OperationDoc : public ExprDoc {
 public:
  OperationDoc() : OperationDoc(make_object<OperationDocNode>()){};
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(OperationDoc, ExprDoc, OperationDocNode);
};

// Call Doc

class CallDocNode : public ExprDocNode {
 public:
  ExprDoc callee;
  Array<ExprDoc> args;
  Array<String> keyword_arg_names;
  Array<ExprDoc> keyword_arg_values;

  static constexpr const char* _type_key = "script.Docs.CallDoc";
  TVM_DECLARE_FINAL_OBJECT_INFO(CallDocNode, ExprDocNode);
};

class CallDoc : public ExprDoc {
 public:
  CallDoc() : CallDoc(make_object<CallDocNode>()){};
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(CallDoc, ExprDoc, CallDocNode);
};

// Tuple Doc

class TupleDocNode : public ExprDocNode {
 public:
  Array<ExprDoc> elements;

  static constexpr const char* _type_key = "script.Docs.TupleDoc";
  TVM_DECLARE_FINAL_OBJECT_INFO(TupleDocNode, ExprDocNode);
};

class TupleDoc : public ExprDoc {
 public:
  TupleDoc(std::initializer_list<ExprDoc> elements) {
    auto node = make_object<TupleDocNode>();
    node->elements = std::move(elements);
    data_ = std::move(node);
  }

  TupleDoc() : TupleDoc(make_object<TupleDocNode>()){};
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(TupleDoc, ExprDoc, TupleDocNode);
};

// Helper functions for ExprDoc

inline ExprDoc ExprDoc::TIRBuilder() {
  return ConstDoc(make_object<ConstDocNode>(ConstDocNode::ConstKind::TIRBuilder));
}

inline ExprDoc ExprDoc::None() {
  return ConstDoc(make_object<ConstDocNode>(ConstDocNode::ConstKind::None));
}

template <typename... AttrType>
inline ExprDoc ExprDoc::TIRBuilderAttribute(AttrType&&... names) {
  auto result = TIRBuilder();
  for (auto& name : std::initializer_list<std::common_type_t<AttrType...>>{std::forward<AttrType>(names)...}) {
    result = result.AccessAttr(name);
  }
  return result;
}

inline ExprDoc ExprDoc::AccessAttr(String attr) {
  AttrAccessDoc expr;
  expr->value = *this;
  expr->attr = attr;
  return expr;
}

template <typename... ArgType>
inline ExprDoc ExprDoc::CallWith(ArgType&&... args) {
  CallDoc expr;
  expr->callee = *this;
  expr->args = {std::forward<ArgType>(args)...};
  return expr;
}

template <typename... IndexType>
inline ExprDoc ExprDoc::IndexWith(IndexType&&... args) {
  IndexDoc expr;
  expr->value = *this;
  expr->indices = {std::forward<IndexType>(args)...};
  return expr;
}

// Type Doc

class TypeDocNode : public DocNode {
 public:
  static constexpr const char* _type_key = "script.Docs.TypeDoc";
  TVM_DECLARE_BASE_OBJECT_INFO(TypeDocNode, DocNode);
};

class TypeDoc : public Doc {
 public:
  TypeDoc() : TypeDoc(make_object<TypeDocNode>()){};

  template <typename... AttrType>
  static TypeDoc TIRPrimitive(AttrType&&... names);
  static TypeDoc NoneType();

  template <typename... ArgType>
  TypeDoc CallWith(ArgType&&... args);

  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(TypeDoc, Doc, TypeDocNode);
};

// Expr Type Doc

class ExprTypeDocNode : public TypeDocNode {
 public:
  ExprDoc expr;

  static constexpr const char* _type_key = "script.Docs.ExprTypeDoc";
  TVM_DECLARE_BASE_OBJECT_INFO(ExprTypeDocNode, TypeDocNode);
};

class ExprTypeDoc : public TypeDoc {
 public:
  explicit ExprTypeDoc(ExprDoc expr) {
    auto node = make_object<ExprTypeDocNode>();
    node->expr = std::move(expr);
    data_ = std::move(node);
  }

  static ExprTypeDoc TIRPrimitive(String type_name);

  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(ExprTypeDoc, TypeDoc, ExprTypeDocNode);
};

// Type Call Doc

/*!
 * Abstract the difference of A[n] and A(n) in different languages
 */
class TypeCallDocNode : public TypeDocNode {
 public:
  TypeDoc base;
  Array<TypeDoc> args;

  static constexpr const char* _type_key = "script.Docs.TypeCall";
  TVM_DECLARE_BASE_OBJECT_INFO(TypeCallDocNode, TypeDocNode);
};

class TypeCallDoc : public TypeDoc {
 public:
  TypeCallDoc() : TypeCallDoc(make_object<TypeCallDocNode>()){};
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(TypeCallDoc, TypeDoc, TypeCallDocNode);
};

// Helper methods for type

template <typename... AttrType>
inline TypeDoc TypeDoc::TIRPrimitive(AttrType&&... names) {
  auto expr = ExprDoc::TIRBuilderAttribute(names...);
  return ExprTypeDoc(std::move(expr));
}

inline TypeDoc TypeDoc::NoneType() { return ExprTypeDoc(ExprDoc::None()); }

template <typename... ArgType>
inline TypeDoc TypeDoc::CallWith(ArgType&&... args) {
  TypeCallDoc ty;
  ty->base = *this;
  ty->args = {std::forward<ArgType>(args)...};
  return ty;
}

// Base Stmt Doc

class StmtDocNode : public DocNode {
 public:
  static constexpr const char* _type_key = "script.Docs.StmtDoc";
  TVM_DECLARE_BASE_OBJECT_INFO(StmtDocNode, DocNode);
};

class StmtDoc : public Doc {
 public:
  StmtDoc() : StmtDoc(make_object<StmtDocNode>()){};
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(StmtDoc, Doc, StmtDocNode);
};

// Seq Stmt Doc

class SeqStmtDocNode : public StmtDocNode {
 public:
  Array<StmtDoc> seq;

  static constexpr const char* _type_key = "script.Docs.SeqStmtDoc";
  TVM_DECLARE_FINAL_OBJECT_INFO(SeqStmtDocNode, StmtDocNode);
};

class SeqStmtDoc : public StmtDoc {
 public:
  SeqStmtDoc() : SeqStmtDoc(make_object<SeqStmtDocNode>()){};
  SeqStmtDoc(Array<StmtDoc> stmts) {  // NOLINT(*)
    auto node = make_object<SeqStmtDocNode>();
    node->seq = std::move(stmts);
    data_ = std::move(node);
  }

  // TODO: Unpack StmtDoc if it's SeqStmtDoc
  SeqStmtDoc& Add(const StmtDoc& stmt) {
    get()->seq.push_back(stmt);
    return *this;
  }

  SeqStmtDoc& Extend(SeqStmtDoc stmts) {
    for (auto& s : stmts->seq) {
      this->operator->()->seq.push_back(s);
    }
    return *this;
  }

  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(SeqStmtDoc, StmtDoc, SeqStmtDocNode);
};

// Assign Doc

class AssignDocNode : public StmtDocNode {
 public:
  enum class AssignKind {
    Regular,
    Addition,
  };

  AssignKind kind = AssignKind::Regular;
  ExprDoc target;
  Optional<ExprDoc> value;  // If null, this doc represents declaration, e.g. `A: T.Buffer[(1,2)]`
  Optional<TypeDoc> type;

  static constexpr const char* _type_key = "script.Docs.AssignDoc";
  TVM_DECLARE_FINAL_OBJECT_INFO(AssignDocNode, StmtDocNode);
};

class AssignDoc : public StmtDoc {
 public:
  AssignDoc() : AssignDoc(make_object<AssignDocNode>()){};
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(AssignDoc, StmtDoc, AssignDocNode);
};

// For Doc

class ForDocNode : public StmtDocNode {
 public:
  ExprDoc target;
  ExprDoc iter;
  StmtDoc body;

  static constexpr const char* _type_key = "script.Docs.ForDoc";
  TVM_DECLARE_FINAL_OBJECT_INFO(ForDocNode, StmtDocNode);
};

class ForDoc : public StmtDoc {
 public:
  ForDoc() : ForDoc(make_object<ForDocNode>()){};
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(ForDoc, StmtDoc, ForDocNode);
};

// Scope Doc

class ScopeDocNode : public StmtDocNode {
 public:
  ExprDoc scope;
  StmtDoc body;

  static constexpr const char* _type_key = "script.Docs.ScopeDoc";
  TVM_DECLARE_FINAL_OBJECT_INFO(ScopeDocNode, StmtDocNode);
};

class ScopeDoc : public StmtDoc {
 public:
  ScopeDoc() : ScopeDoc(make_object<ScopeDocNode>()){};
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(ScopeDoc, StmtDoc, ScopeDocNode);
};

// ExprStmt Doc
class ExprStmtDocNode : public StmtDocNode {
 public:
  ExprDoc expr;

  static constexpr const char* _type_key = "script.Docs.ExprStmtDoc";
  TVM_DECLARE_FINAL_OBJECT_INFO(ExprStmtDocNode, StmtDocNode);
};

class ExprStmtDoc : public StmtDoc {
 public:
  ExprStmtDoc() : ExprStmtDoc(make_object<ExprStmtDocNode>()){};
  ExprStmtDoc(ExprDoc expr) {
    auto node = make_object<ExprStmtDocNode>();
    node->expr = std::move(expr);
    data_ = std::move(node);
  };
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(ExprStmtDoc, StmtDoc, ExprStmtDocNode);
};

// Function Arg Doc

class FunctionArgDocNode : public DocNode {
 public:
  IdentifierDoc name;
  TypeDoc type;

  static constexpr const char* _type_key = "script.Docs.FunctionArgDoc";
  TVM_DECLARE_FINAL_OBJECT_INFO(FunctionArgDocNode, DocNode);
};

class FunctionArgDoc : public Doc {
 public:
  FunctionArgDoc() : FunctionArgDoc(make_object<FunctionArgDocNode>()){};
  FunctionArgDoc(IdentifierDoc name, TypeDoc type) {
    auto node = make_object<FunctionArgDocNode>();
    node->name = std::move(name);
    node->type = std::move(type);
    data_ = std::move(node);
  }

  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(FunctionArgDoc, Doc, FunctionArgDocNode);
};

// Function Doc

class FunctionDocNode : public DocNode {
 public:
  IdentifierDoc name;
  Array<FunctionArgDoc> args;
  TypeDoc return_type;
  StmtDoc body;

  static constexpr const char* _type_key = "script.Docs.FunctionDoc";
  TVM_DECLARE_FINAL_OBJECT_INFO(FunctionDocNode, DocNode);
};

class FunctionDoc : public Doc {
 public:
  FunctionDoc() : FunctionDoc(make_object<FunctionDocNode>()){};
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(FunctionDoc, Doc, FunctionDocNode);
};

// Printer Context

class PrinterBaseContextNode : public Object {
 public:
  static constexpr const char* _type_key = "script.PrinterBaseContext";
  TVM_DECLARE_BASE_OBJECT_INFO(PrinterBaseContextNode, Object);
};

class PrinterBaseContext : public ObjectRef {
 public:
  PrinterBaseContext() { data_ = make_object<PrinterBaseContextNode>(); }
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(PrinterBaseContext, runtime::ObjectRef,
                                                    PrinterBaseContextNode);
};

class PrinterFunctionContextNode : public PrinterBaseContextNode {
 public:
  static constexpr const char* _type_key = "script.PrinterFunctionContext";
  TVM_DECLARE_FINAL_OBJECT_INFO(PrinterFunctionContextNode, Object);
};

class PrinterFunctionContext : public PrinterBaseContext {
 public:
  PrinterFunctionContext() { data_ = make_object<PrinterFunctionContextNode>(); }
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(PrinterFunctionContext, PrinterBaseContext,
                                                    PrinterFunctionContextNode);
};

class PrinterLoopContextNode : public PrinterBaseContextNode {
 public:
  static constexpr const char* _type_key = "script.PrinterLoopContext";
  TVM_DECLARE_FINAL_OBJECT_INFO(PrinterLoopContextNode, Object);
};

class PrinterLoopContext : public PrinterBaseContext {
 public:
  PrinterLoopContext() { data_ = make_object<PrinterLoopContextNode>(); }
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(PrinterLoopContext, PrinterBaseContext,
                                                    PrinterLoopContextNode);
};

class PrinterBlockContextNode : public PrinterBaseContextNode {
 public:
  static constexpr const char* _type_key = "script.PrinterBlockContext";
  TVM_DECLARE_FINAL_OBJECT_INFO(PrinterBlockContextNode, Object);
};

class PrinterBlockContext : public PrinterBaseContext {
 public:
  PrinterBlockContext() { data_ = make_object<PrinterBlockContextNode>(); }
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(PrinterBlockContext, PrinterBaseContext,
                                                    PrinterBlockContextNode);
};

// TODO: Needs more thought on this
class PrinterContextManagerNode : public Object {
 public:
  Array<PrinterBaseContext> contexts;
  std::vector<Map<String, ObjectRef>> symbol_tables;

  template <typename ContextType>
  ContextType EnterContext() {
    ContextType context;
    contexts.push_back(context);
    PushNewSymbolTable();
    return context;
  }

  void ExitContext(PrinterBaseContext&& context) {
    // ICHECK_EQ(context, contexts.back())
    contexts.pop_back();
    symbol_tables.pop_back();
  }

  void AddVar(const tir::Buffer& buffer) { symbol_tables.back().Set(buffer->name, buffer); }

  void AddVar(const tir::Var& var) { symbol_tables.back().Set(var->name_hint, var); }

  Optional<ObjectRef> GetVar(const String& name) {
    if (symbol_tables.empty()) {
      return Optional<ObjectRef>();
    }
    auto current_table = symbol_tables.back();
    return current_table.Get(name);
  }

  static constexpr const char* _type_key = "script.PrinterContextManager";
  TVM_DECLARE_FINAL_OBJECT_INFO(PrinterContextManagerNode, Object);

 private:
  void PushNewSymbolTable() {
    if (symbol_tables.empty()) {
      symbol_tables.emplace_back();
    } else {
      symbol_tables.emplace_back(symbol_tables.back());
    }
  }
};

class PrinterContextManager : public ObjectRef {
 public:
  PrinterContextManager() { data_ = make_object<PrinterContextManagerNode>(); }
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(PrinterContextManager, runtime::ObjectRef,
                                                    PrinterContextManagerNode);
};

}  // namespace script
}  // namespace tvm

#endif  // TVM_TVMSCRIPT_UNIFIED_PRINTER_H_

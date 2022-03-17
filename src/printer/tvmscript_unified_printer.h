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

#include "tvm/relay/expr.h"
#include "tvm/runtime/object.h"

namespace tvm {

// Code Doc

class CodeDocNode : public Object {
 public:
  ObjectRef origin_ir_node;

  CodeDocNode() = default;
  virtual ~CodeDocNode() = default;

  static constexpr const char* _type_key = "script.Docs.Doc";
  TVM_DECLARE_BASE_OBJECT_INFO(CodeDocNode, Object);
};

class CodeDoc : public ObjectRef {
 public:
  CodeDoc() : CodeDoc(make_object<CodeDocNode>()){};
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(CodeDoc, ObjectRef, CodeDocNode);
};

// Base Expr Doc

class ExprDocNode : public CodeDocNode {
 public:
  static constexpr const char* _type_key = "script.Docs.BaseExprDoc";
  TVM_DECLARE_BASE_OBJECT_INFO(ExprDocNode, CodeDocNode);
};

class ExprDoc : public CodeDoc {
 public:
  static ExprDoc TIRBuilder();
  static ExprDoc None();
  static ExprDoc TIRBuilderAttribute(String name);
  ExprDoc AccessAttr(String attr);

  template <typename T>
  ExprDoc CallWith(T&& args);
  template <typename T>
  ExprDoc IndexWith(T&& args);

  ExprDoc() : ExprDoc(make_object<ExprDocNode>()){};
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(ExprDoc, CodeDoc, ExprDocNode);
};

// String Doc

class LiteralStringDocNode : public ExprDocNode {
 public:
  String value;

  static constexpr const char* _type_key = "script.Docs.StirngDoc";
  TVM_DECLARE_FINAL_OBJECT_INFO(LiteralStringDocNode, ExprDocNode);
};

class LiteralStringDoc : public ExprDoc {
 public:
  LiteralStringDoc(String value) {
    auto node = make_object<LiteralStringDocNode>();
    node->value = std::move(value);
    data_ = std::move(node);
  }

  LiteralStringDoc() : LiteralStringDoc(make_object<LiteralStringDocNode>()){};
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(LiteralStringDoc, ExprDoc,
                                                    LiteralStringDocNode);
};

// Number Doc

class LiteralNumberDocNode : public ExprDocNode {
 public:
  ObjectRef value;  // Can only be FloatImm or IntImm

  static constexpr const char* _type_key = "script.Docs.NumberDoc";
  TVM_DECLARE_FINAL_OBJECT_INFO(LiteralNumberDocNode, ExprDocNode);

 private:
  ObjectRef value_;
};

class LiteralNumberDoc : public ExprDoc {
 public:
  LiteralNumberDoc() : LiteralNumberDoc(make_object<LiteralNumberDocNode>()){};
  LiteralNumberDoc(IntImm val) : LiteralNumberDoc(static_cast<ObjectRef>(val)) {}
  LiteralNumberDoc(FloatImm val) : LiteralNumberDoc(static_cast<ObjectRef>(val)) {}

  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(LiteralNumberDoc, ExprDoc,
                                                    LiteralNumberDocNode);

 protected:
  explicit LiteralNumberDoc(ObjectRef value) {
    auto node = make_object<LiteralNumberDocNode>();
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

// Name Doc

class IdentifierDocNode : public ExprDocNode {
 public:
  String name;

  IdentifierDocNode() {}
  IdentifierDocNode(const String& name) : name(name) {}
  IdentifierDocNode(String&& name) : name(std::move(name)) {}

  static constexpr const char* _type_key = "script.Docs.NameDoc";
  TVM_DECLARE_FINAL_OBJECT_INFO(IdentifierDocNode, ExprDocNode);
};

class IdentifierDoc : public ExprDoc {
 public:
  IdentifierDoc(String name) { data_ = make_object<IdentifierDocNode>(std::move(name)); };

  IdentifierDoc() : IdentifierDoc(make_object<IdentifierDocNode>()){};
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(IdentifierDoc, ExprDoc, IdentifierDocNode);
};

// Attr Access Doc

class AttrAccessDocNode : public ExprDocNode {
 public:
  ExprDoc base;
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
  Array<ExprDoc> index;

  static constexpr const char* _type_key = "script.Docs.IndexDoc";
  TVM_DECLARE_FINAL_OBJECT_INFO(IndexDocNode, ExprDocNode);
};

class IndexDoc : public ExprDoc {
 public:
  IndexDoc() : IndexDoc(make_object<IndexDocNode>()){};
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(IndexDoc, ExprDoc, IndexDocNode);
};

// BinOp Doc

class BinOpDocNode : public ExprDocNode {
 public:
  enum class BinOpKind { Add, Sub, Mul, Div, FloorDiv };

  BinOpKind kind;
  ExprDoc lhs;
  ExprDoc rhs;

  static constexpr const char* _type_key = "script.Docs.BinOpDoc";
  TVM_DECLARE_FINAL_OBJECT_INFO(BinOpDocNode, ExprDocNode);
};

class BinOpDoc : public ExprDoc {
 public:
  BinOpDoc() : BinOpDoc(make_object<BinOpDocNode>()){};
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(BinOpDoc, ExprDoc, BinOpDocNode);
};

// Call Doc

class CallDocNode : public ExprDocNode {
 public:
  ExprDoc callee;
  Array<ExprDoc> args;

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
  TVM_DECLARE_FINAL_OBJECT_INFO(CallDocNode, ExprDocNode);
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

inline ExprDoc ExprDoc::TIRBuilderAttribute(String name) { return TIRBuilder().AccessAttr(name); }

inline ExprDoc ExprDoc::AccessAttr(String attr) {
  AttrAccessDoc expr;
  expr->base = *this;
  expr->attr = attr;
  return expr;
}

template <typename T>
inline ExprDoc ExprDoc::CallWith(T&& args) {
  CallDoc expr;
  expr->callee = *this;
  expr->args = std::forward<T>(args);
  return expr;
}

template <typename T>
inline ExprDoc ExprDoc::IndexWith(T&& args) {
  IndexDoc expr;
  expr->value = *this;
  expr->index = std::forward<T>(args);
  return expr;
}
// Base Stmt Doc

class StmtDocNode : public CodeDocNode {
 public:
  static constexpr const char* _type_key = "script.Docs.StmtDoc";
  TVM_DECLARE_BASE_OBJECT_INFO(StmtDocNode, CodeDocNode);
};

class StmtDoc : public CodeDoc {
 public:
  StmtDoc() : StmtDoc(make_object<StmtDocNode>()){};
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(StmtDoc, CodeDoc, StmtDocNode);
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
    this->operator->()->seq.push_back(stmt);
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
  ExprDoc value;

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

// Type Doc

class TypeDocNode : public CodeDocNode {
 public:
  static constexpr const char* _type_key = "script.Docs.TypeDoc";
  TVM_DECLARE_BASE_OBJECT_INFO(TypeDocNode, CodeDocNode);
};

class TypeDoc : public CodeDoc {
 public:
  TypeDoc() : TypeDoc(make_object<TypeDocNode>()){};

  static TypeDoc TIRPrimitive(String name);
  static TypeDoc NoneType();
  TypeDoc CallWith(std::initializer_list<TypeDoc> args);
  TypeDoc CallWith(std::vector<TypeDoc> args);

  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(TypeDoc, CodeDoc, TypeDocNode);
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
  Array<TypeDoc> params;

  static constexpr const char* _type_key = "script.Docs.TypeCall";
  TVM_DECLARE_BASE_OBJECT_INFO(TypeCallDocNode, TypeDocNode);
};

class TypeCallDoc : public TypeDoc {
 public:
  TypeCallDoc() : TypeCallDoc(make_object<TypeCallDocNode>()){};
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(TypeCallDoc, TypeDoc, TypeCallDocNode);
};

// Helper methods for type

inline TypeDoc TypeDoc::TIRPrimitive(String name) {
  auto expr = ExprDoc::TIRBuilderAttribute(name);
  return ExprTypeDoc(std::move(expr));
}

inline TypeDoc TypeDoc::NoneType() { return ExprTypeDoc(ExprDoc::None()); }

inline TypeDoc TypeDoc::CallWith(std::initializer_list<TypeDoc> args) {
  TypeCallDoc ty;
  ty->base = *this;
  ty->params = args;
  return ty;
}

inline TypeDoc TypeDoc::CallWith(std::vector<TypeDoc> args) {
  TypeCallDoc ty;
  ty->base = *this;
  ty->params = args;
  return ty;
}

// Function Arg Doc

class FunctionArgDocNode : public CodeDocNode {
 public:
  IdentifierDoc ident;
  TypeDoc type;

  static constexpr const char* _type_key = "script.Docs.FunctionArgDoc";
  TVM_DECLARE_FINAL_OBJECT_INFO(FunctionArgDocNode, CodeDocNode);
};

class FunctionArgDoc : public CodeDoc {
 public:
  FunctionArgDoc() : FunctionArgDoc(make_object<FunctionArgDocNode>()){};
  FunctionArgDoc(IdentifierDoc identifier, TypeDoc type) {
    auto node = make_object<FunctionArgDocNode>();
    node->ident = std::move(identifier);
    node->type = std::move(type);
    data_ = std::move(node);
  }

  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(FunctionArgDoc, CodeDoc, FunctionArgDocNode);
};

// Function Doc

class FunctionDocNode : public CodeDocNode {
 public:
  IdentifierDoc name;
  Array<FunctionArgDoc> args;
  TypeDoc return_type;
  StmtDoc body;

  static constexpr const char* _type_key = "script.Docs.FunctionDoc";
  TVM_DECLARE_FINAL_OBJECT_INFO(FunctionDocNode, CodeDocNode);
};

class FunctionDoc : public CodeDoc {
 public:
  FunctionDoc() : FunctionDoc(make_object<FunctionDocNode>()){};
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(FunctionDoc, CodeDoc, FunctionDocNode);
};

}  // namespace tvm

#endif  // TVM_TVMSCRIPT_UNIFIED_PRINTER_H_

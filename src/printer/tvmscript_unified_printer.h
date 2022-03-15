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
#include "tvm/relay/expr.h"

namespace tvm {

// Code Doc

class DocNode : public Object {
 public:
  ObjectRef origin_ir_node;

  DocNode() = default;
  virtual ~DocNode() = default;

  static constexpr const char* _type_key = "script.Docs.CodeDoc";
  TVM_DECLARE_BASE_OBJECT_INFO(DocNode, Object);
};

class Doc : public ObjectRef {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(Doc, ObjectRef, DocNode);
};

// Base Expr Doc

class ExprDocNode : public DocNode {
 public:
  static constexpr const char* _type_key = "script.Docs.BaseExprDoc";
  TVM_DECLARE_BASE_OBJECT_INFO(ExprDocNode, DocNode);
};

class ExprDoc : public Doc {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(ExprDoc, Doc, ExprDocNode);
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

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(LiteralStringDoc, ExprDoc, LiteralStringDocNode);
};

// Number Doc

class LiteralNumberDocNode : public ExprDocNode {
 public:
  static constexpr const char* _type_key = "script.Docs.NumberDoc";
  TVM_DECLARE_FINAL_OBJECT_INFO(LiteralNumberDocNode, ExprDocNode);

 private:
  ObjectRef value_;
};

class LiteralNumberDoc : public ExprDoc {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(LiteralNumberDoc, ExprDoc, LiteralNumberDocNode);
};

// TIR Builder Doc

class TIRBuilderConstDocNode : public ExprDocNode {
 public:
  static constexpr const char* _type_key = "script.Docs.TIRBuilderConstDoc";
  TVM_DECLARE_FINAL_OBJECT_INFO(TIRBuilderConstDocNode, ExprDocNode);
};

class TIRBuilderConstDoc : public ExprDoc {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(TIRBuilderConstDoc, ExprDoc,
                                        TIRBuilderConstDocNode);
};

// Name Doc

class IdentifierDocNode : public ExprDocNode {
 public:
  String name;

  IdentifierDocNode(String name) : name(std::move(name)) {}

  static constexpr const char* _type_key = "script.Docs.NameDoc";
  TVM_DECLARE_FINAL_OBJECT_INFO(IdentifierDocNode, ExprDocNode);
};

class IdentifierDoc : public ExprDoc {
 public:
  IdentifierDoc(String name) { data_ = make_object<IdentifierDocNode>(std::move(name)); };

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(IdentifierDoc, ExprDoc, IdentifierDocNode);
};

// Attr Doc

class MemberAccessDocNode : public ExprDocNode {
 public:
  ExprDoc value;
  IdentifierDoc ident;

  static constexpr const char* _type_key = "script.Docs.MemberAccessDoc";
  TVM_DECLARE_FINAL_OBJECT_INFO(MemberAccessDocNode, ExprDocNode);
};

class MemberAccessDoc : public ExprDoc {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(MemberAccessDoc, ExprDoc, MemberAccessDocNode);
};

// Index Doc

class IndexDocNode : public ExprDocNode {
 public:
  ExprDoc value;
  ExprDoc index;

  static constexpr const char* _type_key = "script.Docs.IndexDoc";
  TVM_DECLARE_FINAL_OBJECT_INFO(IndexDocNode, ExprDocNode);
};

class IndexDoc : public ExprDoc {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(IndexDoc, ExprDoc, IndexDocNode);
};

// BinOp Doc

class BinOpDocNode : public ExprDocNode {
 public:
  enum class BinOpKind {
    Plus,
    Minus,
  };

  BinOpKind kind;
  ExprDoc lhs;
  ExprDoc rhs;

  static constexpr const char* _type_key = "script.Docs.BinOpDoc";
  TVM_DECLARE_FINAL_OBJECT_INFO(BinOpDocNode, ExprDocNode);
};

class BinOpDoc : public ExprDoc {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(BinOpDoc, ExprDoc, BinOpDocNode);
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
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(CallDoc, ExprDoc, CallDocNode);
};

// Base Stmt Doc

class StmtDocNode : public DocNode {
 public:
  static constexpr const char* _type_key = "script.Docs.StmtDoc";
  TVM_DECLARE_BASE_OBJECT_INFO(StmtDocNode, DocNode);
};

class StmtDoc : public Doc {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(StmtDoc, Doc, StmtDocNode);
};

// Assign Doc

class AssignDocNode : public StmtDocNode {
 public:
  ExprDoc target;
  ExprDoc value;

  static constexpr const char* _type_key = "script.Docs.AssignDoc";
  TVM_DECLARE_FINAL_OBJECT_INFO(AssignDocNode, StmtDocNode);
};

class AssignDoc : public StmtDoc {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(AssignDoc, StmtDoc, AssignDocNode);
};

// For Doc

class ForDocNode : public StmtDocNode {
 public:
  ExprDoc target;
  ExprDoc iter;
  Array<StmtDoc> body;

  static constexpr const char* _type_key = "script.Docs.ForDoc";
  TVM_DECLARE_FINAL_OBJECT_INFO(ForDocNode, StmtDocNode);
};

class ForDoc : public StmtDoc {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(ForDoc, StmtDoc, ForDocNode);
};

// Scope Doc

class ScopeDocNode : public StmtDocNode {
 public:
  ExprDoc scope;
  Array<StmtDoc> body;

  static constexpr const char* _type_key = "script.Docs.ScopeDoc";
  TVM_DECLARE_FINAL_OBJECT_INFO(ScopeDocNode, StmtDocNode);
};

class ScopeDoc : public StmtDoc {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(ScopeDoc, StmtDoc, ScopeDocNode);
};

// Type Param Doc

class TypeParamDocNode : public DocNode {
 public:
  static constexpr const char* _type_key = "script.Docs.TypeParamDoc";
  TVM_DECLARE_BASE_OBJECT_INFO(TypeParamDocNode, DocNode);
};

class TypeParamDoc : public Doc {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(TypeParamDoc, Doc, TypeParamDocNode);
};

// String Type Param Doc

class StringTypeParamDocNode : public TypeParamDocNode {
 public:
  LiteralStringDoc value;

  static constexpr const char* _type_key = "script.Docs.StringTypeParamDoc";
  TVM_DECLARE_FINAL_OBJECT_INFO(StringTypeParamDocNode, TypeParamDocNode);
};

class StringTypeParamDoc : public TypeParamDoc {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(StringTypeParamDoc, TypeParamDoc,
                                        StringTypeParamDocNode);
};

// Number Type Param Doc

class NumberTypeParamDocNode : public TypeParamDocNode {
 public:
  LiteralNumberDoc value;

  static constexpr const char* _type_key = "script.Docs.NumberTypeParamDoc";
  TVM_DECLARE_FINAL_OBJECT_INFO(NumberTypeParamDocNode, TypeParamDocNode);
};

class NumberTypeParamDoc : public TypeParamDoc {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(NumberTypeParamDoc, TypeParamDoc,
                                        NumberTypeParamDocNode);
};

// Tuple Type Param Doc

class TupleTypeParamDocNode : public TypeParamDocNode {
 public:
  Array<TypeParamDoc> Docs;

  static constexpr const char* _type_key = "script.Docs.TupleTypeParamDoc";
  TVM_DECLARE_FINAL_OBJECT_INFO(TupleTypeParamDocNode, TypeParamDocNode);
};

class TupleTypeParamDoc : public TypeParamDoc {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(TupleTypeParamDoc, TypeParamDoc,
                                        TupleTypeParamDocNode);
};

// Type Type Param Doc

class TypeTypeParamDocNode : public TypeParamDocNode {
 public:
  Doc type;  // TODO: Can we solve the circular def here?

  static constexpr const char* _type_key = "script.Docs.TypeTypeParamDoc";
  TVM_DECLARE_FINAL_OBJECT_INFO(TypeTypeParamDocNode, TypeParamDocNode);
};

class TypeTypeParamDoc : public TypeParamDoc {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(TypeTypeParamDoc, TypeParamDoc,
                                        TypeTypeParamDocNode);
};

// Primitive Type Doc

class PrimitiveTypeDocNode : public DocNode {
 public:
  static constexpr const char* _type_key = "script.Docs.PrimitiveTypeDoc";
  TVM_DECLARE_BASE_OBJECT_INFO(PrimitiveTypeDocNode, DocNode);
};

class PrimitiveTypeDoc : public Doc {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(PrimitiveTypeDoc, Doc,
                                        PrimitiveTypeDocNode);
};

// Number Type Doc

class NumberTypeDocNode : public PrimitiveTypeDocNode {
 public:
  DataType dtype;

  static constexpr const char* _type_key = "script.Docs.NumberTypeDoc";
  TVM_DECLARE_FINAL_OBJECT_INFO(NumberTypeDocNode, PrimitiveTypeDocNode);
};

class NumberTypeDoc : public PrimitiveTypeDoc {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(NumberTypeDoc, PrimitiveTypeDoc,
                                        NumberTypeDocNode);
};

// TVM Type Doc

class TVMTypeDocNode : public PrimitiveTypeDocNode {
 public:
  enum class Kind { Buffer, Handle };

  Kind kind;

  static constexpr const char* _type_key = "script.Docs.TVMTypeDoc";
  TVM_DECLARE_FINAL_OBJECT_INFO(NumberTypeDocNode, PrimitiveTypeDocNode);
};

class TVMTypeDoc : public PrimitiveTypeDoc {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(TVMTypeDoc, PrimitiveTypeDoc, TVMTypeDocNode);
};

// Type Doc

class TypeDocNode : public DocNode {
 public:
  PrimitiveTypeDoc base;
  Array<TypeParamDoc> params;

  static constexpr const char* _type_key = "script.Docs.TypeDoc";
  TVM_DECLARE_BASE_OBJECT_INFO(TypeDocNode, DocNode);
};

class TypeDoc : public Doc {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(TypeDoc, Doc, TypeDocNode);
};

// Function Arg Doc

class FunctionArgDocNode : public DocNode {
 public:
  IdentifierDoc ident;
  TypeDoc type;

  static constexpr const char* _type_key = "script.Docs.FunctionArgDoc";
  TVM_DECLARE_FINAL_OBJECT_INFO(FunctionArgDocNode, DocNode);
};

class FunctionArgDoc : public Doc {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(FunctionArgDoc, Doc, FunctionArgDocNode);
};

// Function Doc

class FunctionDocNode : public DocNode {
 public:
  IdentifierDoc name;
  Array<FunctionArgDoc> args;
  TypeDoc return_type;
  Array<StmtDoc> stmts;

  static constexpr const char* _type_key = "script.Docs.FunctionDoc";
  TVM_DECLARE_FINAL_OBJECT_INFO(FunctionDocNode, DocNode);
};

class FunctionDoc : public Doc {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(FunctionDoc, Doc, FunctionDocNode);
};

}  // namespace tvm

#endif  // TVM_TVMSCRIPT_UNIFIED_PRINTER_H_

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

#include "tvm/runtime/container/map.h"

namespace tvm {

// Code Element

class CodeElementNode : public Object {
 public:
  ObjectRef origin_ir_node;

  CodeElementNode(ObjectRef origin_ir_node) : origin_ir_node(origin_ir_node) {}
  virtual ~CodeElementNode() = default;

  static constexpr const char* _type_key = "script.elements.CodeElement";
  TVM_DECLARE_BASE_OBJECT_INFO(CodeElementNode, Object);
};

class CodeElement : public ObjectRef {
 public:
  explicit CodeElement(ObjectRef origin_ir_node) {
    data_ = make_object<CodeElementNode>(origin_ir_node);
  }

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(CodeElement, ObjectRef, CodeElementNode);
};

// Base Expr Element

class ExprElementNode : public CodeElementNode {
 public:
  static constexpr const char* _type_key = "script.elements.BaseExprElement";
  TVM_DECLARE_BASE_OBJECT_INFO(ExprElementNode, CodeElementNode);
};

class ExprElement : public CodeElement {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(ExprElement, CodeElement, ExprElementNode);
};

// Attributes Element

class AttributesElementNode : public CodeElementNode {
 public:
  Map<String, ExprElement> attrs;

  static constexpr const char* _type_key = "script.elements.AttributesElement";
  TVM_DECLARE_FINAL_OBJECT_INFO(AttributesElementNode, Object);
};

class AttributesElement : public CodeElement {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(AttributesElement, CodeElement, AttributesElementNode);
};

// Const Expr Element

class ConstElementNode : public ExprElementNode {
 public:
  static constexpr const char* _type_key = "script.elements.ConstElement";
  TVM_DECLARE_BASE_OBJECT_INFO(ConstElementNode, ExprElementNode);
};

class ConstElement : public ExprElement {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(ConstElement, ExprElement, ConstElementNode);
};

// String Element

class StringElementNode : public ConstElementNode {
 public:
  String value;

  static constexpr const char* _type_key = "script.elements.StirngElement";
  TVM_DECLARE_FINAL_OBJECT_INFO(StringElementNode, ConstElementNode);
};

class StringElement : public ConstElement {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(StringElement, ConstElement, StringElementNode);
};

// Number Element

class NumberElementNode : public ConstElementNode {
 public:
  static constexpr const char* _type_key = "script.elements.NumberElement";
  TVM_DECLARE_FINAL_OBJECT_INFO(NumberElementNode, ConstElementNode);

 private:
  ObjectRef value_;
};

class NumberElement : public ConstElement {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(NumberElement, ConstElement, NumberElementNode);
};

// TIR Builder Element

class TIRBuilderConstElementNode : public ConstElementNode {
 public:
  static constexpr const char* _type_key = "script.elements.TIRBuilderConstElement";
  TVM_DECLARE_FINAL_OBJECT_INFO(TIRBuilderConstElementNode, ConstElementNode);
};

class TIRBuilderConstElement : public ConstElement {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(TIRBuilderConstElement, ConstElement, TIRBuilderConstElementNode);
};

// Name Element

class IdentElementNode : public ExprElementNode {
 public:
  String name;

  static constexpr const char* _type_key = "script.elements.NameElement";
  TVM_DECLARE_FINAL_OBJECT_INFO(IdentElementNode, ExprElementNode);
};

class IdentElement : public ExprElement {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(IdentElement, ExprElement, IdentElementNode);
};

// Attr Element

class MemberAccessElementNode : public ExprElementNode {
 public:
  ExprElement value;
  IdentElement ident;

  static constexpr const char* _type_key = "script.elements.MemberAccessElement";
  TVM_DECLARE_FINAL_OBJECT_INFO(MemberAccessElementNode, ExprElementNode);
};

class MemberAccessElement : public ExprElement {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(MemberAccessElement, ExprElement, MemberAccessElementNode);
};

// Index Element

class IndexElementNode : public ExprElementNode {
 public:
  ExprElement value;
  ExprElement index;

  static constexpr const char* _type_key = "script.elements.IndexElement";
  TVM_DECLARE_FINAL_OBJECT_INFO(IndexElementNode, ExprElementNode);
};

class IndexElement : public ExprElement {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(IndexElement, ExprElement, IndexElementNode);
};

// BinOp Element

class BinOpElementNode : public ExprElementNode {
 public:
  enum class BinOpKind {
    Plus,
    Minus,
  };

  BinOpKind kind;
  ExprElement lhs;
  ExprElement rhs;

  static constexpr const char* _type_key = "script.elements.BinOpElement";
  TVM_DECLARE_FINAL_OBJECT_INFO(BinOpElementNode, ExprElementNode);
};

class BinOpElement : public ExprElement {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(BinOpElement, ExprElement, BinOpElementNode);
};

// Call Element

class CallElementNode : public ExprElementNode {
 public:
  ExprElement callee;
  Array<ExprElement> args;

  static constexpr const char* _type_key = "script.elements.CallElement";
  TVM_DECLARE_FINAL_OBJECT_INFO(CallElementNode, ExprElementNode);
};

class CallElement : public ExprElement {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(CallElement, ExprElement, CallElementNode);
};

// Base Stmt Element

class StmtElementNode : public CodeElementNode {
 public:
  static constexpr const char* _type_key = "script.elements.StmtElement";
  TVM_DECLARE_BASE_OBJECT_INFO(StmtElementNode, CodeElementNode);
};

class StmtElement : public CodeElement {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(StmtElement, CodeElement, StmtElementNode);
};

// Assign Element

class AssignElementNode : public StmtElementNode {
 public:
  ExprElement target;
  ExprElement value;

  static constexpr const char* _type_key = "script.elements.AssignElement";
  TVM_DECLARE_FINAL_OBJECT_INFO(AssignElementNode, StmtElementNode);
};

class AssignElement : public StmtElement {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(AssignElement, StmtElement, AssignElementNode);
};

// For Element

class ForElementNode : public StmtElementNode {
 public:
  ExprElement target;
  ExprElement iter;
  Array<StmtElement> body;

  static constexpr const char* _type_key = "script.elements.ForElement";
  TVM_DECLARE_FINAL_OBJECT_INFO(ForElementNode, StmtElementNode);
};

class ForElement : public StmtElement {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(ForElement, StmtElement, ForElementNode);
};

// Scope Element

class ScopeElementNode : public StmtElementNode {
 public:
  ExprElement scope;
  AttributesElement attrs;
  Array<StmtElement> body;

  static constexpr const char* _type_key = "script.elements.ScopeElement";
  TVM_DECLARE_FINAL_OBJECT_INFO(ScopeElementNode, StmtElementNode);
};

class ScopeElement : public StmtElement {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(ScopeElement, StmtElement, ScopeElementNode);
};

// Type Param Element

class TypeParamElementNode : public CodeElementNode {
 public:
  static constexpr const char* _type_key = "script.elements.TypeParamElement";
  TVM_DECLARE_BASE_OBJECT_INFO(TypeParamElementNode, CodeElementNode);
};

class TypeParamElement : public CodeElement {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(TypeParamElement, CodeElement, TypeParamElementNode);
};

// String Type Param Element

class StringTypeParamElementNode : public TypeParamElementNode {
 public:
  StringElement value;

  static constexpr const char* _type_key = "script.elements.StringTypeParamElement";
  TVM_DECLARE_FINAL_OBJECT_INFO(StringTypeParamElementNode, TypeParamElementNode);
};

class StringTypeParamElement : public TypeParamElement {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(StringTypeParamElement, TypeParamElement,
                                        StringTypeParamElementNode);
};

// Number Type Param Element

class NumberTypeParamElementNode : public TypeParamElementNode {
 public:
  NumberElement value;

  static constexpr const char* _type_key = "script.elements.NumberTypeParamElement";
  TVM_DECLARE_FINAL_OBJECT_INFO(NumberTypeParamElementNode, TypeParamElementNode);
};

class NumberTypeParamElement : public TypeParamElement {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(NumberTypeParamElement, TypeParamElement,
                                        NumberTypeParamElementNode);
};

// Tuple Type Param Element

class TupleTypeParamElementNode : public TypeParamElementNode {
 public:
  Array<TypeParamElement> elements;

  static constexpr const char* _type_key = "script.elements.TupleTypeParamElement";
  TVM_DECLARE_FINAL_OBJECT_INFO(TupleTypeParamElementNode, TypeParamElementNode);
};

class TupleTypeParamElement : public TypeParamElement {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(TupleTypeParamElement, TypeParamElement,
                                        TupleTypeParamElementNode);
};

// Type Type Param Element

class TypeTypeParamElementNode : public TypeParamElementNode {
 public:
  ConstElement type;  // TODO: Can we solve the circular def here?

  static constexpr const char* _type_key = "script.elements.TypeTypeParamElement";
  TVM_DECLARE_FINAL_OBJECT_INFO(TypeTypeParamElementNode, TypeParamElementNode);
};

class TypeTypeParamElement : public TypeParamElement {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(TypeTypeParamElement, TypeParamElement,
                                        TypeTypeParamElementNode);
};

// Primitive Type Element

class PrimitiveTypeElementNode : public CodeElementNode {
 public:
  static constexpr const char* _type_key = "script.elements.PrimitiveTypeElement";
  TVM_DECLARE_BASE_OBJECT_INFO(PrimitiveTypeElementNode, CodeElementNode);
};

class PrimitiveTypeElement : public CodeElement {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(PrimitiveTypeElement, CodeElement,
                                        PrimitiveTypeElementNode);
};

// Number Type Element

class NumberTypeElementNode : public PrimitiveTypeElementNode {
 public:
  DataType dtype;

  static constexpr const char* _type_key = "script.elements.NumberTypeElement";
  TVM_DECLARE_FINAL_OBJECT_INFO(NumberTypeElementNode, PrimitiveTypeElementNode);
};

class NumberTypeElement : public PrimitiveTypeElement {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(NumberTypeElement, PrimitiveTypeElement,
                                        NumberTypeElementNode);
};

// TVM Type Element

class TVMTypeElementNode : public PrimitiveTypeElementNode {
 public:
  enum class Kind { Buffer, Handle };

  Kind kind;

  static constexpr const char* _type_key = "script.elements.TVMTypeElement";
  TVM_DECLARE_FINAL_OBJECT_INFO(NumberTypeElementNode, PrimitiveTypeElementNode);
};

class TVMTypeElement : public PrimitiveTypeElement {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(TVMTypeElement, PrimitiveTypeElement, TVMTypeElementNode);
};

// Type Element

class TypeElementNode : public CodeElementNode {
 public:
  PrimitiveTypeElement base;
  Array<TypeParamElement> params;

  static constexpr const char* _type_key = "script.elements.TypeElement";
  TVM_DECLARE_BASE_OBJECT_INFO(TypeElementNode, CodeElementNode);
};

class TypeElement : public CodeElement {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(TypeElement, CodeElement, TypeElementNode);
};

// Function Arg Element

class FunctionArgElementNode : public CodeElementNode {
 public:
  IdentElement ident;
  TypeElement type;

  static constexpr const char* _type_key = "script.elements.FunctionArgElement";
  TVM_DECLARE_FINAL_OBJECT_INFO(FunctionArgElementNode, CodeElementNode);
};

class FunctionArgElement : public CodeElement {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(FunctionArgElement, CodeElement, FunctionArgElementNode);
};

// Function Element

class FunctionElementNode : public CodeElementNode {
 public:
  IdentElement name;
  Array<FunctionArgElement> args;
  TypeElement return_type;
  AttributesElement attrs;
  Array<StmtElement> stmts;

  static constexpr const char* _type_key = "script.elements.FunctionElement";
  TVM_DECLARE_FINAL_OBJECT_INFO(FunctionElementNode, CodeElementNode);
};

class FunctionElement : public CodeElement {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(FunctionElement, CodeElement, FunctionElementNode);
};

}  // namespace tvm

#endif  // TVM_TVMSCRIPT_UNIFIED_PRINTER_H_

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
#ifndef TVM_SCRIPT_PRINTER_DOC_H_
#define TVM_SCRIPT_PRINTER_DOC_H_

#include <tvm/ir/expr.h>
#include <tvm/node/node.h>

namespace tvm {
namespace script {
namespace printer {

class DocNode : public Object {
 public:
  mutable Optional<ObjectRef> source{NullOpt};

  void VisitAttrs(AttrVisitor* v) { v->Visit("source", &source); }

  static constexpr const char* _type_key = "script.Doc";
  TVM_DECLARE_BASE_OBJECT_INFO(DocNode, Object);

 public:
  virtual ~DocNode() = default;
};

class Doc : public ObjectRef {
 protected:
  Doc() = default;

 public:
  virtual ~Doc() = default;
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(Doc, ObjectRef, DocNode);
};

class AttrAccessDoc;
class IndexDoc;
class CallDoc;
class ExprDoc;

class ExprDocNode : public DocNode {
 public:
  void VisitAttrs(AttrVisitor* v) { DocNode::VisitAttrs(v); }

  static constexpr const char* _type_key = "script.ExprDoc";
  TVM_DECLARE_BASE_OBJECT_INFO(ExprDocNode, DocNode);

 public:
  AttrAccessDoc Attr(String attr) const;
  IndexDoc Index(Array<Doc> indices) const;
  CallDoc Call(Array<ExprDoc, void> args) const;
  CallDoc Call(Array<ExprDoc, void> args,        //
               Array<String, void> kwargs_keys,  //
               Array<ExprDoc, void> kwargs_values) const;
};

class ExprDoc : public Doc {
 protected:
  ExprDoc() = default;

 public:
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(ExprDoc, Doc, ExprDocNode);
};

class StmtDocNode : public DocNode {
 public:
  mutable Optional<String> inline_comment{NullOpt};

  void VisitAttrs(AttrVisitor* v) {
    DocNode::VisitAttrs(v);
    v->Visit("inline_comment", &inline_comment);
  }

  static constexpr const char* _type_key = "script.StmtDoc";
  TVM_DECLARE_BASE_OBJECT_INFO(StmtDocNode, DocNode);
};

class StmtDoc : public Doc {
 protected:
  StmtDoc() = default;

 public:
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(StmtDoc, Doc, StmtDocNode);
};

class StmtBlockDocNode : public DocNode {
 public:
  Array<StmtDoc> stmts;

  void VisitAttrs(AttrVisitor* v) {
    DocNode::VisitAttrs(v);
    v->Visit("stmts", &stmts);
  }

  static constexpr const char* _type_key = "script.StmtBlockDoc";
  TVM_DECLARE_FINAL_OBJECT_INFO(StmtBlockDocNode, DocNode);
};

class StmtBlockDoc : public Doc {
 public:
  explicit StmtBlockDoc(Array<StmtDoc> stmts);
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(StmtBlockDoc, Doc, StmtBlockDocNode);
};

}  // namespace printer
}  // namespace script
}  // namespace tvm

namespace tvm {
namespace script {
namespace printer {

class LiteralDocNode : public ExprDocNode {
 public:
  /*!
   * Union of: IntImm / FloatImm / String / None
   */
  ObjectRef value;

  void VisitAttrs(AttrVisitor* v) {
    ExprDocNode::VisitAttrs(v);
    v->Visit("value", &value);
  }

  static constexpr const char* _type_key = "script.LiteralDoc";
  TVM_DECLARE_FINAL_OBJECT_INFO(LiteralDocNode, ExprDocNode);
};

class LiteralDoc : public ExprDoc {
 protected:
  explicit LiteralDoc(ObjectRef value);

 public:
  static LiteralDoc None() { return LiteralDoc(ObjectRef(nullptr)); }
  static LiteralDoc Int(IntImm v) { return LiteralDoc(v); }
  static LiteralDoc Bool(Bool v) { return LiteralDoc(v); }
  static LiteralDoc Float(FloatImm v) { return LiteralDoc(v); }
  static LiteralDoc Str(String v) { return LiteralDoc(v); }

  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(LiteralDoc, ExprDoc, LiteralDocNode);
};

class IdDocNode : public ExprDocNode {
 public:
  String name;

  void VisitAttrs(AttrVisitor* v) {
    ExprDocNode::VisitAttrs(v);
    v->Visit("name", &name);
  }

  static constexpr const char* _type_key = "script.IdDoc";
  TVM_DECLARE_FINAL_OBJECT_INFO(IdDocNode, ExprDocNode);
};

class IdDoc : public ExprDoc {
 public:
  explicit IdDoc(String name);
  explicit IdDoc(std::nullptr_t) : ExprDoc(nullptr) {}
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(IdDoc, ExprDoc, IdDocNode);
};

class SliceDocNode : public DocNode {
 public:
  Optional<ExprDoc> start;
  Optional<ExprDoc> stop;

  void VisitAttrs(AttrVisitor* v) {
    DocNode::VisitAttrs(v);
    v->Visit("start", &start);
    v->Visit("stop", &stop);
  }

  static constexpr const char* _type_key = "script.SliceDoc";
  TVM_DECLARE_FINAL_OBJECT_INFO(SliceDocNode, DocNode);
};

class SliceDoc : public Doc {
 public:
  explicit SliceDoc(Optional<ExprDoc> start, Optional<ExprDoc> stop);
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(SliceDoc, Doc, SliceDocNode);
};

class AttrAccessDocNode : public ExprDocNode {
 public:
  ExprDoc value{nullptr};
  String attr;

  void VisitAttrs(AttrVisitor* v) {
    ExprDocNode::VisitAttrs(v);
    v->Visit("value", &value);
    v->Visit("attr", &attr);
  }

  static constexpr const char* _type_key = "script.AttrAccessDoc";
  TVM_DECLARE_FINAL_OBJECT_INFO(AttrAccessDocNode, ExprDocNode);
};

class AttrAccessDoc : public ExprDoc {
 public:
  explicit AttrAccessDoc(ExprDoc value, String attr);
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(AttrAccessDoc, ExprDoc, AttrAccessDocNode);
};

class IndexDocNode : public ExprDocNode {
 public:
  ExprDoc value{nullptr};
  Array<Doc> indices;  // Each element is union of: Slice / ExprDoc

  void VisitAttrs(AttrVisitor* v) {
    ExprDocNode::VisitAttrs(v);
    v->Visit("value", &value);
    v->Visit("indices", &indices);
  }

  static constexpr const char* _type_key = "script.IndexDoc";
  TVM_DECLARE_FINAL_OBJECT_INFO(IndexDocNode, ExprDocNode);
};

class IndexDoc : public ExprDoc {
 public:
  explicit IndexDoc(ExprDoc value, Array<Doc> indices);
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(IndexDoc, ExprDoc, IndexDocNode);
};

class CallDocNode : public ExprDocNode {
 public:
  ExprDoc callee{nullptr};
  Array<ExprDoc> args;
  Array<String> kwargs_keys;
  Array<ExprDoc> kwargs_values;

  void VisitAttrs(AttrVisitor* v) {
    ExprDocNode::VisitAttrs(v);
    v->Visit("callee", &callee);
    v->Visit("args", &args);
    v->Visit("kwargs_keys", &kwargs_keys);
    v->Visit("kwargs_values", &kwargs_values);
  }

  static constexpr const char* _type_key = "script.CallDoc";
  TVM_DECLARE_FINAL_OBJECT_INFO(CallDocNode, ExprDocNode);
};

class CallDoc : public ExprDoc {
 public:
  CallDoc(ExprDoc callee, Array<ExprDoc> args, Array<String> kwargs_keys,
          Array<ExprDoc> kwargs_values);
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(CallDoc, ExprDoc, CallDocNode);
};

class OperationDocNode : public ExprDocNode {
 public:
  enum class Kind : int32_t {
    kUndefined = 0,
    kAdd = 1,
    kSub = 2,
    kMul = 3,
    kFloorDiv = 4,
    kFloorMod = 5,
  };

  Kind kind;
  Array<ExprDoc> operands;

  void VisitAttrs(AttrVisitor* v) {
    ExprDocNode::VisitAttrs(v);
    v->Visit("kind", &kind);
    v->Visit("operands", &operands);
  }

  static constexpr const char* _type_key = "script.OperationDoc";
  TVM_DECLARE_FINAL_OBJECT_INFO(OperationDocNode, ExprDocNode);
};

class OperationDoc : public ExprDoc {
 public:
  explicit OperationDoc(OperationDocNode::Kind kind, Array<ExprDoc> operands);
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(OperationDoc, ExprDoc, OperationDocNode);
};

class TupleDocNode : public ExprDocNode {
 public:
  Array<ExprDoc> elements;

  void VisitAttrs(AttrVisitor* v) {
    ExprDocNode::VisitAttrs(v);
    v->Visit("elements", &elements);
  }

  static constexpr const char* _type_key = "script.TupleDoc";
  TVM_DECLARE_FINAL_OBJECT_INFO(TupleDocNode, ExprDocNode);
};

class TupleDoc : public ExprDoc {
 public:
  explicit TupleDoc(Array<ExprDoc> elements);
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(TupleDoc, ExprDoc, TupleDocNode);
};

class ListDocNode : public ExprDocNode {
 public:
  Array<ExprDoc> elements;

  void VisitAttrs(AttrVisitor* v) {
    ExprDocNode::VisitAttrs(v);
    v->Visit("elements", &elements);
  }

  static constexpr const char* _type_key = "script.ListDoc";
  TVM_DECLARE_FINAL_OBJECT_INFO(ListDocNode, ExprDocNode);
};

class ListDoc : public ExprDoc {
 public:
  explicit ListDoc(Array<ExprDoc> elements);
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(ListDoc, ExprDoc, ListDocNode);
};

class DictDocNode : public ExprDocNode {
 public:
  Array<ExprDoc> keys;
  Array<ExprDoc> values;

  void VisitAttrs(AttrVisitor* v) {
    ExprDocNode::VisitAttrs(v);
    v->Visit("keys", &keys);
    v->Visit("values", &values);
  }

  static constexpr const char* _type_key = "script.DictDoc";
  TVM_DECLARE_FINAL_OBJECT_INFO(DictDocNode, ExprDocNode);
};

class DictDoc : public ExprDoc {
 public:
  explicit DictDoc(Array<ExprDoc> keys, Array<ExprDoc> values);
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(DictDoc, ExprDoc, DictDocNode);
};

}  // namespace printer
}  // namespace script
}  // namespace tvm

namespace tvm {
namespace script {
namespace printer {

class AssignDocNode : public StmtDocNode {
 public:
  ExprDoc lhs{nullptr};
  Optional<ExprDoc> rhs;  // If null, this doc represents declaration, e.g. `A: T.Buffer[(1,2)]`
  Optional<ExprDoc> annotation;

  void VisitAttrs(AttrVisitor* v) {
    StmtDocNode::VisitAttrs(v);
    v->Visit("lhs", &lhs);
    v->Visit("rhs", &rhs);
    v->Visit("annotation", &annotation);
  }

  static constexpr const char* _type_key = "script.AssignDoc";
  TVM_DECLARE_FINAL_OBJECT_INFO(AssignDocNode, StmtDocNode);
};

class AssignDoc : public StmtDoc {
 public:
  explicit AssignDoc(ExprDoc lhs, Optional<ExprDoc> rhs, Optional<ExprDoc> annotation);
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(AssignDoc, StmtDoc, AssignDocNode);
};

class ForDocNode : public StmtDocNode {
 public:
  ExprDoc lhs{nullptr};
  ExprDoc rhs{nullptr};
  Array<StmtDoc> body;

  void VisitAttrs(AttrVisitor* v) {
    StmtDocNode::VisitAttrs(v);
    v->Visit("lhs", &lhs);
    v->Visit("rhs", &rhs);
    v->Visit("body", &body);
  }

  static constexpr const char* _type_key = "script.ForDoc";
  TVM_DECLARE_FINAL_OBJECT_INFO(ForDocNode, StmtDocNode);
};

class ForDoc : public StmtDoc {
 public:
  explicit ForDoc(ExprDoc lhs, ExprDoc rhs, Array<StmtDoc> body);
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(ForDoc, StmtDoc, ForDocNode);
};

class ScopeDocNode : public StmtDocNode {
 public:
  Optional<ExprDoc> lhs{NullOpt};
  ExprDoc rhs{nullptr};
  Array<StmtDoc> body;

  void VisitAttrs(AttrVisitor* v) {
    StmtDocNode::VisitAttrs(v);
    v->Visit("lhs", &lhs);
    v->Visit("rhs", &rhs);
    v->Visit("body", &body);
  }

  static constexpr const char* _type_key = "script.ScopeDoc";
  TVM_DECLARE_FINAL_OBJECT_INFO(ScopeDocNode, StmtDocNode);
};

class ScopeDoc : public StmtDoc {
 public:
  explicit ScopeDoc(ExprDoc lhs, ExprDoc rhs, Array<StmtDoc> body);
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(ScopeDoc, StmtDoc, ScopeDocNode);
};

class ExprStmtDocNode : public StmtDocNode {
 public:
  ExprDoc expr{nullptr};

  void VisitAttrs(AttrVisitor* v) {
    StmtDocNode::VisitAttrs(v);
    v->Visit("expr", &expr);
  }

  static constexpr const char* _type_key = "script.ExprStmtDoc";
  TVM_DECLARE_FINAL_OBJECT_INFO(ExprStmtDocNode, StmtDocNode);
};

class ExprStmtDoc : public StmtDoc {
 public:
  explicit ExprStmtDoc(ExprDoc expr);
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(ExprStmtDoc, StmtDoc, ExprStmtDocNode);
};

}  // namespace printer
}  // namespace script
}  // namespace tvm

namespace tvm {
namespace script {
namespace printer {

class FunctionDocNode : public StmtDocNode {
 public:
  IdDoc name{nullptr};
  Array<AssignDoc> args;
  Array<ExprDoc> decorators;
  ExprDoc return_type{nullptr};
  Array<StmtDoc> body;

  void VisitAttrs(AttrVisitor* v) {
    DocNode::VisitAttrs(v);
    v->Visit("name", &name);
    v->Visit("args", &args);
    v->Visit("decorators", &decorators);
    v->Visit("return_type", &return_type);
    v->Visit("body", &body);
  }

  static constexpr const char* _type_key = "script.FunctionDoc";
  TVM_DECLARE_FINAL_OBJECT_INFO(FunctionDocNode, StmtDocNode);
};

class FunctionDoc : public StmtDoc {
 public:
  explicit FunctionDoc(IdDoc name, Array<AssignDoc> args, Array<ExprDoc> decorators,
                       ExprDoc return_type, Array<StmtDoc> body);
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(FunctionDoc, StmtDoc, FunctionDocNode);
};

}  // namespace printer
}  // namespace script
}  // namespace tvm

namespace tvm {
namespace script {
namespace printer {

inline AttrAccessDoc ExprDocNode::Attr(String attr) const {
  return AttrAccessDoc(GetRef<ExprDoc>(this), attr);
}

inline IndexDoc ExprDocNode::Index(Array<Doc> indices) const {
  return IndexDoc(GetRef<ExprDoc>(this), indices);
}

inline CallDoc ExprDocNode::Call(Array<ExprDoc, void> args) const {
  return CallDoc(GetRef<ExprDoc>(this), args, {}, {});
}

inline CallDoc ExprDocNode::Call(Array<ExprDoc, void> args, Array<String, void> kwargs_keys,
                                 Array<ExprDoc, void> kwargs_values) const {
  return CallDoc(GetRef<ExprDoc>(this), args, kwargs_keys, kwargs_values);
}

}  // namespace printer
}  // namespace script

}  // namespace tvm

#endif

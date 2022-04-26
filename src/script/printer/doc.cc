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
#include "./doc.h"

#include <tvm/runtime/registry.h>

namespace tvm {
namespace script {
namespace printer {

TVM_REGISTER_NODE_TYPE(DocNode);
TVM_REGISTER_NODE_TYPE(ExprDocNode);
TVM_REGISTER_NODE_TYPE(StmtDocNode);

StmtBlockDoc::StmtBlockDoc(Array<StmtDoc> stmts) {
  ObjectPtr<StmtBlockDocNode> n = make_object<StmtBlockDocNode>();
  n->stmts = stmts;
  this->data_ = n;
}
TVM_REGISTER_NODE_TYPE(StmtBlockDocNode);
TVM_REGISTER_GLOBAL("script.StmtBlockDoc").set_body_typed([](Array<StmtDoc> stmts) {
  return StmtBlockDoc(stmts);
});

}  // namespace printer
}  // namespace script
}  // namespace tvm

namespace tvm {
namespace script {
namespace printer {

LiteralDoc::LiteralDoc(ObjectRef value) {
  ObjectPtr<LiteralDocNode> n = make_object<LiteralDocNode>();
  n->value = value;
  this->data_ = n;
}
TVM_REGISTER_NODE_TYPE(LiteralDocNode);
TVM_REGISTER_GLOBAL("script.LiteralDocInt").set_body_typed([](IntImm v) {
  return LiteralDoc::Int(v);
});
TVM_REGISTER_GLOBAL("script.LiteralDocFloat").set_body_typed([](FloatImm v) {
  return LiteralDoc::Float(v);
});
TVM_REGISTER_GLOBAL("script.LiteralDocBool").set_body_typed([](Bool v) {
  return LiteralDoc::Bool(v);
});
TVM_REGISTER_GLOBAL("script.LiteralDocStr").set_body_typed([](String v) {
  return LiteralDoc::Str(v);
});
TVM_REGISTER_GLOBAL("script.LiteralDocNone").set_body_typed([]() { return LiteralDoc::None(); });

IdDoc::IdDoc(String name) {
  ObjectPtr<IdDocNode> n = make_object<IdDocNode>();
  n->name = name;
  this->data_ = n;
}
TVM_REGISTER_NODE_TYPE(IdDocNode);
TVM_REGISTER_GLOBAL("script.IdDoc").set_body_typed([](String name) { return IdDoc(name); });

SliceDoc::SliceDoc(Optional<ExprDoc> start, Optional<ExprDoc> stop) {
  ObjectPtr<SliceDocNode> n = make_object<SliceDocNode>();
  n->start = start;
  n->stop = stop;
  this->data_ = n;
}
TVM_REGISTER_NODE_TYPE(SliceDocNode);
TVM_REGISTER_GLOBAL("script.SliceDoc")
    .set_body_typed([](Optional<ExprDoc> start, Optional<ExprDoc> stop) {
      return SliceDoc(start, stop);
    });

AttrAccessDoc::AttrAccessDoc(ExprDoc value, String attr) {
  ObjectPtr<AttrAccessDocNode> n = make_object<AttrAccessDocNode>();
  n->value = value;
  n->attr = attr;
  this->data_ = n;
}
TVM_REGISTER_NODE_TYPE(AttrAccessDocNode);
TVM_REGISTER_GLOBAL("script.AttrAccessDoc").set_body_typed([](ExprDoc value, String attr) {
  return AttrAccessDoc(value, attr);
});

IndexDoc::IndexDoc(ExprDoc value, Array<Doc> indices) {
  ObjectPtr<IndexDocNode> n = make_object<IndexDocNode>();
  n->value = value;
  n->indices = indices;
  this->data_ = n;
}
TVM_REGISTER_NODE_TYPE(IndexDocNode);
TVM_REGISTER_GLOBAL("script.IndexDoc").set_body_typed([](ExprDoc value, Array<Doc> indices) {
  return IndexDoc(value, indices);
});

CallDoc::CallDoc(ExprDoc callee, Array<ExprDoc> args, Array<String> kwargs_keys,
                 Array<ExprDoc> kwargs_values) {
  ObjectPtr<CallDocNode> n = make_object<CallDocNode>();
  n->callee = callee;
  n->args = args;
  n->kwargs_keys = kwargs_keys;
  n->kwargs_values = kwargs_values;
  this->data_ = n;
}
TVM_REGISTER_NODE_TYPE(CallDocNode);
TVM_REGISTER_GLOBAL("script.CallDoc")
    .set_body_typed([](ExprDoc callee, Array<ExprDoc> args, Array<String> kwargs_keys,
                       Array<ExprDoc> kwargs_values) {
      return CallDoc(callee, args, kwargs_keys, kwargs_values);
    });

OperationDoc::OperationDoc(OperationDocNode::Kind kind, Array<ExprDoc> operands) {
  ObjectPtr<OperationDocNode> n = make_object<OperationDocNode>();
  n->kind = kind;
  n->operands = operands;
  this->data_ = n;
}
TVM_REGISTER_NODE_TYPE(OperationDocNode);
TVM_REGISTER_GLOBAL("script.OperationDoc").set_body_typed([](int kind, Array<ExprDoc> operands) {
  return OperationDoc(static_cast<OperationDocNode::Kind>(kind), operands);
});

TupleDoc::TupleDoc(Array<ExprDoc> elements) {
  ObjectPtr<TupleDocNode> n = make_object<TupleDocNode>();
  n->elements = elements;
  this->data_ = n;
}
TVM_REGISTER_NODE_TYPE(TupleDocNode);
TVM_REGISTER_GLOBAL("script.TupleDoc").set_body_typed([](Array<ExprDoc> elements) {
  return TupleDoc(elements);
});

ListDoc::ListDoc(Array<ExprDoc> elements) {
  ObjectPtr<ListDocNode> n = make_object<ListDocNode>();
  n->elements = elements;
  this->data_ = n;
}
TVM_REGISTER_NODE_TYPE(ListDocNode);
TVM_REGISTER_GLOBAL("script.ListDoc").set_body_typed([](Array<ExprDoc> elements) {
  return ListDoc(elements);
});

DictDoc::DictDoc(Array<ExprDoc> keys, Array<ExprDoc> values) {
  ObjectPtr<DictDocNode> n = make_object<DictDocNode>();
  n->keys = keys;
  n->values = values;
  this->data_ = n;
}
TVM_REGISTER_NODE_TYPE(DictDocNode);
TVM_REGISTER_GLOBAL("script.DictDoc")
    .set_body_typed([](Array<ExprDoc> keys, Array<ExprDoc> values) {
      return DictDoc(keys, values);
    });

}  // namespace printer
}  // namespace script
}  // namespace tvm

namespace tvm {
namespace script {
namespace printer {

AssignDoc::AssignDoc(ExprDoc lhs, Optional<ExprDoc> rhs, Optional<ExprDoc> annotation) {
  ObjectPtr<AssignDocNode> n = make_object<AssignDocNode>();
  n->lhs = lhs;
  n->rhs = rhs;
  n->annotation = annotation;
  this->data_ = n;
}
TVM_REGISTER_NODE_TYPE(AssignDocNode);
TVM_REGISTER_GLOBAL("script.AssignDoc")
    .set_body_typed([](ExprDoc lhs, Optional<ExprDoc> rhs, Optional<ExprDoc> annotation) {
      return AssignDoc(lhs, rhs, annotation);
    });

ForDoc::ForDoc(ExprDoc lhs, ExprDoc rhs, Array<StmtDoc> body) {
  ObjectPtr<ForDocNode> n = make_object<ForDocNode>();
  n->lhs = lhs;
  n->rhs = rhs;
  n->body = body;
  this->data_ = n;
}
TVM_REGISTER_NODE_TYPE(ForDocNode);
TVM_REGISTER_GLOBAL("script.ForDoc")
    .set_body_typed([](ExprDoc lhs, ExprDoc rhs, Array<StmtDoc> body) {
      return ForDoc(lhs, rhs, body);
    });

ScopeDoc::ScopeDoc(ExprDoc lhs, ExprDoc rhs, Array<StmtDoc> body) {
  ObjectPtr<ScopeDocNode> n = make_object<ScopeDocNode>();
  n->lhs = lhs;
  n->rhs = rhs;
  n->body = body;
  this->data_ = n;
}
TVM_REGISTER_NODE_TYPE(ScopeDocNode);
TVM_REGISTER_GLOBAL("script.ScopeDoc")
    .set_body_typed([](ExprDoc lhs, ExprDoc rhs, Array<StmtDoc> body) {
      return ScopeDoc(lhs, rhs, body);
    });

ExprStmtDoc::ExprStmtDoc(ExprDoc expr) {
  ObjectPtr<ExprStmtDocNode> n = make_object<ExprStmtDocNode>();
  n->expr = expr;
  this->data_ = n;
}
TVM_REGISTER_NODE_TYPE(ExprStmtDocNode);
TVM_REGISTER_GLOBAL("script.ExprStmtDoc").set_body_typed([](ExprDoc expr) {
  return ExprStmtDoc(expr);
});

}  // namespace printer
}  // namespace script
}  // namespace tvm

namespace tvm {
namespace script {
namespace printer {

FunctionDoc::FunctionDoc(IdDoc name, Array<AssignDoc> args, Array<ExprDoc> decorators,
                         ExprDoc return_type, Array<StmtDoc> body) {
  ObjectPtr<FunctionDocNode> n = make_object<FunctionDocNode>();
  n->name = name;
  n->args = args;
  n->decorators = decorators;
  n->return_type = return_type;
  n->body = body;
  this->data_ = n;
}
TVM_REGISTER_NODE_TYPE(FunctionDocNode);
TVM_REGISTER_GLOBAL("script.FunctionDoc")
    .set_body_typed([](IdDoc name, Array<AssignDoc> args, Array<ExprDoc> decorators,
                       ExprDoc return_type, Array<StmtDoc> body) {
      return FunctionDoc(name, args, decorators, return_type, body);
    });

}  // namespace printer
}  // namespace script
}  // namespace tvm

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
#include <tvm/node/functor.h>
#include <tvm/runtime/data_type.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/var.h>

#include "../util.h"
#include "./utils.h"

namespace tvm {
namespace script {
namespace printer {

ExprDoc PrintVar(tir::Var v, IRDocsifier p) {
  Optional<ExprDoc> var_doc = p->sym->GetObjectDoc(v);
  if (var_doc == nullptr) {
    auto frame = p->GetFrame<VarDefFrame>().value();
    IdDoc free_var_doc = frame->DefByName(v, p->sym->GetUniqueName(v->name_hint));
    return free_var_doc;
  } else {
    return var_doc.value();
  }
}

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::Var>(PrintVar);
TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::SizeVar>(PrintVar);

ExprDoc PrintIterVar(tir::IterVar v, IRDocsifier p) {
  ExprDoc dom_arg = LiteralDoc::None();
  if (v->dom.defined()) {
    Range dom = v->dom;
    dom_arg = IdDoc("slice")->Call({p->AsExprDoc(dom->min), p->AsExprDoc(dom->min + dom->extent)});
  }

  return TIR(p)
      ->Attr("iter_var")
      ->Call({p->AsExprDoc(v->var), dom_arg, LiteralDoc::Str(IterVarType2String(v->iter_type)),
              LiteralDoc::Str(v->thread_tag)});
}

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::IterVar>(PrintIterVar);
}  // namespace printer
}  // namespace script
}  // namespace tvm

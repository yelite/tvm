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
  ICHECK_NOTNULL(var_doc);
  return var_doc.value();
}

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::Var>(PrintVar);
TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::SizeVar>(PrintVar);

ExprDoc PrintIterVar(tir::IterVar v, IRDocsifier p) {
  LOG(FATAL) << "Cannot print iter var directly. Please use the helper functions in tir.h for "
                "specific usage of IterVar.";
  throw;
}

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::IterVar>(PrintIterVar);
}  // namespace printer
}  // namespace script
}  // namespace tvm

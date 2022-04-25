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
#include <tvm/ir/type.h>

#include "./utils.h"

namespace tvm {
namespace script {
namespace printer {

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<PrimType>("tir", [](PrimType ty, IRDocsifier p) -> Doc {
      using runtime::DLDataType2String;
      return TIR(p)->Attr(DLDataType2String(ty->dtype));
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<PointerType>("tir", [](PointerType ty, IRDocsifier p) -> Doc {
      ExprDoc element_type = p->AsDoc<ExprDoc>(ty->element_type);
      if (ty->storage_scope.empty()) {
        return TIR(p)->Attr("Ptr")->Call({element_type});
      } else {
        return TIR(p)->Attr("Ptr")->Call({element_type, LiteralDoc::Str(ty->storage_scope)});
      }
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<TupleType>("tir", [](TupleType ty, IRDocsifier p) -> Doc {
      if (ty->fields.empty()) {
        return LiteralDoc::None();
      }
      Array<ExprDoc> types;
      types.reserve(ty->fields.size());
      for (auto field : ty->fields) {
        types.push_back(p->AsDoc<ExprDoc>(field));
      }
      return TIR(p)->Attr("Tuple")->Call(types);
    });

}  // namespace printer
}  // namespace script
}  // namespace tvm

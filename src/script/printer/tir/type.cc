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

#include "../util.h"
#include "./utils.h"

namespace tvm {
namespace script {
namespace printer {

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<PrimType>("tir", [](TracedObject<PrimType> ty, IRDocsifier p) -> Doc {
      auto dtype = ty.GetAttr(&PrimTypeNode::dtype);
      String ty_str = runtime::DLDataType2String(dtype.Get());
      return TIR(p)->Attr(MakeTraced(ty_str, ty.GetPath()));
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<PointerType>("tir", [](TracedObject<PointerType> ty, IRDocsifier p) -> Doc {
      auto element_type = ty.GetAttr(&PointerTypeNode::element_type);
      auto storage_scope = ty.GetAttr(&PointerTypeNode::storage_scope);

      ExprDoc element_type_doc = p->AsDoc<ExprDoc>(element_type);
      if (storage_scope.Get().empty()) {
        return TIR(p)->Attr("Ptr")->Call({element_type_doc});
      } else {
        return TIR(p)->Attr("Ptr")->Call({element_type_doc, LiteralDoc::Str(storage_scope)});
      }
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<TupleType>("tir", [](TracedObject<TupleType> ty, IRDocsifier p) -> Doc {
      auto fields = ty.GetAttr(&TupleTypeNode::fields);

      if (fields.empty()) {
        return LiteralDoc::None(fields.GetPath());
      }
      return TIR(p)->Attr("Tuple")->Call(AsExprDocArray(fields, p));
    });

}  // namespace printer
}  // namespace script
}  // namespace tvm

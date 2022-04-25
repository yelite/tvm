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
#include <tvm/runtime/device_api.h>
#include <tvm/tir/stmt_functor.h>

#include <algorithm>

#include "./utils.h"

namespace tvm {
namespace script {
namespace printer {

ExprDoc BufferPrintInfo::AsCall(const ExprDoc& prefix,
                                std::function<ExprDoc(const PrimExpr&)> converter) const {
  Array<ExprDoc> args;
  Array<String> kwargs_keys;
  Array<ExprDoc> kwargs_values;
  {
    Array<ExprDoc> results;
    results.reserve(shape.size());
    for (PrimExpr e : shape) {
      results.push_back(converter(e));
    }
    kwargs_keys.push_back("shape");
    kwargs_values.push_back(TupleDoc(results));
  }
  if (dtype.defined()) {
    args.push_back(dtype.value());
  }
  if (data.defined()) {
    kwargs_keys.push_back("data");
    kwargs_values.push_back(converter(data.value()));
  }
  if (strides.defined()) {
    Array<ExprDoc> results;
    results.reserve(strides.value().size());
    for (PrimExpr stride : strides.value()) {
      results.push_back(converter(stride));
    }
    kwargs_keys.push_back("strides");
    kwargs_values.push_back(TupleDoc(results));
  }
  if (elem_offset.defined()) {
    kwargs_keys.push_back("elem_offset");
    kwargs_values.push_back(converter(elem_offset.value()));
  }
  if (scope.defined()) {
    kwargs_keys.push_back("scope");
    kwargs_values.push_back(scope.value());
  }
  if (align.defined()) {
    kwargs_keys.push_back("align");
    kwargs_values.push_back(align.value());
  }
  if (offset_factor.defined()) {
    kwargs_keys.push_back("offset_factor");
    kwargs_values.push_back(offset_factor.value());
  }
  if (buffer_type.defined()) {
    kwargs_keys.push_back("buffer_type");
    kwargs_values.push_back(buffer_type.value());
  }
  return prefix->Call(args, kwargs_keys, kwargs_values);
}

std::vector<BufferPrintInfo> GetBufferPrintInfo(
    const std::vector<tir::Buffer>& buffers,  //
    std::function<bool(const tir::VarNode*)> f_var_defined,
    std::unordered_set<const tir::VarNode*>* var_explicit_def,
    std::unordered_map<const tir::VarNode*, const tir::BufferNode*>* var_associated_def) {
  using namespace tvm::tir;
  auto check_associated_def = [&](const PrimExpr& e, const Buffer& buffer) -> void {
    if (const auto* v = e.as<VarNode>()) {
      if (!f_var_defined(v) && !var_associated_def->count(v)) {
        var_associated_def->insert({v, buffer.get()});
      }
    }
  };
  auto check_explicit_def = [&](const PrimExpr& e) -> void {
    PostOrderVisit(e, [&](const ObjectRef& n) -> void {
      if (const auto* v = n.as<VarNode>()) {
        if (!f_var_defined(v) && !var_associated_def->count(v)) {
          var_explicit_def->insert(v);
        }
      }
    });
  };
  auto is_associated_with = [&](const PrimExpr& e, const Buffer& buffer) -> bool {
    if (const auto* v = e.as<VarNode>()) {
      if (var_associated_def->count(v)) {
        return var_associated_def->at(v) == buffer.get();
      }
    }
    return false;
  };
  for (const Buffer& buffer : buffers) {
    check_associated_def(buffer->data, buffer);
    check_associated_def(buffer->elem_offset, buffer);
  }
  for (const Buffer& buffer : buffers) {
    std::for_each(buffer->shape.begin(), buffer->shape.end(), check_explicit_def);
    std::for_each(buffer->strides.begin(), buffer->strides.end(), check_explicit_def);
    check_explicit_def(buffer->data);
    check_explicit_def(buffer->elem_offset);
  }
  std::vector<BufferPrintInfo> results;
  for (const Buffer& buffer : buffers) {
    BufferPrintInfo info;
    String scope = buffer.scope();
    info.buffer = buffer;
    info.shape = buffer->shape;
    if (buffer->dtype == DataType::Float(32)) {
      info.dtype = NullOpt;
    } else {
      info.dtype = DType2Literal(buffer->dtype);
    }
    if (is_associated_with(buffer->data, buffer)) {
      info.data = NullOpt;
    } else {
      info.data = buffer->data;
    }
    if (buffer->strides.defined() && !buffer->strides.empty()) {
      info.strides = buffer->strides;
    } else {
      info.strides = NullOpt;
    }
    if (buffer->elem_offset.defined()) {
      if (const auto* v = buffer->elem_offset.as<VarNode>()) {
        if (is_associated_with(buffer->elem_offset, buffer)) {
          info.elem_offset = NullOpt;
        } else {
          info.elem_offset = GetRef<Var>(v);
        }
      } else if (const auto* i = buffer->elem_offset.as<IntImmNode>()) {
        if (i->value == 0 && i->dtype == DataType::Int(32)) {
          info.elem_offset = NullOpt;
        } else {
          info.elem_offset = GetRef<IntImm>(i);
        }
      } else {
        info.elem_offset = buffer->elem_offset;
      }
    } else {
      info.elem_offset = NullOpt;
    }
    if (scope != "global") {
      info.scope = LiteralDoc::Str(scope);
    } else {
      info.scope = NullOpt;
    }
    if (buffer->data_alignment != runtime::kAllocAlignment) {
      info.align = LiteralDoc::Int(Integer(buffer->data_alignment));
    } else {
      info.align = NullOpt;
    }
    if (buffer->offset_factor != 1) {
      info.offset_factor = LiteralDoc::Int(Integer(buffer->offset_factor));
    } else {
      info.offset_factor = NullOpt;
    }
    if (buffer->buffer_type != BufferType::kDefault) {
      info.buffer_type = LiteralDoc::Str("auto");
    } else {
      info.buffer_type = NullOpt;
    }
    results.push_back(info);
  }
  return results;
}

}  // namespace printer
}  // namespace script
}  // namespace tvm

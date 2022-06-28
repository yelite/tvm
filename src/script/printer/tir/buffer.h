#ifndef TVM_SCRIPT_PRINTER_TIR_BUFFER_H_
#define TVM_SCRIPT_PRINTER_TIR_BUFFER_H_

#include <tvm/tir/buffer.h>

#include <unordered_map>
#include <vector>

#include "../ir_docsifier.h"
#include "../traced_object.h"
#include "utils.h"

namespace tvm {
namespace script {
namespace printer {

class AssociatedVariables {
 public:
  void Disassociate(const tir::VarNode* var) { var2buffer_.erase(var); }

  void AssociateIfNotAlready(const tir::VarNode* var, const tir::Buffer& buffer) {
    var2buffer_.insert({var, buffer});
  }

  bool IsAssociated(const tir::VarNode* var) const { return var2buffer_.count(var) != 0; }

  bool IsAssociatedWith(const PrimExpr& e, const tir::Buffer& buffer) const {
    if (const auto* v = e.as<tir::VarNode>()) {
      auto it = var2buffer_.find(v);
      return it != var2buffer_.end() && it->second == buffer;
    }
    return false;
  }

  void DefineVariables(const Frame& frame) const {
    for (const auto& kv : var2buffer_) {
      const tir::VarNode* var = kv.first;
      const tir::Buffer& buffer = kv.second;

      if (buffer->data.get() == var) {
        DefineBufferDataVariable(buffer, frame);
      } else if (buffer->elem_offset.get() == var) {
        DefineBufferElemOffsetVariable(buffer, frame);
      } else {
        ICHECK(false) << "Unexpected association. Buffer: " << buffer
                      << "; Var: " << GetRef<tir::Var>(var);
      }
    }
  }

 private:
  std::unordered_map<const tir::VarNode*, tir::Buffer> var2buffer_;
};

struct BufferPrintInfo {
  TracedObject<tir::Buffer> buffer;
  TracedArray<PrimExpr> shape;
  Optional<ExprDoc> dtype;
  TracedOptional<tir::Var> data;
  TracedOptional<Array<PrimExpr>> strides;
  TracedOptional<PrimExpr> elem_offset;
  Optional<ExprDoc> scope;
  Optional<ExprDoc> align;
  Optional<ExprDoc> offset_factor;
  Optional<ExprDoc> buffer_type;

  ExprDoc AsCall(const ExprDoc& prefix,
                 std::function<ExprDoc(const TracedObject<PrimExpr>&)> converter) const;
  ExprDoc AsCall(const ExprDoc& prefix, const Array<ExprDoc>& extra_args,
                 std::function<ExprDoc(const TracedObject<PrimExpr>&)> converter) const;
};

std::vector<BufferPrintInfo> GetBufferPrintInfo(
    const std::vector<TracedObject<tir::Buffer>>& buffers,  //
    std::function<bool(const tir::VarNode*)> f_var_defined,
    std::unordered_map<const tir::VarNode*, ObjectPath>* var_explicit_def,
    AssociatedVariables& associated_vars);

}  // namespace printer
}  // namespace script
}  // namespace tvm

#endif  // TVM_SCRIPT_PRINTER_TIR_BUFFER_H_
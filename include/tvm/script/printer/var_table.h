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
#ifndef TVM_SCRIPT_PRINTER_SYMBOL_TABLE_H_
#define TVM_SCRIPT_PRINTER_SYMBOL_TABLE_H_

#include <tvm/node/node.h>
#include <tvm/node/object_path.h>
#include <tvm/script/printer/doc.h>

namespace tvm {
namespace script {
namespace printer {

/*!
 * \breif Variable Table manages mapping from variable object to ExprDoc during
 * the process of printing TVMScript.
 *
 * The value type of this map is ExprDoc rather than IdDoc or String. It's
 * because variables can be implicitly defined. For example in TIR buffer (tir::Buffer),
 * `buf->data` is a variable, while its representation in TVMScript should be an
 * expression `x.data`, where `x` is the variable for the buffer itself.
 */
class VarTableNode : public Object {
 public:
  void VisitAttrs(AttrVisitor*) {}

  using DocFactory = std::function<ExprDoc()>;

  /*!
   * \brief Define variable by doc factory.
   * \param obj The variable object.
   * \param doc_factory The function to return an ExprDoc object for this variable.
   * \param object_path The object_path for the returned ExprDoc.
   *
   * \return The doc for this variable.
   *
   * This function takes a DocFactory instead of Doc. It's because GetVarDoc needs to
   * return a new Doc object with the object_path parameter as its source path. Currently
   * there isn't a good way to deep copy a TVMObject so it needs to have a DocFactory to
   * do this work.
   */
  ExprDoc Define(const ObjectRef& obj, DocFactory doc_factory, const ObjectPath& object_path);

  /*!
   * \brief Define variable by name.
   * \param obj The variable object.
   * \param name_hint The hint for variable name.
   * \param object_path The object_path for the returned ExprDoc.
   *
   * \return The id doc for this variable.
   *
   * This function will rename the variable to avoid name conflict with other variables
   * in the table.
   */
  IdDoc Define(const ObjectRef& obj, const String& name_hint, const ObjectPath& object_path);

  /*!
   * \brief Remove variable.
   * \param obj The variable to remove.
   */
  void Remove(const ObjectRef& obj);

  /*!
   * \brief Get the doc for variable.
   * \param obj The variable object.
   * \param object_path The object path for the variable.
   *
   * \return The doc for variable, if it exists in the table. Otherwise it returns NullOpt.
   */
  Optional<ExprDoc> GetVarDoc(const ObjectRef& obj, const ObjectPath& object_path) const;

  /*!
   * \brief Check if a variable exists in the table.
   * \param obj The variable object.
   *
   * \return a boolean for whether variable exists.
   */
  bool IsVarDefined(const ObjectRef& obj) const;

  static constexpr const char* _type_key = "script.printer.VarTable";
  TVM_DECLARE_FINAL_OBJECT_INFO(VarTableNode, Object);

 private:
  struct VariableInfo {
    DocFactory doc_factory;
    Optional<String> name;
  };
  std::unordered_map<ObjectRef, VariableInfo, ObjectPtrHash, ObjectPtrEqual> obj2info;
  std::unordered_set<String> defined_names;
};

/*!
 * \breif Reference type of VarTableNode.
 */
class VarTable : public ObjectRef {
 public:
  /*!
   * \brief Create an empty VarTable.
   */
  VarTable();
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(VarTable, ObjectRef, VarTableNode);
};

}  // namespace printer
}  // namespace script
}  // namespace tvm

#endif  // TVM_SCRIPT_PRINTER_SYMBOL_TABLE_H_

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

#include "./doc.h"

namespace tvm {
namespace script {
namespace printer {

class SymbolTableNode : public Object {
 public:
  Map<String, ObjectRef> name2obj;
  Map<ObjectRef, ExprDoc> obj2doc;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("name2obj", &name2obj);
    v->Visit("obj2doc", &obj2doc);
  }

  IdDoc DefByName(const ObjectRef& obj, const String& name) {
    ICHECK(!name2obj.count(name)) << "Duplicated name: " << name;
    ICHECK(!obj2doc.count(obj)) << "Duplicated object: " << obj;
    IdDoc doc = IdDoc(name);
    name2obj.Set(name, obj);
    obj2doc.Set(obj, doc);
    return doc;
  }

  ExprDoc DefByDoc(const ObjectRef& obj, const ExprDoc& doc) {
    ICHECK(!obj2doc.count(obj)) << "Duplicated object: " << obj;
    obj2doc.Set(obj, doc);
    return doc;
  }

  void UndefByObject(const ObjectRef& obj) {
    ExprDoc doc = obj2doc[obj];
    if (const auto* id = doc.as<IdDocNode>()) {
      name2obj.erase(id->name);
    }
    obj2doc.erase(obj);
  }

  Optional<ExprDoc> GetObjectDoc(const ObjectRef& obj) const {
    auto it = obj2doc.find(obj);
    if (it == obj2doc.end()) {
      return NullOpt;
    }
    return (*it).second;
  }

  String GetUniqueName(const String& prefix) const {
    String name = prefix;
    for (int i = 1; name2obj.count(name) != 0; ++i) {
      name = prefix + "_" + std::to_string(i);
    }
    return name;
  }

  static constexpr const char* _type_key = "script.SymbolTable";
  TVM_DECLARE_FINAL_OBJECT_INFO(SymbolTableNode, Object);
};

class SymbolTable : public ObjectRef {
 public:
  SymbolTable();
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(SymbolTable, ObjectRef, SymbolTableNode);
};

}  // namespace printer
}  // namespace script
}  // namespace tvm

#endif  // TVM_SCRIPT_PRINTER_SYMBOL_TABLE_H_

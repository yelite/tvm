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

#include <../../src/script/printer/traced_object.h>
#include <dmlc/logging.h>
#include <gtest/gtest.h>
#include <tvm/node/repr_printer.h>
#include <tvm/runtime/container/map.h>

using namespace tvm;

namespace {

class DummyObjectNode : public Object {
 public:
  void VisitAttrs(AttrVisitor* v) {}

  TVM_DECLARE_FINAL_OBJECT_INFO(DummyObjectNode, Object);
};

class DummyObject : public ObjectRef {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(DummyObject, ObjectRef, DummyObjectNode);
};

TVM_REGISTER_NODE_TYPE(DummyObjectNode);

class ObjectWithAttrsNode : public Object {
 public:
  int64_t int64_attr = 5;
  Map<String, String> map_attr;
  Array<String> array_attr;
  DummyObject obj_attr;

  ObjectWithAttrsNode() : obj_attr(make_object<DummyObjectNode>()) {}

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("int64_attr", &int64_attr);
    v->Visit("map_attr", &map_attr);
    v->Visit("array_attr", &array_attr);
    v->Visit("obj_attr", &obj_attr);
  }

  TVM_DECLARE_FINAL_OBJECT_INFO(ObjectWithAttrsNode, Object);
};

class ObjectWithAttrs : public ObjectRef {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(ObjectWithAttrs, ObjectRef, ObjectWithAttrsNode);
};

TVM_REGISTER_NODE_TYPE(ObjectWithAttrsNode);

}  // anonymous namespace

TEST(TracedObjectTest, MakeTraced_RootObject) {
  ObjectWithAttrs root(make_object<ObjectWithAttrsNode>());
  auto root_traced = MakeTraced(root);

  static_assert(std::is_same<decltype(root_traced), TracedObject<ObjectWithAttrs>>::value);
  ICHECK(root_traced.GetPath().PathsEqual(ObjectPath::Root()));
  ICHECK_EQ(root_traced.Get().get(), root.get());
}

TEST(TracedObjectTest, GetAttr_ObjectRef) {
  ObjectWithAttrs root(make_object<ObjectWithAttrsNode>());
  auto root_traced = MakeTraced(root);
  auto obj_attr = root_traced.GetAttr(&ObjectWithAttrsNode::obj_attr);
  static_assert(std::is_same<decltype(obj_attr), TracedObject<DummyObject>>::value);
  ICHECK(obj_attr.GetPath().PathsEqual(ObjectPath::Root()->Attr("obj_attr")));
  ICHECK_EQ(obj_attr.Get().get(), root->obj_attr.get());
}

TEST(TracedObjectTest, GetAttr_Map) {
  ObjectWithAttrs root(make_object<ObjectWithAttrsNode>());
  root->map_attr.Set("foo", "bar");

  auto root_traced = MakeTraced(root);
  auto map_attr = root_traced.GetAttr(&ObjectWithAttrsNode::map_attr);
  static_assert(std::is_same<decltype(map_attr), TracedMap<String, String>>::value);
  ICHECK(map_attr.GetPath().PathsEqual(ObjectPath::Root()->Attr("map_attr")));
  ICHECK_EQ(map_attr.Get().get(), root->map_attr.get());

  auto map_val = map_attr.at("foo");
  ICHECK_EQ(map_val.Get(), "bar");
  ICHECK(
      map_val.GetPath().PathsEqual(ObjectPath::Root()->Attr("map_attr")->MapValue(String("foo"))));
}

TEST(TracedObjectTest, GetAttr_Array) {
  ObjectWithAttrs root(make_object<ObjectWithAttrsNode>());
  root->array_attr.push_back("foo");
  root->array_attr.push_back("bar");

  auto root_traced = MakeTraced(root);
  auto array_attr = root_traced.GetAttr(&ObjectWithAttrsNode::array_attr);
  static_assert(std::is_same<decltype(array_attr), TracedArray<String>>::value);
  ICHECK(array_attr.GetPath().PathsEqual(ObjectPath::Root()->Attr("array_attr")));
  ICHECK_EQ(array_attr.Get().get(), root->array_attr.get());

  auto array_val = array_attr[1];
  ICHECK_EQ(array_val.Get(), "bar");
  ICHECK(array_val.GetPath().PathsEqual(ObjectPath::Root()->Attr("array_attr")->ArrayIndex(1)));
}

TEST(TracedObjectTest, GetAttr_Int64) {
  ObjectWithAttrs root(make_object<ObjectWithAttrsNode>());
  auto root_traced = MakeTraced(root);

  auto int64_attr = root_traced.GetAttr(&ObjectWithAttrsNode::int64_attr);
  static_assert(std::is_same<decltype(int64_attr), TracedBasicValue<int64_t>>::value);
  ICHECK_EQ(int64_attr.Get(), 5);
  ICHECK(int64_attr.GetPath().PathsEqual(ObjectPath::Root()->Attr("int64_attr")));
}

TEST(TracedObjectTest, MapIterator) {
  Map<String, String> m({{"k1", "foo"}, {"k2", "bar"}});
  auto traced = MakeTraced(m);

  size_t k1_count = 0;
  size_t k2_count = 0;

  for (const auto& kv : traced) {
    if (kv.first == "k1") {
      ++k1_count;
      ICHECK_EQ(kv.second.Get(), "foo");
      ICHECK(kv.second.GetPath().PathsEqual(ObjectPath::Root()->MapValue(String("k1"))));
    } else if (kv.first == "k2") {
      ++k2_count;
      ICHECK_EQ(kv.second.Get(), "bar");
      ICHECK(kv.second.GetPath().PathsEqual(ObjectPath::Root()->MapValue(String("k2"))));
    } else {
      ICHECK(false);
    }
  }

  ICHECK_EQ(k1_count, 1);
  ICHECK_EQ(k2_count, 1);
}

TEST(TracedObjectTest, ArrayIterator) {
  Array<String> a = {"foo", "bar"};
  auto traced = MakeTraced(a);

  size_t index = 0;
  for (const auto& x : traced) {
    if (index == 0) {
      ICHECK_EQ(x.Get(), "foo");
      ICHECK(x.GetPath().PathsEqual(ObjectPath::Root()->ArrayIndex(0)));
    } else if (index == 1) {
      ICHECK_EQ(x.Get(), "bar");
      ICHECK(x.GetPath().PathsEqual(ObjectPath::Root()->ArrayIndex(1)));
    } else {
      ICHECK(false);
    }
    ++index;
  }

  ICHECK_EQ(index, 2);
}

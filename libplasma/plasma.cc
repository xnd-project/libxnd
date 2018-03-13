#include <iostream>
#include "plasma/client.h"

#include "plasma.h"

extern "C" {
  ObjectID objectID_from_random() {
    plasma::ObjectID * object_id = new plasma::ObjectID;
    *object_id = plasma::ObjectID::from_random();
    return reinterpret_cast<ObjectID>(object_id);
  }
  const uint8_t * objectID_data(ObjectID v){
    return reinterpret_cast<plasma::ObjectID*>(v)->data();
  }
  const char * objectID_hex(ObjectID v){
    return reinterpret_cast<plasma::ObjectID*>(v)->hex().c_str();
  }
}

void print(const char *t) {
   if (*t == '\0')
      return;
   printf("%c", *t);
   print(++t);
}

int main() {
    // plasma::PlasmaClient client;
    // client.Connect("/tmp/store", "", 0);
    // plasma::ObjectID object_id = plasma::ObjectID::from_random();
    // int64_t data_size = 100;
    // uint8_t metadata[] = {5};
    // int64_t metadata_size = sizeof(metadata);
    // std::shared_ptr<Buffer> data;
    // client.Create(object_id, data_size, metadata, metadata_size, &data);
    print(objectID_hex(objectID_from_random()));
}

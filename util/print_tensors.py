from tensorflow.python import pywrap_tensorflow
import os
model_dir = "/home/sid/rddnn/models/"
checkpoint_path = os.path.join(model_dir, "test.ckpt")
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
for key in var_to_shape_map:
    print("tensor_name: ", key)
   # print(reader.get_tensor(key))
#print(reader.get_tensor("fc4_weights"))
print("===========fc3_wieghts(test)================")
print(reader.get_tensor("fc3_weights"))
print("===========fc6_wieghts(test)================")
print(reader.get_tensor("fc6_weights"))

train_checkpoint_path = os.path.join(model_dir, "trained.ckpt")
reader_train = pywrap_tensorflow.NewCheckpointReader(train_checkpoint_path)
print("===========fc3_wieghts(train)================")
print(reader_train.get_tensor("fc3_weights"))
print("===========fc6_wieghts(train)================")
print(reader_train.get_tensor("fc6_weights"))

print("Train tensor names ==========================")
var_to_shape_map_train = reader_train.get_variable_to_shape_map()
for key in var_to_shape_map_train:
    print("tensor_name: ", key)
 

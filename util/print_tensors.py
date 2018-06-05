from tensorflow.python import pywrap_tensorflow
import os
model_dir = "/home/sid/rddnn/models/"
checkpoint_path = os.path.join(model_dir, "test_model_.ckpt")
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
for key in var_to_shape_map:
    print("tensor_name: ", key)
   # print(reader.get_tensor(key))
print(reader.get_tensor("fc4_weights"))
print("===========================")
print(reader.get_tensor("fc3_weights"))

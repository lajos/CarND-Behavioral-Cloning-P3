# Running this code on Windows 10 produces the following error:
#
# Traceback (most recent call last):
#   File "c:\Projects\udacity\carnd\tmp\test.py", line 7, in <module>
#     model.save('model.h5')
#   File "C:\Dev\miniconda3\envs\carnd-term1\lib\site-packages\keras\engine\topology.py", line 2416, in save
#     save_model(self, filepath, overwrite)
#   File "C:\Dev\miniconda3\envs\carnd-term1\lib\site-packages\keras\models.py", line 101, in save_model
#     'config': model.get_config()
#   File "C:\Dev\miniconda3\envs\carnd-term1\lib\site-packages\keras\models.py", line 1176, in get_config
#     'config': layer.get_config()})
#   File "C:\Dev\miniconda3\envs\carnd-term1\lib\site-packages\keras\layers\core.py", line 668, in get_config
#     function = func_dump(self.function)
#   File "C:\Dev\miniconda3\envs\carnd-term1\lib\site-packages\keras\utils\generic_utils.py", line 177, in func_dump
#     code = marshal.dumps(func.__code__).decode('raw_unicode_escape')
# UnicodeDecodeError: 'rawunicodeescape' codec can't decode bytes in position 89-93: truncated \uXXXX
#
#
# how to fix:
#
# modify the following line in C:\Dev\miniconda3\envs\carnd-term1\Lib\site-packages\keras\utils\generic_utils.py:
#
# code = marshal.dumps(func.__code__).decode('raw_unicode_escape')
#
# to:
#
# code = marshal.dumps(func.__code__).replace(b'\\',b'/').decode('raw_unicode_escape')
#
#


from keras.models import Sequential
from keras.layers import Lambda

model = Sequential()
model.add(Lambda(lambda x: (x/255.0)-0.5, input_shape = (2,2)))

model.save('model.h5')

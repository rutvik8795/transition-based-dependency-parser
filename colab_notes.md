This file is best viewed in Markdown reader (eg. https://jbt.github.io/markdown-editor/)

----

Follow these steps for running experiments on google colab with gpu support:

1. Open a new google colab notebook and enable GPU runtime.
2. Create a private github repository and push your finalized code to it.
3. Copy paste the following commands into google colab. Each block corresponds to one cell.

**Note:** Make sure it's a private repository. If someone copies code from you we will have no way of knowing who plagiarized.

----

Make sure tf2 is being used:
```
%tensorflow_version 2.x
```

Clone your **private** repository:
```
import os
from getpass import getpass
import urllib

user = input('User name: ')
password = getpass('Password: ')
password = urllib.parse.quote(password) # your password is converted into url format
repo_name = input('Repo name: ')

cmd_string = 'git clone https://{0}:{1}@github.com/{0}/{2}.git'.format(user, password, repo_name)

os.system(cmd_string)
cmd_string, password = "", "" # removing the password from the variable
```

Cd into the repository:
```
cd your-repository-name/
```

Make sure GPU runtime is on:
```
import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))
```

Download glove and unarchive:
```
!./download_glove.sh
```

Run all the experiments in a go. It will take about 15X5 minutes. It would be better if you copy paste one experiment from `experiments.sh`.
```
!./experiments.sh
```

zip the serialization directories:
```
!zip -r serialization_dirs.zip serialization_dirs/
```

Download the zipped serialization directories:
```
from google.colab import files
files.download('serialization_dirs.zip')
```

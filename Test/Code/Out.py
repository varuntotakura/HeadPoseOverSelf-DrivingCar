'''
## Licence:

This repository contains a variety of content; some developed by VARUN, and some from third-parties.
The third-party content is distributed under the license provided by those parties.
The content developed by VARUN is distributed under the following license:
I am providing code and resources in this repository to you under an open source license.
Because this is my personal repository, the license you receive to my code and resources is from me.

More about Licence at [link](https://github.com/t-varun/Face-Recognition/blob/master/LICENSE).
'''

import numpy as np
import pandas as pd
import cv2
from random import shuffle

train_data = np.load('../training_data.npy')

TOTAL = []

for data in train_data:
    img = data[0]
    person = data[1]
    if person == "VARUN":
        TOTAL.append([img, 0])
    elif person == "BUNNY":
        TOTAL.append([img, 1])

shuffle(TOTAL)
np.save('../training_data_cleaned.npy', TOTAL)

##for data in train_data:
##    img = data[0]
##    person = data[1]
##    cv2.imshow('test', img)
##    print(person)
##    if cv2.waitKey(25) & 0xFF == ord('q'):
##        cv2.destroyAllWindows()
##        break
    

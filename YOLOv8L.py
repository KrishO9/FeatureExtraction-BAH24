
import ultralytics
ultralytics.checks()








#!yolo predict model='runs/detect/train10_200epoch_custom_data/weights/yolo8m_ep200_b16.pt' source='Test_Datasets/Farm_Pond'








# In[2]:



# In[9]:



# In[7]:




# In[6]:




# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
# <img align="left" src="https://user-images.githubusercontent.com/26833433/212889447-69e5bdf1-5800-4e29-835e-2ed2336dede2.jpg" width="600">

# # 2. Val
# Validate a model's accuracy on the [COCO](https://docs.ultralytics.com/datasets/detect/coco/) dataset's `val` or `test` splits. The latest YOLOv8 [models](https://github.com/ultralytics/ultralytics#models) are downloaded automatically the first time they are used. See [YOLOv8 Val Docs](https://docs.ultralytics.com/modes/val/) for more information.

# In[ ]:


# Download COCO val
import torch
torch.hub.download_url_to_file('https://ultralytics.com/assets/coco2017val.zip', 'tmp.zip')  # download (780M - 5000 images)
 # unzip

# In[ ]:


# Validate YOLOv8n on COCO8 val
# !yolo val model=yolov8l.pt data=coco8.yaml

# # 3. Train
# 
# <p align=""><a href="https://ultralytics.com/hub"><img width="1000" src="https://github.com/ultralytics/assets/raw/main/yolov8/banner-integrations.png"/></a></p>
# 
# Train YOLOv8 on [Detect](https://docs.ultralytics.com/tasks/detect/), [Segment](https://docs.ultralytics.com/tasks/segment/), [Classify](https://docs.ultralytics.com/tasks/classify/) and [Pose](https://docs.ultralytics.com/tasks/pose/) datasets. See [YOLOv8 Train Docs](https://docs.ultralytics.com/modes/train/) for more information.

# In[3]:


#@title Select YOLOv8 🚀 logger {run: 'auto'}
# logger = 'Comet' #@param ['Comet', 'TensorBoard']

# if logger == 'Comet':
#  import comet_ml; comet_ml.init()
# elif logger == 'TensorBoard':
 

# In[2]:


# import comet_ml

# comet_ml.api_key = "nXn8mFNRu9h2NqCA0vO6Gya1f"
# comet_ml.login(project_name="your_project_name")

# In[ ]:


# Train YOLOv8n on COCO8 for 3 epochs
# !yolo train model=yolov8l.pt data=coco8.yaml epochs=3 imgsz=640


# In[5]:


# Train YOLOv8n on COCO8 for 3 epochs
# from ultralytics import YOLO
# model = YOLO("yolov8l.yaml")
# !yolo train model=yolov8l.pt data='my_dataset2/data.yaml' epochs=200 batch=16 imgsz=640 flipud=0.2 mosaic =0.2 degrees= 90 patience=100 


from ultralytics import YOLO

# Load the model
model = YOLO("yolov8l.yaml")

# Train the model with the specified parameters
model.train(data='my_dataset2/data.yaml', 
            epochs=200, 
            batch=16, 
            imgsz=640, 
            flipud=0.2, 
            mosaic=0.2, 
            degrees=90, 
            patience=100)



# !yolo export model=yolov8l.pt format=torchscript
model.export(format='torchscript')

# # 5. Python Usage
# 
# YOLOv8 was reimagined using Python-first principles for the most seamless Python YOLO experience yet. YOLOv8 models can be loaded from a trained checkpoint or created from scratch. Then methods are used to train, val, predict, and export the model. See detailed Python usage examples in the [YOLOv8 Python Docs](https://docs.ultralytics.com/usage/python/).

# In[ ]:


# from ultralytics import YOLO

# # Load a model
# model = YOLO('yolov8n.yaml')  # build a new model from scratch
# model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

# # Use the model
# results = model.train(data='coco8.yaml', epochs=3)  # train the model
# results = model.val()  # evaluate model performance on the validation set
# results = model('https://ultralytics.com/images/bus.jpg')  # predict on an image
# results = model.export(format='onnx')  # export the model to ONNX format



# # Load YOLOv8n, train it on COCO128 for 3 epochs and predict an image with it
# from ultralytics import YOLO

# model = YOLO('yolov8n.pt')  # load a pretrained YOLOv8n detection model
# model.train(data='coco8.yaml', epochs=3)  # train the model
# model('https://ultralytics.com/images/bus.jpg')  # predict on an image







# ## 4. Oriented Bounding Boxes (OBB)
# 
# YOLOv8 _OBB_ models use the `-obb` suffix, i.e. `yolov8n-obb.pt` and are pretrained on the DOTA dataset. See [OBB Docs](https://docs.ultralytics.com/tasks/obb/) for full details.

# In[ ]:


# Load YOLOv8n-obb, train it on DOTA8 for 3 epochs and predict an image with it
# from ultralytics import YOLO

# model = YOLO('yolov8n-obb.pt')  # load a pretrained YOLOv8n OBB model
# model.train(data='coco8-dota.yaml', epochs=3)  # train the model
# model('https://ultralytics.com/images/bus.jpg')  # predict on an image




# Pip install from source
# !pip install git+https://github.com/ultralytics/ultralytics@main

# In[ ]:


# Git clone and run tests on updates branch
# !git clone https://github.com/ultralytics/ultralytics -b main
# %pip install -qe ultralytics

# In[ ]:


# Run tests (Git clone only)
# !pytest ultralytics/tests

# In[ ]:


# Validate multiple models
# for x in 'nsmlx':
  # !yolo val model=yolov8{x}.pt data=coco.yaml

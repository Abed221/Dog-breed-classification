# Image Classification using AWS SageMaker

Use AWS Sagemaker to train a pretrained model that can perform image classification by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices. This can be done on either the provided dog breed classication data set or one of your choice.

## Project Set Up and Installation
Enter AWS through the gateway in the course and open SageMaker Studio. 
Download the starter files.
Download/Make the dataset available. 

## Dataset
The provided dataset is the dogbreed classification dataset which can be found in the classroom.
The project is designed to be dataset independent so if there is a dataset that is more interesting or relevant to your work, you are welcome to use it to complete the project.

### Access
Upload the data to an S3 bucket through the AWS Gateway so that SageMaker has access to the data. 

## Hyperparameter Tuning
I used ResNet18 for this project as it's basically designed for image identification

The three hyperparamters that I tuned over my training job were:

- batch_size
- epochs
- lr

Here you can see a screenshot including all of the training jobs:
![Training Running](https://github.com/Abed221/Dog-breed-classification/raw/main/screenshots/training-jobs.png)

And this a screenshot of hyperparameter tuning jobs that were done in this project:
![hpo Running](https://github.com/Abed221/Dog-breed-classification/raw/main/screenshots/hpt-jobs.png)

The best model we obtain was built using these hyperparameters:

- batch_size : 128
- epochs : 4
- lr : 0.005200376250895876

## Debugging and Profiling

Debugging and profiling was performed according to the the following rules:

```python
    ProfilerRule.sagemaker(rule_configs.LowGPUUtilization()),
    ProfilerRule.sagemaker(rule_configs.ProfilerReport()),
    Rule.sagemaker(rule_configs.vanishing_gradient()),
    Rule.sagemaker(rule_configs.overfit()),
```

To allow for debugging; the `train_model.py` has been created based on the `hpo.py` and setting up hooks inside functions of training and testing.

### Results


The results of the debugging/profiling session arose the following output:


VanishingGradient: NoIssuesFound
Overfit: NoIssuesFound
LowGPUUtilization: NoIssuesFound
ProfilerReport: NoIssuesFound

You can also read the profiler report html file in case of finding any issues in one of the above


## Model Deployment
To deploy the model, it was required to create a python script called inference.py which loads the model and transforms the input image.

To call the model, you execute inference.py replacing IMAGE_PATH by the path where your image is stored and ENDPOINT by the name of your endpoint.
```python
import io
import sagemaker
from PIL import Image
from sagemaker.serializers import IdentitySerializer
from sagemaker.pytorch.model import PyTorchPredictor

serializer = IdentitySerializer("image/jpeg")
predictor = PyTorchPredictor(ENDPOINT, serializer=serializer, sagemaker_session=sagemaker.Session())

buffer = io.BytesIO()
Image.open(IMAGE_PATH).save(buffer, format="JPEG")
response = predictor.predict(buffer.getvalue())
```

And as we can in this screenshot, the endpoint is deployed and in service:
![Endpoint Running](https://github.com/Abed221/Dog-breed-classification/raw/main/screenshots/Endpoint.PNG)



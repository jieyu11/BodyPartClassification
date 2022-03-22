# Body Parts Image Classification
The functions in the *modeling* folder are performing the image classification regardless of the actual
contents in the images. As long as the data location, image classes, outputs and model
hyperparameters, etc are properly set in the configuration file, it will train and the classification
model.

## Datasets
Any data used for fitting is expected to have at least 2 classes (and more is fine). Each class data is
located in one folder. All images in the corresponding class's folder is scanned and used for training, 
validation and testing.

The location of the datasets are specified in the config file.

### Feet vs Shoes Dataset
In the following example, the dataset is copied from:
```
/home/jie/3dselfie/3DSelfie/data/feet-vs-shoes/bare_feet
/home/jie/3dselfie/3DSelfie/data/feet-vs-shoes/with_shoes
```
to where one runs this example:
```
/<some-path>/tron/demo/feet-vs-shoes/bare_feet
/<some-path>/tron/demo/feet-vs-shoes/with_shoes
```

## Train and Testing a Classification Model

### Configuration
In the following example, it is assumed that the configuration files are in:
```
/<some-path>/tron/demo/feet-vs-shoes/model_train.toml
/<some-path>/tron/demo/feet-vs-shoes/model_test.toml
```

And this location `/<some-path>/tron/` is the work directory.

The following lines in the `model_train.toml` file specifies the location and the labels of the input images:
```
[images]
folders = ["/data/feet-vs-shoes/bare_feet", "/data/feet-vs-shoes/with_shoes"]
labels = ["feet", "shoes"]
```
where, with the current setup, your local `/<some-path>/tron/demo` is mounted to `/data` in docker containers.
To make sure the input images exist, try to list them with:
`ls demo/feet-vs-shoes/bare_feet demo/feet-vs-shoes/with_shoes`

### Build a Docker Image
A docker image is first built with:
```
cd setup/
./gepetto_build.sh
```
where it builds an image named `bodypartclassification:$TAG` (`$TAG` is user specific.)

### Run Model Training
Create a file e.g. `bodypart-modeling.py` under `/<some-path>/tron/` with the following content:

```
from modules.internal.bodypartclassification.gepetto.bodypartclassifier import \
    BodyPartClassifierImpl

# docker image TAG, modify it to use your own!
docker_tag = "3af2dc_jie"

# this indicates the output is in demo/
output_filename = "demo/model_shoes-vs-feet.h5"
input_config = "demo/feet-vs-shoes/model_train.toml"
parameter, values = None, None

# True for testing and False for training
testing = False
impl = BodyPartClassifierImpl(
    input_config=input_config,
    output_filename=output_filename, tag=docker_tag,
    tune_parameter=parameter, tune_values=values,
    testing=testing)
impl.run()
```

Run the above code with:
`python bodypart-modeling.py`

### Run Model Testing
To run model testing, one needs to setup additional dataset that hasn't been used in the model training.
A small dataset is prepared, and one can copy it to the data directory:
`cp -r /home/jie/3dselfie/3DSelfie/data/feet-vs-shoes/mpii_part_labeled_jy /<some-path>/tron/demo/feet-vs-shoes/`

Two things needs to be changed in the above code to run testing:
```
input_config = "demo/feet-vs-shoes/model_test.toml"
testing = True
```

Run the above code with:
`python bodypart-modeling.py`


## Hyperparameters Tuning
The parameter name the corresponding values to tune are setup with e.g. 

* Model Complexity Choice
    ```
    parameter = "pre_model_name"
    values = ["EfficientNetB%d" % i for i in range(8)]
    ```
* Learning Rate
    ```
    parameter = "learning_rate"
    values = [1.0/pow(10, n) for n in range(1, 10)]
    ```

* Image Size
    ```
    parameter = "image_size"
    values = [128, 160, 192, 256, 320, 384, 512, 768, 1024, 1280, 1536, 2048]
    ```

which needs to be setup in `bodypart-modeling.py` one at a time.

# OpenImages Dataset Image Filter
OpenImages is a very large image dataset, which has 600 classes of objects from 1.9M images. They are
annotated by human or by annotation tools and can be found from: 
https://storage.googleapis.com/openimages/web/download.html

They are downloaded to shared disk reachable from any of our servers, e.g. anakin, at this address:
`/home/jie/3dselfie/3DSelfie/data/openimages/images/`

The dataset is so large that it needs a filtering tool to select the images. 

## Image Filtering
The image filtering code is found in `image_filter.py`. A shell script to run this filter is like, assuming the
current folder contains `image_filter.py` and the dataset is in `/home/jie/3dselfie/3DSelfie/data/openimages`:
```
number_of_images=5000
nvidia-docker container run -it --rm --name  jyu-image-filtering \
    -v /home/jie/3dselfie/3DSelfie/data/openimages:/openimages \
    -v ${PWD}:/workarea \
    bodypartclassification:3af2dc_jie \
    /bin/bash -c "cd /workarea; \
    python image_filter.py -l \
    /openimages/meta-info/class-descriptions-boxable.csv \
    -o /openimages/meta-info/oidv6-train-annotations-vrd.csv \
    -img /openimages/images/train_0 -n ${number_of_images}"
```

The `number_of_images` is to set to limit the amount of images in the output.

The output images are found: `${PWD}/output`, which contains the csv file which has
the list of image IDs to contain the filtered items. The pre-defined filtered object names
are defined in `image_filter.py`. A future development will move those names to a 
configuration file.

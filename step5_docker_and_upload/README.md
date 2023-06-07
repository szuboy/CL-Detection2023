
# CL-detection 2023 reference docker

This docker image contains a reference implementation of RetinaNet with ResNet101 using MMdetection toolbox for the CL-detection 2023 challenge which has been trained for 100 epochs.

The container will serve as a reference of how the organizer and the grand-challenge.org plattform expect the pre-defined outputs. Additionally, this reference serves as a baseline for participants to implement or propose their own algorithm for the CL-detection 2023 challenge algorithm submission.

For reference, you may also want to read the blog post of grand-challenge.org on [how to create an algorithm](https://grand-challenge.org/blogs/create-an-algorithm/).

## Content:
1. [Prerequisites](#prerequisites)
2. [An overview of the dictionary structure for this example](#overview)
3. [Implementing your algorithm into a docker container image](#todocker)
4. [Building your container](#build)
5. [Testing your container](#test)
6. [Generating the bundle for uploading your algorithm](#export)


## 1. Prerequisites <a name="prerequisites"></a>

The container is based on docker, please [install docker here](https://www.docker.com/get-started). 

Second, you need to clone CL_detection2023_reference_docker repository:
```
git clone https://github.com/cwwang1979/CL_detection2023_reference_docker
```

You also need to install evalutils package:
```
pip install evalutils
```

## 2. An overview of the dictionary structure for this example <a name="overview"></a>

The main inference processing is executed in the file detection.py. It loads the trained model and provides the method process_image() that takes a stacked test image using MMdetection toolbox and returns the detections as shown in figures below.

The main file that is executed by the container is process.py. It imports the trained model and configs file using pickle. It then loads stacked image that are part of the dummy test set and processes (using the process_image() method). Before generating final output in the form of dictionary which contains all individual detected landmark in each corresponding individual image id (z-coordinate), which are ultimately stored in the file /output/orthodontic-landmarks.json.

The output json file is a dictionary and will result as the following format:
```
{   "name": "Orthodontic landmarks",
    "type": "Multiple points",
    "points": [
        {
            "name": "1",
            "point": [
                1916,
                1489,
                1
            ],
        },
	.
	.
	.
        {
            "name": "38",
            "point": [
                1916,
                1489,
                2
            ],
        }
    ],
    "version": {
        "major": 1,
        "minor": 0
    }
```
*Note that each point is described by the following dictionary: image

The dictionary "name" indicates the landmark class.

## 3. Implementing your algorithm into a docker container image <a name="todocker"></a>
We recommend you to develop and adapt this docker image example to your own cephalometric landmark detection solution. You can adapt, modify or build the code from scratch.

If you need a different base image to build your container (e.g., Tensorflow instead of Pytorch, or other AI toolbox), if you need additional libraries and to make sure that all source files (and weights) are copied to the docker container, you will have to adapt the Dockerfile and the requirements.txt file accordingly.

Please refer to the image below (Dockerfile): image
<img width="1299" src="docs/1.png">

## 4. Building your container <a name="build"></a>
To test if all dependencies are met, you should run the file build.bat (Windows) / build.sh (Linux) to build the docker container. Please note that the next step (testing the container) also runs a build, so this step is not mandatory if you are certain that everything is set up correctly.
<img width="1299" src="docs/2.png">

## 5. Testing your container <a name="test"></a>
To test your container, you should run test.bat (on Windows) or test.sh (on Linux, might require sudo priviledges). This will run the test image(s) provided in the test folder through your model. It will check them against what you provide in test/expected_output.json. Be aware that this will, of course, initially not be equal to the demo detections we put there for testing our reference model.

## 6. Generating the bundle for uploading your algorithm <a name="export"></a>
Finally, you need to run the export.sh (Linux) or export.bat script to package your docker image. This step creates a file with the extension "tar.gz", which you can then upload to grand-challenge to submit your algorithm.


## 7. Creating an "Algorithm" as your solution to the CL-detection 2023 Challenge

In order to submit your docker container, you first have to add an Algorithm entry for your docker container here.

Please enter a name for the algorithm:
<img width="1299" src="docs/3.png">
Please note that it can take a while (several minutes) until the container becomes active. You can determine which one is active in the same dialog:
<img width="1299" src="docs/5.png">

Before submit your algorithm, we highly recommend participants to try out their algorithm with small stack image (stack1.mha) which available on [stack1.mha](test/stack1.mha).
<img width="1299" src="docs/6.png">

And make sure your algorithm output has the same format with our pre-defined output.
<img width="1299" src="docs/7.png">

Finally, you can submit your docker container to CL-detection 2023:
1. Deadline for validation submission: July 1, 2023 (00.00 TWT)
<img width="1299" src="docs/4.png">

2. Deadline for testing submission: August 15, 2023 (00.00 TWT)
<img width="1299" src="docs/8.png">

*For testing phase submission, participants are mandatory to submit their first draft of the paper submission* in the LNCS Springer format (with 4 pages).

*Deadline for final paper submission* in the LNCS Springer format (with 4 pages): September 15, 2023 (Participants are able to send the final revised paper until September 15, 2023, to Prof Wang (cweiwang@mail.ntust.edu.tw; cwwang1979@gmail.com) and Mr. Hikam Muzakky (m11123801@mail.ntust.edu.tw))

## License
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)

## Contact Information
-Prof. Ching-Wei Wang : cweiwang@mail.ntust.edu.tw ; cwwang1979@gmail.com
-Mr. Hikam Muzakky : m11123801@mail.ntust.edu.tw

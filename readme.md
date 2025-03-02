<h1 align="center">Camera-based Intruder Detection and Monitoring of Ship Crew Work Hours</h1>
<hr style="border: 1px solid  #256ae2 ;">


```bibtex
(will be added soon)
```

## Get started
Follow these steps to get started:
### 1. Install Requirements
Install Python 3.10 and the necessary dependencies.

```bash
pip install -r requirements.txt
```
### 2. Download Data
Download the datasets from the [link](https://usf.box.com/s/u2dj73hrjfgztmdxr5b52o9ucok93dtq).
Paste the dataset inside ```./data/videos/```. For example, you will get many sub-folders titled by ```dirxxxxx``` in the link. Keep these sub-directories inside the ```./data/videos/``` in such a way, so that
you get ```./data/videos/dirxxxxx``` sub-directories.
 
    
### 3. Experiment
Run ```main.py``` file for the experiments. There are three type of tasks for the experiment: 'face_detection', 'crew_recognition', and 'work_hour_estimation'.

#### 'face_detection':
It is used to detect and crop the faces from the video video dataset. Then we manually labeled each faces with a crew id. You do not need to do this experiment, as we already provided these files in ```./data/face_photos/unsplit_data```.
#### 'crew_recognition':
Experiment to recognize the crew members and to detect the intruders.
#### 'work_hour_estimation':
Experiment for estimating the work hours of the crew memebers.

### Contact
For more information: Please contact mmurad@usf.edu

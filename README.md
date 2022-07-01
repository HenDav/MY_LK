# Lucas-Kanade-tracking-and-Pyramid-Implementation
This repository contains implementation of **Lucas-Kanade algorithm** proposed by Lucas and Kanade. Lucas-Kanade algorithm can be used for **sparse optical flow** (associate feature points across frames) and **tracking** (associate image patch cross frames). This repo implements the algorithm for tracking a single template across 400 frames video.   
This repo is based on [ziliHarvee's repo](https://github.com/ziliHarvey/Lucas-Kanade-tracking-and-Correlation-Filters)
## Lucas Kanade Tracking with one single template and Pyramid  
The "vanilla" algorithm for tracking. Detailed derivation can be referred to [Lucas-Kanade 20 Years On: A Unifying Framework](https://www.ri.cmu.edu/pub_files/pub3/baker_simon_2002_3/baker_simon_2002_3.pdf). This tracker runs around 36 Hz on my local machine.    
**Files included:**     
/data/carseq.npy  
/src/LucasKanadeHenVicImprovements.py
/src/testCarSequence.py  
**Run**
```
python testCarSequence.py
```
**Sample results**  
<img src="https://github.com/ziliHarvey/Lucas-Kanade-tracking-and-Correlation-Filters/raw/master/result/tracking_with_one_single_template/Figure_1.png" width=30% height=30%>
<img src="https://github.com/ziliHarvey/Lucas-Kanade-tracking-and-Correlation-Filters/raw/master/result/tracking_with_one_single_template/Figure_2.png" width=30% height=30%>
<img src="https://github.com/ziliHarvey/Lucas-Kanade-tracking-and-Correlation-Filters/raw/master/result/tracking_with_one_single_template/Figure_3.png" width=30% height=30%>  
/src/carseqrects.npy stores the vertices coordinates of bounding box in each frame.  

---
layout: post
title: OBJ2TEXT implementation with PyTorch
---

方法、demo及源码，参考[http://www.cs.virginia.edu/~xy4cm/obj2text/](http://www.cs.virginia.edu/~xy4cm/obj2text/)

### Download MSCOCO dataset including annotations

`train2014/`, `val2014/`, `test2014/` and `annotations/captions_train2014.json`,  `annotations/captions_val2014.json`

### Collect object detection results on MS COCO dataset

* Checkout the yolo-coco-result branch of this repository[https://github.com/xuwangyin/darknet/tree/yolo-coco-result](https://github.com/xuwangyin/darknet/tree/yolo-coco-result), compile the code and download the weight file named yolo.weights according to https://pjreddie.com/darknet/yolo/, or just run this command ```wget https://pjreddie.com/media/files/yolo.weights```
* List absolute paths of all the MS COCO image files into a text file ```find pwd -name "*.jpg" > image_files.txt```, where `pwd` is the directory that contains MS COCO image dirs `train2014`, `val2014` and `test2014`. 
* Put `image_files.txt` in the darknet proejct directory and run ```./darknet detect cfg/yolo.cfg yolo.weights data/dog.jpg > coco_detection_result``` to collect the result. I think it should be `image_files.txt` instead of `data/dog.jpg`.

Now we get `coco_detection_result` file, dectection result of coco dataset images using yolo.

### Image caption

Clone [https://github.com/xuwangyin/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning](https://github.com/xuwangyin/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning).

#### Processing
```pythob build_vocab.py --caption_path```
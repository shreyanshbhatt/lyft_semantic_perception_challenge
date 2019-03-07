The project consist of semantic segmentation of an image to identify drivable portions
and cars on road.
It is primarily developed as part of the lyft-perception-challenge. It can work for other
datasets. The dataset and the trained model can be downloaded from [here](https://drive.google.com/drive/folders/1Z1bVgFOfrvtVi28Jlg68TfnCpt0bwLpz?usp=sharing).

The evaluation metric is F-score for pixel classification and FPS for processing video to annotate it appropriate pixel class.

main.py consists of code to train the model.
FreezeGraph.py consists of code to freeze the trained model.
graph_utils.py consists of code to annotate a set of images or a video with appropriate labels(colors) for road and cars.

annotated_video.mp4 consists of results for one video. It is doing a good job in identifying the road but not doing a good job in identifying cars. Please find the details in report.pdf.

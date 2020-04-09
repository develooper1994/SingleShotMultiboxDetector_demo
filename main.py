import torchvision
from torchvision.models.detection import faster_rcnn
from torchvision.models.detection.rpn import AnchorGenerator


def change_number_of_output_classes(number_classes=3):
    """
    Download pretrained model and change the predictor.
    https://medium.com/@jonathan_hui/understanding-feature-pyramid-networks-for-object-detection-fpn-45b227b9106c
    I made a better approach with New-SEGAN. I am combining between encoder-decoder skip connections
    with decoder upper layer connections with concat
    :return:
    """
    # download a pretrained "Feature Pyramid Networks for object detection (FPN)" 50 layer model
    frcnn = faster_rcnn.fasterrcnn_resnet50_fpn(pretrained=True)
    # replace the classifier with a new one, user-defined number of classes
    number_classes = number_classes + 1  # 1 class (person) + background
    # number of output features of previous layer for classifier
    in_features = frcnn.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    frcnn.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(in_features, number_classes)


def different_backbone():
    pass


def modify():
    pass


def main():
    """
    Fine tuning(transfer learning)
    :return: None
    """
    change_number_of_output_classes()


if __name__ == "__main__":
    main()

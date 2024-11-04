from ikomia import dataprocess


# --------------------
# - Interface class to integrate the process with Ikomia application
# - Inherits PyDataProcess.CPluginProcessInterface from Ikomia API
# --------------------
class IkomiaPlugin(dataprocess.CPluginProcessInterface):

    def __init__(self):
        dataprocess.CPluginProcessInterface.__init__(self)

    def get_process_factory(self):
        # Instantiate algorithm object
        from train_yolo_v11_classification.train_yolo_v11_classification_process import TrainYoloV11ClassificationFactory
        return TrainYoloV11ClassificationFactory()

    def get_widget_factory(self):
        # Instantiate associated widget object
        from train_yolo_v11_classification.train_yolo_v11_classification_widget import TrainYoloV11ClassificationWidgetFactory
        return TrainYoloV11ClassificationWidgetFactory()

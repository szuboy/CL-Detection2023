
import SimpleITK
from pathlib import Path
import numpy as np
import torch
import json
from skimage import transform as sk_transform

from evalutils import DetectionAlgorithm
from evalutils.validators import UniquePathIndicesValidator, UniqueImagesValidator

# import your model | 导入自己的模型结构
from model import load_model


class Cldetection_alg_2023(DetectionAlgorithm):
    # self._input_path = input_path
    # self._output_file = output_file
    def __init__(self):
        # 请不要修改初始化父类的函数
        super().__init__(
            validators=dict(input_image=(UniqueImagesValidator(), UniquePathIndicesValidator())),
            input_path=Path("/input/images/lateral-dental-x-rays/"),
            output_file=Path("/output/orthodontic-landmarks.json"))

        print("==> Starting...")

        # 使用对应的GPU，注意grand-challenge只有一块GPU，请保证下面的权重加载，加上map_location=self.device设置避免不同设备导致的错误
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 加载自己的模型和权重文件，这里的权重文件路径是 /opt/algorithm/best_model.pt，
        # 这是因为在docker中会将当前目录挂载为 /opt/algorithm/，所以你要访问当前文件夹的任何文件，在代码中的路径都应该是 /opt/algorithm/
        self.model = load_model(model_name='UNet')
        model_weight_path = '/opt/algorithm/best_model.pt'
        self.model.load_state_dict(torch.load(model_weight_path, map_location=self.device))
        self.model.to(self.device)

        print("==> Using ", self.device)
        print("==> Initializing model")
        print("==> Weights loaded")

    def save(self):
        """TODO: 重写父类函数，根据自己的 predict() 函数返回类型，将结果保存在 self._output_file 中"""

        # 因为我们就只传入了一个文件名，则 self._case_results 列表中只有一个元素，我们仅需要取出来进行解码
        # 这个 all_images_predict_landmarks_list 就是 predict() 函数的返回值
        all_images_predict_landmarks_list = self._case_results[0]

        # 将预测结果调整为挑战赛需要的JSON格式，借助字典的形式作为中间类型
        json_dict = {'name': 'Orthodontic landmarks', 'type': 'Multiple points'}

        all_predict_points_list = []
        for image_id, predict_landmarks in enumerate(all_images_predict_landmarks_list):
            for landmark_id, landmark in enumerate(predict_landmarks):
                points = {'name': str(landmark_id + 1),
                          'point': [landmark[0], landmark[1], image_id + 1]}
                all_predict_points_list.append(points)
        json_dict['points'] = all_predict_points_list

        # 提交的版本信息，可以为自己的提交备注不同的版本记录
        major = 1
        minor = 0
        json_dict['version'] = {'major': major, 'minor': minor}

        # 转为JSON接受的字符串形式
        json_string = json.dumps(json_dict, indent=4)
        with open(str(self._output_file), "w") as f:
            f.write(json_string)

    def process_case(self, *, idx, case):
        """!IMPORTANT: 请不要修改这个函数的任何内容，下面是具体的注释信息"""

        # 调用父类的加载函数，case 这个变量包含当前的堆叠了所有测试图片的文件名，类似如：/.../../test_stack_image.mha
        input_image, input_image_file_path = self._load_input_image(case=case)

        # 传入对应的 input_image SimpleITK.Image 格式
        predict_result = self.predict(input_image=input_image)

        # 返回预测结果出去
        return predict_result

    def predict(self, *, input_image: SimpleITK.Image):
        """TODO: 请修改这里的逻辑，执行自己设计的模型预测，返回值可以是任何形式的"""

        # 将 SimpleITK.Image 格式转为 Numpy.ndarray 格式进行处理， stacked_image_array 的形状为 (100, 2400, 2880, 3)
        stacked_image_array = SimpleITK.GetArrayFromImage(input_image)

        # 所有图像的预测结果
        all_images_predict_landmarks_list = []

        # 遍历每一张图片进行预测
        with torch.no_grad():
            self.model.eval()
            for i in range(stacked_image_array.shape[0]):
                # 索引出来每一张单独的图片
                image = np.array(stacked_image_array[i, :, :, :])

                # 图片的预处理操作
                torch_image, image_info_dict = self.preprocess_one_image(image)

                # 模型的预测
                predict_heatmap = self.model(torch_image)

                # 结果的后处理
                predict_landmarks = self.postprocess_model_prediction(predict_heatmap, image_info_dict=image_info_dict)

                # 将模型的预测结果存在
                all_images_predict_landmarks_list.append(predict_landmarks)

        print("==========================================")
        print('The prediction is successfully generated!!')
        print("==========================================")

        return all_images_predict_landmarks_list

    def preprocess_one_image(self, image_array: np.ndarray):
        """TODO：这是一个自定义的函数，功能是对每一张图片输入前的预处理操作"""

        # 下面的一段操作是去除多余的0填充
        row = np.sum(image_array, axis=(1, 2))
        column = np.sum(image_array, axis=(0, 2))

        non_zero_row_indices = np.argwhere(row != 0)
        non_zero_column_indices = np.argwhere(column != 0)

        last_row = int(non_zero_row_indices[-1])
        last_column = int(non_zero_column_indices[-1])

        image_array = image_array[:last_row + 1, :last_column + 1, :]

        # 图像的基本信息，保证可以给到其他函数使用
        image_info_dict = {'width': np.shape(image_array)[1], 'height': np.shape(image_array)[0]}

        # 缩放图像
        scaled_image_array = sk_transform.resize(image_array, (512, 512), mode='constant', preserve_range=False)

        # 调整通道位置，增加一个batch-size格式，并转为torch格式
        transpose_image_array = np.transpose(scaled_image_array, (2, 0, 1))
        torch_image = torch.from_numpy(transpose_image_array[np.newaxis, :, :, :])

        # 转到特定的device上
        torch_image = torch_image.float().to(self.device)

        return torch_image, image_info_dict

    def postprocess_model_prediction(self, predict_heatmap: torch.Tensor, image_info_dict: dict):
        """TODO: 对模型的预测结果进行解码，关键点的预测坐标值"""

        # 得到一些必要的图像信息进行后处理
        width, height = image_info_dict['width'], image_info_dict['height']

        # 转为Numpy矩阵进行处理: 去除梯度，转为CPU，转为Numpy
        predict_heatmap = predict_heatmap.detach().cpu().numpy()

        # 去除第一个batch-size维度
        predict_heatmap = np.squeeze(predict_heatmap)

        # 遍历不同的热图通道，得到最后的输出值
        landmarks_list = []
        for i in range(np.shape(predict_heatmap)[0]):
            # 索引得到不同的关键点热图
            landmark_heatmap = predict_heatmap[i, :, :]
            yy, xx = np.where(landmark_heatmap == np.max(landmark_heatmap))
            # there may be multiple maximum positions, and a simple average is performed as the final result
            x0, y0 = np.mean(xx), np.mean(yy)
            # zoom to original image size
            x0, y0 = x0 * width / 512, y0 * height / 512
            # append to landmarks list
            landmarks_list.append([x0, y0])

        # 返回出去
        return landmarks_list


if __name__ == "__main__":
    algorithm = Cldetection_alg_2023()
    algorithm.process()
    # 问：这里没有实现 process() 函数，怎么可以进行调用呢？
    # 答：因为这是 Cldetection_alg_2023 继承了 DetectionAlgorithm，父类函数，子类也就有了，然后进行执行，背后会自动调用相关函数

    # 问：调用 process() 函数，背后执行了什么操作呢？
    # 答：我们可通过跳转到源码可以看到，process() 函数，这里是源码显示：
    #    def process(self):
    #        self.load()
    #        self.validate()
    #        self.process_cases()
    #        self.save()
    #    我们可以看到背后执行了这四个函数，而对应在 process_cases() 函数中又进行了调用 process_case() 函数：
    #    def process_cases(self, file_loader_key: Optional[str] = None):
    #        if file_loader_key is None:
    #            file_loader_key = self._index_key
    #        self._case_results = []
    #        for idx, case in self._cases[file_loader_key].iterrows():
    #            self._case_results.append(self.process_case(idx=idx, case=case))
    #    因此，对应这我们挑战赛的内容，您仅需要在 process_case() 和 save() 函数中实现你想要的功能

    # 问：又说仅需要 process_case() 和 save() 进行实现，为什么又跳出一个 predict() 函数呢？
    # 答：predict() 函数是父类 DetectionAlgorithm 要求实现的，负责预测每一个case的结果，不然会提示 NotImplementedError 错误



    

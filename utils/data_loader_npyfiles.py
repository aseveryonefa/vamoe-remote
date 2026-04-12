# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
'''Module providing dataset functions'''
import os
import abc
import datetime
import random
import json

# import h5py
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


# https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2020MS002203
# PRESSURE_LEVELS_WEATHERBENCH_13 = (
#     50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)

# The list of all possible atmospheric variables. Taken from:
# https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation#ERA5:datadocumentation-Table9
ALL_ATMOSPHERIC_VARS = (
    "potential_vorticity",
    "specific_rain_water_content",
    "specific_snow_water_content",
    "geopotential",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
    "vertical_velocity",
    "vorticity",
    "divergence",
    "relative_humidity",
    "ozone_mass_mixing_ratio",
    "specific_cloud_liquid_water_content",
    "specific_cloud_ice_water_content",
    "fraction_of_cloud_cover",
)

TARGET_SURFACE_VARS = (
    "2m_temperature",
    "mean_sea_level_pressure",
    "10m_v_component_of_wind",
    "10m_u_component_of_wind",
    "total_precipitation_6hr",
)
TARGET_SURFACE_NO_PRECIP_VARS = (
    "2m_temperature",
    "mean_sea_level_pressure",
    "10m_v_component_of_wind",
    "10m_u_component_of_wind",
)
TARGET_ATMOSPHERIC_VARS = (
    "temperature",
    "geopotential",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
    "specific_humidity",
)
TARGET_ATMOSPHERIC_NO_W_VARS = (
    "temperature",
    "geopotential",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
)

# FEATURE_DICT = {'Z500': (5, 0), 'T850': (2, 4), 'U10': (-4, 0), 'T2M': (-5, 0)}
FEATURE_DICT = {'Z500': (5, 0), 'Q500': (5, 1), 'U500': (5, 2), 'V500': (5, 3), 'T500': (5, 4), 'T850': (2, 4), 'U10': (-4, 0), 'V10': (-3, 0)}

SIZE_DICT = {0.25: [721, 1440], 0.5: [360, 720], 1.4: [128, 256]}

# higher+ surface z*13+q*13+u*13+v*13+t*13
# surface_features = []    # 'tp1h'

#修改！！！！！！！！！！！
# surface_features = ['t2m', 'u10', 'v10', 'msl', 'sp']    # 'tp1h' ['t2m', 'u10', 'v10', 'msl', 'sp'] 
surface_features = []



# surface_features = ['t2m', 'u10', 'v10']    # 'tp1h'
# surface_features = ['v10', 'u10', 't2m', 'msl', 'sp', 'u100', 'v100', 'tisr', 'tcc', 'tp', 'tp1h', 'tp2h', 'tp3h', 'tp4h', 'tp5h', 'tp6h', 'ssr', 'ssr1h', 'ssr6h', 'mwd', 'mwp', 'swh']
# surface_features = ['t2m', 'u10', 'v10', 'msl', 'sp', 'mwp', 'swh']


# higher_features = ['z', 'q', 'u', 'v', 't']
higher_features = ['z', 'q', 'u', 'v', 't']

# pressure_level = [1000.0, 850.0, 700.0, 600.0, 500.0, 400.0, 300.0, 200.0, 150.0, 100.0, 50.0]
pressure_level = [1000.0, 925.0, 850.0, 700.0, 600.0, 500.0, 400.0, 300.0, 250.0, 200.0, 150.0, 100.0, 50.0]

total_levels= [1000.,  975.,  950.,  925.,  900.,  875.,  850.,  825.,  800.,
                775.,  750.,  700.,  650.,  600.,  550.,  500.,  450.,  400.,
                350.,  300.,  250.,  225.,  200.,  175.,  150.,  125.,  100.,
                70.,   50.,   30.,   20.,   10.,    7.,    5.,    3.,    2.,  1.]


#修改！！！！！！！！！！！
#mapping_dict = [total_levels.index(i) for i in pressure_level]
#修改mapping_dict的索引
mapping_dict = list(range(len(pressure_level)))



def get_datapath_from_date(start_date, idx):
    t0 = start_date
    t = t0 + datetime.timedelta(hours=idx)
    year = t.year
    month = t.month
    day = t.day
    hour = t.hour
    date_file_name = f'{year}/{year}-{str(month).zfill(2)}-{str(day).zfill(2)}/{str(hour).zfill(2)}-00-00-'
    return date_file_name, f'{year}-{str(month).zfill(2)}-{str(day).zfill(2)}-{str(hour).zfill(2)}'
    # static_file_name = f'{year}/{year}.npy'
    # return date_file_name, static_file_name


# class Data:
#     """
#     This class is the base class of Dataset.

#     Args:
#         root_dir (str, optional): The root dir of input data. Default: ".".

#     Raises:
#         TypeError: If the type of train_dir is not str.
#         TypeError: If the type of test_dir is not str.

#     Supported Platforms:
#         ``Ascend`` ``GPU``
#     """

#     def __init__(self, root_dir="."):
#         self.train_dir = os.path.join(root_dir, "train")
#         self.valid_dir = os.path.join(root_dir, 'valid')
#         self.test_dir = os.path.join(root_dir, "test")

#     @abc.abstractmethod
#     def __getitem__(self, index):
#         """Defines behavior for when an item is accessed. Return the corresponding element for given index."""
#         raise NotImplementedError(
#             "{}.__getitem__ not implemented".format(self.dataset_type))

#     @abc.abstractmethod
#     def __len__(self):
#         """Return length of dataset"""
#         raise NotImplementedError(
#             "{}.__len__ not implemented".format(self.dataset_type))


class Era5Data(Dataset):
# class Era5Data(Data):
    """
    This class is used to process ERA5 re-analyze data, and is used to generate the dataset generator supported by
    MindSpore. This class inherits the Data class.

    Args:
        data_params (dict): dataset-related configuration of the model.
        run_mode (str, optional): whether the dataset is used for training, evaluation or testing. Supports [“train”,
            “test”, “valid”]. Default: 'train'.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> from mindearth.data import Era5Data
        >>> data_params = {
        'name': 'era5',
        'root_dir': './dataset',
        'feature_dims': 69,
        't_in': 1,
        't_out_train': 1,
        't_out_valid': 20,
        't_out_test': 20,
        'valid_interval': 1,
        'test_interval': 1,
        'train_interval': 1,
        'pred_lead_time': 6,
        'data_frequency': 6,
        'train_period': [2015, 2015],
        'valid_period': [2016, 2016],
        'test_period': [2017, 2017],
        'patch': True,
        'patch_size': 8,
        'batch_size': 8,
        'num_workers': 1,
        'grid_resolution': 1.4,
        'h_size': 128,
        'w_size': 256
        ... }
        >>> dataset_generator = Era5Data(data_params)
    """

    ## TODO: example should include all possible infos:
    #  data_frequency, patch/patch_size
    def __init__(self,
                 data_params,
                 run_mode='train'):

        super(Era5Data, self).__init__()
        none_type = type(None)
        self.root_dir = data_params['root_dir']
        self.root_surface_dir = os.path.join(self.root_dir, "single")
        self.climate_dir = data_params['root_dir'] + 'climate_mean_day_128x256/1993-2016/'
        self.climate_surface_dir = data_params['root_dir'] + 'single/climate_mean_day_128x256/1993-2016/'

        # self.train_dir = os.path.join(root_dir, "train")
        # self.valid_dir = os.path.join(root_dir, 'valid')
        # self.test_dir = os.path.join(root_dir, "test")

        # self.train_surface_dir = os.path.join(root_dir, "train_surface")
        # self.valid_surface_dir = os.path.join(root_dir, "valid_surface")
        # self.test_surface_dir = os.path.join(root_dir, "test_surface")

        # self.train_static = os.path.join(root_dir, "train_static")
        # self.valid_static = os.path.join(root_dir, "valid_static")
        # self.test_static = os.path.join(root_dir, "test_static")
        # self.train_surface_static = os.path.join(root_dir, "train_surface_static")
        # self.valid_surface_static = os.path.join(root_dir, "valid_surface_static")
        # self.test_surface_static = os.path.join(root_dir, "test_surface_static")

        self.statistic_dir = os.path.join(self.root_dir, "statistic")

        self._get_statistic()

        self.run_mode = run_mode
        self.t_in = data_params['t_in']
        self.h_size = data_params['h_size']
        self.w_size = data_params['w_size']
        self.data_frequency = data_params['data_frequency']
        self.valid_interval = data_params['valid_interval'] * self.data_frequency
        self.test_interval = data_params['test_interval'] * self.data_frequency
        self.train_interval = data_params['train_interval'] * self.data_frequency
        self.pred_lead_time = data_params['pred_lead_time']
        self.train_period = data_params['train_period']
        self.valid_period = data_params['valid_period']
        self.test_period = data_params['test_period']
        self.feature_dims = data_params['feature_dims']
        self.output_dims = data_params['feature_dims']
        self.surface_feature_size = data_params['surface_feature_size']
        self.level_feature_size = (self.feature_dims -
                                   self.surface_feature_size) // data_params['pressure_level_num']
        self.patch = data_params['patch']
        if self.patch:
            self.patch_size = data_params['patch_size']

        # ============================================================
        # [旧代码 - 原始 run_mode 分支，已完整注释保留]
        # 问题说明：原始代码对三种模式各自使用独立的年份 Period 来确定 start_date。
        #           当三个 Period 完全相同时（即数据集重叠配置），三种模式会从
        #           同一个时间起点读取数据，导致训练集与验证集数据完全相同，
        #           无法真实评估模型的泛化能力（Data Leakage 数据泄露）。
        #
        # if run_mode == 'train':
        #     self.t_out = data_params['t_out_train']
        #     # self.path = self.train_dir
        #     # self.surface_path = self.train_surface_dir
        #     # self.static_path = self.train_static
        #     # self.static_surface_path = self.train_surface_static
        #     self.interval = self.train_interval
        #     self.start_date = datetime.datetime(self.train_period[0], 1, 1, 0, 0, 0)
        #
        # elif run_mode == 'valid':
        #     self.t_out = data_params['t_out_valid']
        #     # self.path = self.valid_dir
        #     # self.surface_path = self.valid_surface_dir
        #     # self.static_path = self.valid_static
        #     # self.static_surface_path = self.valid_surface_static
        #     self.interval = self.valid_interval
        #     self.start_date = datetime.datetime(self.valid_period[0], 1, 1, 0, 0, 0)
        #
        # else:
        #     self.t_out = data_params['t_out_test']
        #     # self.path = self.test_dir
        #     # self.surface_path = self.test_surface_dir
        #     # self.static_path = self.test_static
        #     # self.static_surface_path = self.test_surface_static
        #     self.interval = self.test_interval
        #     self.start_date = datetime.datetime(self.test_period[0], 1, 1, 0, 0, 0)
        # ============================================================

        # ============================================================
        # [新代码] run_mode 分支 - 支持通过 config 指定 train_split_ratio 进行时序切分
        # 逻辑说明：
        #   Step 1. 先按原逻辑确定各 mode 的 interval、t_out 和 start_date 基准值。
        #   Step 2. 读取 config 中的 train_split_ratio 参数（如 0.8）。
        #   Step 3. 若 train_split_ratio 不为 null，则按比例将数据切分：
        #             - 前 ratio 部分 → 训练集（train）
        #             - 后 (1-ratio) 部分 → 验证集（valid）和测试集（test），两者相同
        #   Step 4. 通过修改 start_date（时间锚点）和 available_hours（可用小时数）
        #           来实现逻辑切分，__getitem__ 无需任何改动。
        # ============================================================
        if run_mode == 'train':
            self.t_out = data_params['t_out_train']
            # self.path = self.train_dir
            # self.surface_path = self.train_surface_dir
            # self.static_path = self.train_static
            # self.static_surface_path = self.train_surface_static
            self.interval = self.train_interval
            self.start_date = datetime.datetime(self.train_period[0], 1, 1, 0, 0, 0)

        elif run_mode == 'valid':
            self.t_out = data_params['t_out_valid']
            # self.path = self.valid_dir
            # self.surface_path = self.valid_surface_dir
            # self.static_path = self.valid_static
            # self.static_surface_path = self.valid_surface_static
            self.interval = self.valid_interval
            self.start_date = datetime.datetime(self.valid_period[0], 1, 1, 0, 0, 0)

        else:  # test
            self.t_out = data_params['t_out_test']
            # self.path = self.test_dir
            # self.surface_path = self.test_surface_dir
            # self.static_path = self.test_static
            # self.static_surface_path = self.test_surface_static
            self.interval = self.test_interval
            self.start_date = datetime.datetime(self.test_period[0], 1, 1, 0, 0, 0)

        # ============================================================
        # [新增] 读取 config 中的 train_split_ratio，执行时序切分
        # 仅当 train_split_ratio 不为 None/null 时才激活此逻辑，
        # 设为 null 则完全退化为旧代码行为，向后兼容。
        # ============================================================
        self.train_split_ratio = data_params['train_split_ratio'] if 'train_split_ratio' in data_params else None
        self.available_hours = None  # 默认不限制，由旧逻辑计算全量

        if self.train_split_ratio is not None:
            # 使用 train_period 的范围来计算总可用小时数
            total_hours = self._get_file_count(self.root_dir, self.train_period)
            
            # [关键修正]：按比例计算点，但必须向下取整到 24 (一天) 的倍数！
            # 因为硬盘上的文件只在 00, 06, 12, 18 生成。
            # 比如 8784 取 0.8 是 7027，偏移7027小时后时间的 hour 变成 19:00，导致文件找不着报错！
            # // 24 * 24 保证时间锚点永远卡在 00:00:00
            split_point_hours = int(total_hours * self.train_split_ratio)
            split_point_hours = (split_point_hours // 24) * 24

            if run_mode == 'train':
                # 训练集：只能访问前 ratio 比例的数据，start_date 保持不变（从头开始）
                self.available_hours = split_point_hours
            else:
                # 验证集 / 测试集：访问后 (1-ratio) 比例的数据
                self.available_hours = total_hours - split_point_hours
                # [防御性修正]：强制以 train_period[0] 作为原点进行偏移，防止用户在 yaml 中弄错 valid_period
                base_start_date = datetime.datetime(self.train_period[0], 1, 1, 0, 0, 0)
                self.start_date = base_start_date + datetime.timedelta(hours=split_point_hours)
        # ============================================================

    # ============================================================
    # [旧代码 - 原始 __len__ 方法，已完整注释保留]
    # 问题说明：原方法每次调用都重新扫描对应 Period 的所有年份文件来统计总量。
    #           在数据集重叠或按比例切分的场景下，三种模式均会统计相同的全量
    #           文件数，导致 __len__ 返回的长度不能反映实际切分后的数据范围。
    #
    # def __len__(self):
    #     if self.run_mode == 'train':
    #         self.train_len = self._get_file_count(self.root_dir, self.train_period)
    #         length = (self.train_len * self.data_frequency -
    #                   (self.t_out + self.t_in) * self.pred_lead_time) // self.train_interval
    #
    #     elif self.run_mode == 'valid':
    #         self.valid_len = self._get_file_count(self.root_dir, self.valid_period)
    #         length = (self.valid_len * self.data_frequency -
    #                   (self.t_out + self.t_in) * self.pred_lead_time) // self.valid_interval
    #
    #     else:
    #         self.test_len = self._get_file_count(self.root_dir, self.test_period)
    #         length = (self.test_len * self.data_frequency -
    #                   (self.t_out + self.t_in) * self.pred_lead_time) // self.test_interval
    #     return length
    # ============================================================

    def __len__(self):
        # ============================================================
        # [新代码] 支持按比例切分的 __len__ 方法
        # 若 available_hours 不为 None（说明 train_split_ratio 已激活），
        # 则直接使用切分后的小时数计算长度，否则回退到原始全量统计逻辑。
        # ============================================================
        if self.available_hours is not None:
            # 已激活切分模式：使用预先计算好的可用小时数（精确对应各 mode 的数据范围）
            l_hours = self.available_hours
        else:
            # 未激活切分模式（train_split_ratio=null）：回退到原始逻辑，扫描全量文件数
            if self.run_mode == 'train':
                l_hours = self._get_file_count(self.root_dir, self.train_period)
            elif self.run_mode == 'valid':
                l_hours = self._get_file_count(self.root_dir, self.valid_period)
            else:
                l_hours = self._get_file_count(self.root_dir, self.test_period)

        # 计算有效样本数，算式与旧代码完全一致
        # (可用小时数 * 数据频率 - 输入输出步长的时间占用) / 采样间隔
        length = (l_hours * self.data_frequency -
                  (self.t_out + self.t_in) * self.pred_lead_time) // self.interval
        return max(0, length)

    def __getitem__(self, idx):
        inputs_lst = []
        inputs_surface_lst = []
        label_lst = []
        label_surface_lst = []
        idx = idx * self.interval
        # if self.run_mode != 'train':
        #     self.climate_lst = []
        #     self.climate_surface_lst = []

        for t in range(self.t_in):
            cur_input_data_idx = idx + t * self.pred_lead_time
            half_path, date_time = get_datapath_from_date(self.start_date, cur_input_data_idx)
            x, x_surface = self._get_weather_data(half_path)
            # print('inputs data path:', half_path, 'date_time: ', date_time)

            # x = np.load(os.path.join(self.path, input_date))[:, :, :self.h_size].astype(np.float32)
            # x_surface = np.load(os.path.join(self.surface_path,
            #                                  input_date))[:, :self.h_size].astype(np.float32)
            # x_static = np.load(os.path.join(self.static_path, year_name)).astype(np.float32)
            # x_surface_static = np.load(os.path.join(self.static_surface_path,
            #                                         year_name)).astype(np.float32)
            # x = self._get_origin_data(x, x_static)
            # x_surface = self._get_origin_data(x_surface, x_surface_static)
            x, x_surface = self._normalize(x, x_surface)
            inputs_lst.append(x)
            inputs_surface_lst.append(x_surface)

        for t in range(self.t_out):
            cur_label_data_idx = idx + (self.t_in + t) * self.pred_lead_time
            label_path, date_time = get_datapath_from_date(self.start_date, cur_label_data_idx)
            label, label_surface = self._get_weather_data(label_path)
            # print('label data path: ', label_path, 'date_time: ', date_time)

            # label = np.load(os.path.join(self.path, label_date))[:, :, :self.h_size].astype(np.float32)
            # label_surface = np.load(os.path.join(self.surface_path,
            #                                      label_date))[:, :self.h_size].astype(np.float32)
            # label_static = np.load(os.path.join(self.static_path,
            #                                     year_name)).astype(np.float32)
            # label_surface_static = np.load(os.path.join(self.static_surface_path,
            #                                             year_name)).astype(np.float32)
            # label = self._get_origin_data(label, label_static)
            # label_surface = self._get_origin_data(label_surface, label_surface_static)
            label, label_surface = self._normalize(label, label_surface)
            label_lst.append(label)
            label_surface_lst.append(label_surface)

            # if self.run_mode != 'train':
            #     if '02-29' in date_time:  date_time = date_time.replace('02-29', '02-28')
            #     climate_features, climate_surface_features = self._get_climate_data(date_time)
            #     self.climate_lst.append(climate_features)
            #     self.climate_surface_lst.append(climate_surface_features)



        x = np.concatenate(inputs_lst, axis=0).astype(np.float32)    # [t,h,w,level,feature]
        label = np.concatenate(label_lst, axis=0).astype(np.float32)
        #使用stack增加维度，将4维变为5维！！！！！！！！！！！！！！！！
        # x = np.stack(inputs_lst, axis=0).astype(np.float32) 
        # label = np.stack(label_lst, axis=0).astype(np.float32)


        if surface_features != []:
            x_surface = np.concatenate(inputs_surface_lst, axis=0).astype(np.float32)
            label_surface = np.concatenate(label_surface_lst, axis=0).astype(np.float32)

        # if self.run_mode != 'train':
        #     self.climate_features = np.stack(self.climate_lst, axis=0).astype(np.float32)
        #     self.climate_surface_features = np.stack(self.climate_surface_lst, axis=0).astype(np.float32)
        #     self.climate_features = self.climate_features.transpose((0, 4, 3, 1, 2)).reshape(self.t_out, len(pressure_level)*len(higher_features) , self.h_size, self.w_size)
        #     if surface_features != [] :
        #         self.climate_surface_features = self.climate_surface_features.transpose((0, 3, 1, 2)).reshape(self.t_out, len(surface_features), self.h_size, self.w_size)
        #         self.climate = np.squeeze(np.concatenate([self.climate_features, self.climate_surface_features], axis=1))
        #     else:
        #         self.climate_surface_features = None
        #         self.climate = np.squeeze(self.climate_features)

        return self._process_fn(x, x_surface, label, label_surface)

    # @staticmethod
    # def _get_origin_data(x, static):
    #     data = x * static[..., 0] + static[..., 1]
    #     return data

    def _get_weather_data(self, half_path):
        # get pressure level data
        all_level_paths = [[i+'-'+str(j)+'.npy' for j in pressure_level] for i in higher_features]
        all_level_features = []
        for i, _ in enumerate(higher_features):
            level_feature = [np.load(self.root_dir + '/' + half_path + subpath) for subpath in all_level_paths[i]]
            level_feature = np.stack(level_feature, axis=-1)    # [h, w, level]
            all_level_features.append(level_feature)
        all_level_features = np.stack(all_level_features, axis=-1)    # [h, w, level, feature]

        all_surface_features = [np.load(self.root_surface_dir + '/' + half_path + i + '.npy').reshape([1, self.h_size, self.w_size]) for i in surface_features]

        if surface_features == []:
            all_surface_features = None
        else:
            all_surface_features = np.stack(all_surface_features, axis=-1)     # [h, w, feature]

        # print('level_feature:', all_level_features.shape, 'surface feature:', all_surface_features.shape)

        return all_level_features, all_surface_features

    # def _get_climate_data(self, date_time):
    #     date_list = date_time.split('-')[1:-1]
    #     path_name = '-'.join(date_list)

    #     all_level_paths = [[i+'-'+str(j)+'.npy' for j in pressure_level] for i in higher_features]
    #     all_level_climate_features = []
    #     for i, _ in enumerate(higher_features):
    #         level_climate_feature = [np.load(self.climate_dir + path_name + '/' + subpath) for subpath in all_level_paths[i]]
    #         level_climate_feature = np.stack(level_climate_feature, axis=-1)    # [level, h, w]
    #         all_level_climate_features.append(level_climate_feature)
    #     all_level_climate_features = np.stack(all_level_climate_features, axis=-1)    # [level, h, w, feature]

    #     all_surface_climate_features = [np.load(self.climate_surface_dir + path_name + '/' + i + '.npy').reshape([self.h_size, self.w_size]) for i in surface_features]

    #     if surface_features != []:
    #         all_surface_climate_features = np.stack(all_surface_climate_features, axis=-1)     # [h, w, feature]
    #     else:
    #         all_surface_climate_features = None

    #     # print('level_climate:', all_level_climate_features.shape, 'surface climate:', all_surface_climate_features.shape)

    #     return all_level_climate_features, all_surface_climate_features

    @staticmethod
    def _get_file_count(path, period):
        count = 0
        for i in range(period[0], period[1]+1, 1):
            subpath = os.path.join(path, str(i))
            if os.path.exists(subpath):
                tmp_lst = os.listdir(subpath)
                count += 24*len(tmp_lst)

        # file_lst = os.listdir(path)
        # count = 0
        # for f in file_lst:
        #     if period[0] <= int(f) <= period[1]:
        #         tmp_lst = os.listdir(os.path.join(path, f))
        #         count += len(tmp_lst)
        return count

    def _get_statistic(self):
        self.surface_path = os.path.join(self.statistic_dir, 'mean_std_single.json')
        fs = open(self.surface_path, mode='r')
        self.mean_std_surface = json.load(fs)
        fs.close()

        self.level_path = os.path.join(self.statistic_dir, 'mean_std.json')
        fl = open(self.level_path, mode='r')
        self.mean_std_level = json.load(fl)
        fl.close()

        self.all_mean_surface = self.mean_std_surface['mean']
        self.all_mean_level = self.mean_std_level['mean']
        self.all_std_surface = self.mean_std_surface['std']
        self.all_std_level = self.mean_std_level['std']

        # print('mean surface:', self.all_mean_surface)
        # print('std surface:', self.all_std_surface)
        # print(self.all_mean_surface.keys())
        # print(self.all_std_surface.keys())
        
        self.mean_surface = [self.all_mean_surface[i] for i in surface_features]
        self.std_surface = [self.all_std_surface[i] for i in surface_features]
        self.mean_surface = np.array(self.mean_surface)
        self.std_surface = np.array(self.std_surface)

        self.mean_pressure_level = [[self.all_mean_level[i][str(pressure_level[j])] for j in mapping_dict] for i in higher_features]
        self.std_pressure_level = [[self.all_std_level[i][str(pressure_level[j])] for j in mapping_dict] for i in higher_features]
        self.mean_pressure_level = np.array(self.mean_pressure_level).transpose((1,0))
        self.std_pressure_level = np.array(self.std_pressure_level).transpose((1,0))

        # self.mean_pressure_level = np.squeeze(self.mean_pressure_level)
        # self.std_pressure_level = np.squeeze(self.std_pressure_level)


        # print('pressure level:', self.mean_pressure_level.shape, self.std_pressure_level.shape)
        # print('surface level:', self.mean_surface.shape, self.std_surface.shape)

        # self.mean_pressure_level = np.load(os.path.join(self.statistic_dir, 'mean.npy'))
        # self.std_pressure_level = np.load(os.path.join(self.statistic_dir, 'std.npy'))
        # self.mean_surface = np.load(os.path.join(self.statistic_dir, 'mean_s.npy'))
        # self.std_surface = np.load(os.path.join(self.statistic_dir, 'std_s.npy'))

    def _normalize(self, x, x_surface):
        # print('normalized before: ', x.shape )
        x = (x - self.mean_pressure_level) / self.std_pressure_level
        if x_surface is not None:
            x_surface = (x_surface - self.mean_surface) / self.std_surface
        # print('normalized after: ', x.shape )
        return x, x_surface

    def _process_fn(self, x, x_surface, label, label_surface):
        '''process_fn'''
        # print('x shape:', x.shape)
        # print('label shape:', label.shape)
        #数据维度不匹配，增加时间步为1！！！！！！！！！！！！！！！！！
        if len(x.shape) == 4:
                # 如果只有4维，添加时间步维度
                x = x[np.newaxis, ...]
                label = label[np.newaxis, ...]
                # print(f"Debug _process_fn: after fix, x shape = {x.shape}")
        if len(x.shape) == 5:
            t_in, h, w, level_size, feature_size = x.shape
            # print(f"Debug _process_fn: t_in={t_in}, h={h}, w={w}, level_size={level_size}, feature_size={feature_size}")
        else:
            raise ValueError(f"Unexpected x shape: {x.shape}, expected 5 dimensions")
    

        _, _, _, level_size, feature_size = x.shape

        # if self.patch:
        self.h_size = self.h_size - self.h_size % self.patch_size
        self.w_size = self.w_size - self.w_size % self.patch_size

        x = x[:, :self.h_size, :self.w_size, ...]
        label = label[:, :self.h_size, :self.w_size, ...]
        x = x.transpose((0, 4, 3, 1, 2)).reshape(self.t_in, level_size * feature_size, self.h_size, self.w_size)
        label = label.transpose((0, 4, 3, 1, 2)).reshape(self.t_out, level_size * feature_size,
                                                            self.h_size, self.w_size)
        
                                                            
        if surface_features != []:
            surface_size = x_surface.shape[-1]
            x_surface = x_surface[:, :self.h_size, :self.w_size, ...]
            label_surface = label_surface[:, :self.h_size, :self.w_size, ...]
            x_surface = x_surface.transpose((0, 3, 1, 2)).reshape(self.t_in, surface_size, self.h_size, self.w_size)
            label_surface = label_surface.transpose((0, 3, 1, 2)).reshape(self.t_out, surface_size,
                                                                        self.h_size, self.w_size)

        if surface_features != []:
            inputs = np.concatenate([x, x_surface], axis=1)
            labels = np.concatenate([label, label_surface], axis=1)
        else:
            inputs = x
            labels = label

        # else:
        #     x = x.transpose((0, 4, 3, 1, 2)).reshape(self.t_in, self.h_size * self.w_size,
        #                                              level_size * feature_size)
        #     x_surface = x_surface.reshape(self.t_in, self.h_size * self.w_size, surface_size)
        #     label = label.transpose((0, 4, 3, 1, 2)).reshape(self.t_out, self.h_size * self.w_size,
        #                                                      level_size * feature_size)
        #     label_surface = label_surface.reshape(self.t_out, self.h_size * self.w_size, surface_size)
        #     inputs = np.concatenate([x, x_surface], axis=-1)
        #     labels = np.concatenate([label, label_surface], axis=-1)
        #     inputs = inputs.transpose((1, 0, 2)).reshape(self.h_size * self.w_size,
        #                                                  self.t_in * (level_size * feature_size + surface_size))
        # if self.run_mode != 'train':
            # print(self.climate_features.shape, self.climate_surface_features.shape)])

        # if self.patch:
        #     labels = self._patch(labels, (self.h_size, self.w_size), self.patch_size,
        #                          level_size * feature_size + surface_size)
        inputs = np.squeeze(inputs)
        labels = np.squeeze(labels)

        # if self.run_mode == 'train':
        return inputs, labels
        # else:
        #     return inputs, labels, self.climate

    def _patch(self, x, img_size, patch_size, output_dims):
        """ Partition the data into patches. """
        if self.run_mode == 'train':
            x = x.transpose(0, 2, 3, 1)
            h, w = img_size[0] // patch_size, img_size[1] // patch_size
            x = x.reshape(x.shape[0], h, patch_size, w, patch_size, output_dims)
            x = x.transpose(0, 1, 3, 2, 4, 5)
            x = np.squeeze(x.reshape(x.shape[0], h * w, patch_size * patch_size * output_dims))
        else:
            x = x.transpose(1, 0, 2, 3)
        return x


def get_data_loader_npy(params, distributed, run_mode):
    dataset = Era5Data(params, run_mode)
    sampler = DistributedSampler(dataset, shuffle=False) if distributed else None
    
    dataloader = DataLoader(dataset,
                            batch_size=int(params['batch_size']),
                            num_workers=params['num_data_workers'],
                            shuffle=False, #(sampler is None),
                            sampler=sampler if run_mode=='train' else None,
                            drop_last=True,
                            # pin_memory=torch.cuda.is_available()
                            pin_memory=False
                            )

    if run_mode=='train':
        return dataloader, dataset, sampler
    else:
        return dataloader, dataset


if __name__ == '__main__':
    data_params = {
            'name': 'era5',
            'root_dir': './data/era5_np128x256/',
            'feature_dims': 49,   # 68, 69, 
            't_in': 1,
            't_out_train': 1,
            't_out_valid': 1,
            't_out_test': 20,
            'valid_interval': 12,
            'test_interval': 12,
            'train_interval': 6,
            'pred_lead_time': 6,
            'data_frequency': 1,
            'train_period': [2000, 2020],   # [1979, 2020]
            'valid_period': [2021, 2021],
            'test_period': [2005, 2005],
            'patch': True,
            'patch_size': 4,
            'batch_size': 1,
            'num_data_workers': 20,
            'grid_resolution': 1.4,
            'h_size': 128,
            'w_size': 256,
            'surface_feature_size': 10,
            'pressure_level_num': 13
        }
        
    from tqdm import tqdm

    dataloader, dataset, _ = get_data_loader_npy(data_params, False, 'train')

    for idx, data in tqdm(enumerate(dataloader)):
    # for idx, data in enumerate(dataloader):
        # inputs, targets, climate = data
        # print(inputs.shape, targets.shape, climate.shape)

        inputs, targets = data
        # print(inputs.shape, targets.shape)
        # if idx == 2: exit()




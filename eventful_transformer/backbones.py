import torch.nn as nn

from eventful_transformer import blocks
from eventful_transformer.base import ExtendedModule
from eventful_transformer.utils import PositionEncoding

import torch


class ViTBackbone(ExtendedModule):
    """
    Common backbone for vision Transformers.
    """

    def __init__(
        self,
        block_config,
        depth,
        position_encoding_size,
        input_size,
        block_class="Block",
        has_class_token=False,
        window_indices=(),
        windowed_class=None,
        windowed_overrides=None,
    ):
        """
        :param block_config: A dict containing kwargs for the
        block_class constructor
        :param depth: The number of blocks to use
        :param position_encoding_size: The size (in tokens) assumed for
        position encodings
        :param input_size: The expected size of the inputs in tokens
        :param block_class: The specific Block class to use (see
        blocks.py for options)
        :param has_class_token: Whether to add an extra class token
        :param window_indices: Block indices that should use windowed
        attention
        :param windowed_class: The specific Block class to use with
        windowed attention (if None, fall back to block_class)
        :param windowed_overrides: A dict containing kwargs overrides
        for windowed_class
        """
        super().__init__()
        self.position_encoding = PositionEncoding(
            block_config["dim"], position_encoding_size, input_size, has_class_token
        )
        self.blocks = nn.Sequential()
        for i in range(depth):
            block_class_i = block_class
            block_config_i = block_config.copy()
            if i in window_indices:
                if windowed_class is not None:
                    block_class_i = windowed_class
                if windowed_overrides is not None:
                    block_config_i |= windowed_overrides
            else:
                block_config_i["window_size"] = None
            self.blocks.append(
                getattr(blocks, block_class_i)(input_size=input_size, **block_config_i)
            )
        
        self.mvs = None
        self.residuals = None
        self.recompute = 0
        self.data_num = 0
        self.exchange_num = 0
        self.reuse_flag = None
        self.IntersectCoords = None
        self.curIntersectCoords = None
        self.force_index = None
        self.exchange_index = None

    def get_force_index(self, mvs, residuals):
        mask = torch.ones(mvs.shape[0], dtype=torch.int)
        condition = (mvs == 100000) & (residuals <= 8)
        mask[condition] = 0
        force_index = torch.nonzero(mask==1).squeeze(-1)
        return force_index.unsqueeze(0).to(mvs.device)

    def get_exchange_index(self, mvs, residuals):
        combined_condition = (mvs != 0) & (mvs!=100000) & (residuals<=10)
        index = torch.nonzero(combined_condition).squeeze()
        if index.numel() == 0:
            return None
        if index.numel() == 1:
            return index.unsqueeze(0)
        coords_cur = self.curIntersectCoords[index]
        coords_exchange = self.IntersectCoords[index+mvs[index]]
        ious, change_IntersectCoords = self.calculate_iou_and_update_coords(coords_cur, coords_exchange)
        condition = ious > 0.3
        index_refresh = index[~condition]
        index = index[condition]
        if index_refresh.numel() != 0:
            self.IntersectCoords[index_refresh] = torch.tensor([0,15,15,0], dtype=self.IntersectCoords.dtype).to(index.device)
        if index.numel() == 0:
            return None
        if index.numel() == 1:
            return index.unsqueeze(0)
        self.IntersectCoords[index] = change_IntersectCoords[condition]
        return index

    def calculate_iou_and_update_coords(self, coords_cur, coords_exchange):
        """
        Calculate the intersection area of ​​the two sets of boxes and update coords_cur to the coordinates of the upper left and lower right corners of the intersection
        :param coords_cur: Coordinate tensor of the first set of boxes, shape [n, 4]
        :param coords_exchange: Coordinate tensor of the second set of boxes, shape [n, 4]
        :return: Intersection area tensor (shape [n, 1]) and updated coords_cur
        """
        x1 = torch.max(coords_cur[:, 0], coords_exchange[:, 0])
        y1 = torch.min(coords_cur[:, 1], coords_exchange[:, 1])
        x2 = torch.min(coords_cur[:, 2], coords_exchange[:, 2])
        y2 = torch.max(coords_cur[:, 3], coords_exchange[:, 3])

        intersection_width = torch.clamp(x2 - x1, min=0)
        intersection_height = torch.clamp(y1 - y2, min=0)

        intersection_area = intersection_width * intersection_height

        intersection_area = intersection_area

        valid = (x2 > x1) & (y2 > y1)
        updated_coords = torch.zeros_like(coords_cur)
        updated_coords[valid] = torch.stack([x1[valid], y1[valid], x2[valid], y2[valid]], dim=1)

        union_area = torch.full_like(intersection_area, 16 * 16)

        return intersection_area/union_area, updated_coords


    def forward(self, x):
        x = self.position_encoding(x)
        force_index = self.force_index
        exchange_index = self.exchange_index
        for i,block in enumerate(self.blocks):
            block.force_index = force_index
            block.mvs = self.mvs
            block.exchange_index = exchange_index
            x = block(x)
        if exchange_index is not None:
            self.exchange_num += exchange_index.shape[0]
        # x = self.blocks(x)
        if force_index is not None:
            self.recompute += force_index.shape[1]
        self.data_num += 1
        if self.data_num % 100 == 0:
            print("num of recompute", f"{self.recompute/self.data_num:.2f}", "---------num of exchange", f"{self.exchange_num/self.data_num:.2f}")
        return x

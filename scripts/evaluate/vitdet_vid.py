#!/usr/bin/env python3

# import sys
# sys.path.append('/home/wangqinyu/qyw/efficient-eventful')

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm

from datasets.vid import VIDResize, VID
from models.vitdet import ViTDet
from utils.config import initialize_run
from utils.evaluate import run_evaluations
from utils.misc import dict_to_device, squeeze_dict

import cv_reader
import torchvision.transforms.functional as func
from torchvision import transforms
import torch.nn.functional as F

class VID_Video_info():
    def __init__(self, video_path: str) -> None:
        self.video_info = cv_reader.read_video(video_path=video_path, with_residual=True)
        self.curIntersectCoords = None
        self.IntersectCoords = None
        self.model_size = 1024#672

    def get_idx_index(self, idx, frame):
        mv = torch.tensor(self.video_info[idx]['motion_vector'])[:,:,:2]
        res = torch.abs(torch.tensor(self.video_info[idx]['residual'], dtype=torch.float32)-128)
        mv, res, IntersectCoords = self.get_new_mv_res(mv, res)
        # res = torch.abs(res - 128*3)
        self.curIntersectCoords = IntersectCoords
        force_index, mask = self.get_force_index(mv, res)
        exchange_index = self.get_exchange_index(mv, res,mask)
        return force_index, exchange_index, mv

    def get_force_index(self, mv, res):
        mask = torch.ones(mv.shape[0], dtype=torch.int)
        flag = (mv == 100000)   #不变的
        num_nochange = mv.shape[0] - 2048
        if torch.sum(flag) == mv.shape[0]:
            return None, None
        elif torch.sum(flag) < num_nochange:
            remaining = num_nochange - torch.sum(flag)
            valid_indices = torch.nonzero(flag == 0, as_tuple=True)[0]
            valid_res = res[valid_indices]
            topk_res, topk_indices = torch.topk(valid_indices, k=remaining, largest=False)
            # 找到这些索引对应的原始索引
            original_indices = valid_indices[topk_indices]
            flag[original_indices] = 1
            # force_index = torch.nonzero(flag).squeeze(-1)
            # print("force_index.shape", torch.nonzero(flag).squeeze(-1).shape,"topk_indices.shape", topk_indices.shape)
        else:
            valid_indices = torch.nonzero(flag == 1, as_tuple=True)[0]
            valid_res = res[valid_indices]
            _, topk_indices = torch.topk(valid_res, k=num_nochange, largest=False)
            # 找到这些索引对应的原始索引
            original_indices = valid_indices[topk_indices]
            # print("topk_indices.shape",topk_indices.shape,res[flag==1].min(),res[flag==1].max(),res[flag==1].mean(),res[topk_indices].min(),res[topk_indices].max(),res[topk_indices].mean())
            flag[:] = 0
            flag[original_indices] = 1
        
        # condition = (mv == 100000) & (res <= 15)

        # condition = (mv == 100000) | (mv == 0)
        # condition = condition & (res <= 4)

        #按照res差的最大的来判断
        # _, force_index = torch.topk(res, 2048)

        # condition = res<=4
        
        # mask[condition] = 0
        # force_index = torch.nonzero(mask==1).squeeze(-1)
        force_index = torch.nonzero(flag==0).squeeze(-1)
        return force_index.unsqueeze(0), flag

    def get_exchange_index(self, mv, res,mask):
        return None
        # combined_condition = (mv != 0) & (mv!=100000) & (res<=5)
        combined_condition = (mask == 0) & (res<=5) & (mv!=100000)
        index = torch.nonzero(combined_condition).squeeze()
        if index.numel() == 0:
            return None
        if index.numel() == 1:
            return index.unsqueeze(0)
        coords_cur = self.curIntersectCoords[index]
        # print("self.IntersectCoords.shape", self.IntersectCoords.shape,"mv.shape", mv.shape,"index.shape", index.shape)
        coords_exchange = self.IntersectCoords[index+mv[index]]
        ious, change_IntersectCoords = self.calculate_iou_and_update_coords(coords_cur, coords_exchange) 
        # print("ious.mean=", ious.mean(),"change_IntersectCoords", change_IntersectCoords.shape,"index", index.shape)
        condition = ious > 0.3
        index_refresh = index[~condition]
        index = index[condition]
        if index_refresh.numel() != 0:
            self.IntersectCoords[index_refresh] = torch.tensor([0,15,15,0], dtype=self.IntersectCoords.dtype)
        if index.numel() == 0:
            return None
        if index.numel() == 1:
            return index.unsqueeze(0)
        self.IntersectCoords[index] = change_IntersectCoords[condition]
        return index

    def calculate_iou_and_update_coords(self, coords_cur, coords_exchange):
        """
        计算两组框的交集面积，并将 coords_cur 更新为交集的左上角和右下角坐标
        :param coords_cur: 第一组框的坐标张量，形状为 [n, 4]
        :param coords_exchange: 第二组框的坐标张量，形状为 [n, 4]
        :return: 交集面积张量（形状为 [n, 1]）和更新后的 coords_cur
        """
        # 计算交集的左上角和右下角坐标
        x1 = torch.max(coords_cur[:, 0], coords_exchange[:, 0])
        y1 = torch.min(coords_cur[:, 1], coords_exchange[:, 1])
        x2 = torch.min(coords_cur[:, 2], coords_exchange[:, 2])
        y2 = torch.max(coords_cur[:, 3], coords_exchange[:, 3])

        # 计算交集的宽度和高度
        intersection_width = torch.clamp(x2 - x1, min=0)
        intersection_height = torch.clamp(y1 - y2, min=0)

        # 计算交集面积
        intersection_area = intersection_width * intersection_height

        # 调整交集面积形状为 [n, 1]
        intersection_area = intersection_area

        # 确保交集的左上角坐标小于右下角坐标
        valid = (x2 > x1) & (y2 > y1)
        updated_coords = torch.zeros_like(coords_cur)
        updated_coords[valid] = torch.stack([x1[valid], y1[valid], x2[valid], y2[valid]], dim=1)

        # 假设并集面积为 16*16
        union_area = torch.full_like(intersection_area, 16 * 16)

        return intersection_area/union_area, updated_coords
    def get_new_mv_res(self, mv, res):
        padding_mv = torch.zeros((256-mv.shape[0], 256, 2), dtype=mv.dtype, device=mv.device)
        # padding_mv = torch.zeros((168-mv.shape[0], 168, 2), dtype=mv.dtype, device=mv.device)
        mv = torch.cat([mv, padding_mv], dim=0)
        #合并4*4-》16*16
        mv = mv[::4, ::4, :]
        mask = (mv[:,:,0] == 0) & (mv[:,:,1] == 0)
        # 从mv转化为mv_mb
        ori_mv = mv
        mv = torch.round(mv/16.0).int()
        relative_mv = ori_mv - mv*16
        mv = self.mv_flatten(mv,mask)
        # 交集的左上&右下坐标
        IntersectCoords = torch.zeros((ori_mv.shape[0], ori_mv.shape[1], 4), dtype=ori_mv.dtype, device=ori_mv.device)
        IntersectCoords[:, :, 0] = torch.clamp(relative_mv[:, :, 0], min=0, max=15)
        IntersectCoords[:, :, 1] = torch.clamp(15+relative_mv[:, :, 1], min=0, max=15)
        IntersectCoords[:, :, 2] = torch.clamp(15+relative_mv[:, :, 0], min=0, max=15)
        IntersectCoords[:, :, 3] = torch.clamp(relative_mv[:, :, 1], min=0, max=15)

        padding = torch.zeros((self.model_size-res.shape[0], res.shape[1], res.shape[2]), dtype=res.dtype, device=res.device)
        res = torch.cat([res, padding], dim=0).float()
        # res = torch.cat([res, padding], dim=0).float().sum(dim=2)
        # res = res.reshape(42, 16, 42, 16).mean(dim=(1, 3))
        res = res.reshape(res.shape[0]//16, 16, res.shape[1]//16, 16, 3).permute(0, 2, 1, 3, 4).sum(-1).mean(-1).mean(-1)
        res = res.flatten()
        return mv, res, IntersectCoords.flatten(0,1)

    def mv_flatten(self, mv, mask):
        h = mv.shape[0]
        w = mv.shape[1]
        i_grid, j_grid = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        i_grid = i_grid
        j_grid = j_grid

        mvs_0 = mv[:,:,0]
        mvs_1 = mv[:,:,1]
        mvs_tilde =  ((i_grid + mvs_1) * w + j_grid + mvs_0) - (i_grid * w + j_grid)
        mvs_tilde[mask] = 100000
        return mvs_tilde.flatten()

    def get_part_frame(self, frame, change_index):
        return frame


def evaluate_vitdet_metrics(device, model, data, config):
    model.counting()
    model.clear_counts()
    n_frames = 0
    outputs = []
    labels = []
    n_items = config.get("n_items", len(data))
    video_num=0
    for vid_id, vid_item in tqdm(zip(range(n_items), data), total=n_items, ncols=0):
        # if vid_id == 5:
        #     break
        video_num+=1
        vid_path = vid_item.get_video_path()
        video = VID_Video_info(vid_path)
        # video_info = cv_reader.read_video(video_path=vid_path, with_residual=True)
        vid_item = DataLoader(vid_item, batch_size=1)
        n_frames += len(vid_item)
        model.reset()
        model.backbone.is_key = True
        # model.backbone.IntersectCoords = torch.zeros((1764, 4), dtype=torch.int, device=device)
        # model.backbone.IntersectCoords[:] = torch.tensor([0,15,15,0]).to(device)
        video.IntersectCoords = torch.zeros((4096, 4), dtype=torch.int) #1764
        video.IntersectCoords[:] = torch.tensor([0,15,15,0])
        for idx,(frame, annotations) in enumerate(vid_item):
            if idx==0:
                model.image_buffer = frame
            force_index,exchange_index, mv = video.get_idx_index(idx, frame)
            force_index = force_index.to(device) if force_index is not None else None
            exchange_index = exchange_index.to(device) if exchange_index is not None else None
            model.backbone.mvs = mv.to(device)
            model.backbone.force_index = force_index
            model.backbone.exchange_index = exchange_index
            model.index = force_index
            with torch.inference_mode(): 
                if idx==0 or force_index is None or force_index.numel() == 0:
                    outputs.extend(model(frame.to(device)))
                else:
                    outputs.extend(model(frame, partial = True))
            labels.append(squeeze_dict(dict_to_device(annotations, device), dim=0))
       
    # MeanAveragePrecision is extremely slow. It seems fastest to call
    # update() and compute() just once, after all predictions are done.
    mean_ap = MeanAveragePrecision()
    mean_ap.update(outputs, labels)
    metrics = mean_ap.compute()

    counts = model.total_counts() / n_frames
    model.clear_counts()
    return {"metrics": metrics, "counts": counts}

# def get_new_mv_res(mv, res):
#     padding_mv = torch.zeros((168-mv.shape[0], 168, 2), dtype=mv.dtype, device=mv.device)
#     mv = torch.cat([mv, padding_mv], dim=0)
#     #合并4*4-》16*16
#     mv = mv[::4, ::4, :]
#     mask = (mv[:,:,0] == 0) & (mv[:,:,1] == 0)
#     # 从mv转化为mv_mb
#     ori_mv = mv
#     mv = torch.round(mv/16.0).int()
#     relative_mv = ori_mv - mv*16
#     mv = mv_flatten(mv,mask)
#     # 交集的左上&右下坐标
#     IntersectCoords = torch.zeros((ori_mv.shape[0], ori_mv.shape[1], 4), dtype=ori_mv.dtype, device=ori_mv.device)
#     IntersectCoords[:, :, 0] = torch.clamp(relative_mv[:, :, 0], min=0, max=15)
#     IntersectCoords[:, :, 1] = torch.clamp(15+relative_mv[:, :, 1], min=0, max=15)
#     IntersectCoords[:, :, 2] = torch.clamp(15+relative_mv[:, :, 0], min=0, max=15)
#     IntersectCoords[:, :, 3] = torch.clamp(relative_mv[:, :, 1], min=0, max=15)

#     padding = torch.zeros((self.model_size-res.shape[0], res.shape[1], res.shape[2]), dtype=res.dtype, device=res.device)
#     res = torch.cat([res, padding], dim=0).float()
#     # res = torch.cat([res, padding], dim=0).float().sum(dim=2)
#     # res = res.reshape(42, 16, 42, 16).mean(dim=(1, 3))
#     res = res.reshape(res.shape[0]//16, 16, res.shape[1]//16, 16, 3).permute(0, 2, 1, 3, 4).sum(-1).mean(-1).mean(-1)
#     res = res.flatten()
#     return mv, res, IntersectCoords.flatten(0,1)

# def mv_flatten(mv, mask):
#     h = mv.shape[0]
#     w = mv.shape[1]
#     i_grid, j_grid = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
#     i_grid = i_grid.to(mv.device)
#     j_grid = j_grid.to(mv.device)

#     mvs_0 = mv[:,:,0]
#     mvs_1 = mv[:,:,1]
#     mvs_tilde =  ((i_grid + mvs_1) * w + j_grid + mvs_0) - (i_grid * w + j_grid)
#     mvs_tilde[mask] = 100000
#     return mvs_tilde.flatten()


def main():
    config = initialize_run(config_location=Path("configs", "evaluate", "vitdet_vid"))
    long_edge = max(config["model"]["input_shape"][-2:])
    data = VID(
        Path("/data/wangqinyu/qyw/data", "vid"),
        split=config["split"],
        tar_path=Path("/data/wangqinyu/qyw/data", "vid", "data.tar"),
        combined_transform=VIDResize(
            short_edge_length=640 * long_edge // 1024, max_size=long_edge
        ),
    )
    run_evaluations(config, ViTDet, data, evaluate_vitdet_metrics)


if __name__ == "__main__":
    main()

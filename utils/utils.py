# from __future__ import print_function
# from cProfile import label
# import torch.distributed as dist
#
# import functools
# import operator
# import os
# import sys
# import re
# import json
# import numpy as np
# import torch
# import torch.nn as nn
# import string
#
#
# def largest(bboxes):
#     maxS = 0
#     use_gpu = torch.cuda.is_available()
#     maxBox = torch.tensor([0, 0, 0, 0])
#     if use_gpu:
#         maxBox = maxBox.cuda()
#     for box in bboxes:
#         left, top, right, bottom = box[0], box[1], box[2], box[3]
#         s = (right - left) * (bottom - top)
#         if s > maxS:
#             maxS = s
#             maxBox = box
#     return maxBox
#
#
# def confidence(score, bboxes):
#     maxIdx = np.argmax(score)
#     return bboxes[maxIdx]
#
#
# def union(bboxes):
#     leftmin, topmin, rightmax, bottommax = 999, 999, 0, 0
#     for box in bboxes:
#         left, top, right, bottom = box
#         if left == 0 and top == 0:
#             continue
#         leftmin, topmin, rightmax, bottommax = (
#             min(left, leftmin),
#             min(top, topmin),
#             max(right, rightmax),
#             max(bottom, bottommax),
#         )
#
#     return [leftmin, topmin, rightmax, bottommax]
#
#
# def union_target(bboxes_list):
#     target_box_list = []
#     for boxes in bboxes_list:
#         # boxes: [12, 5]
#         target_box = union(boxes)  # target_box: [4]
#         target_box_list.append(target_box)
#     return target_box_list  # [query, 4]
#
#
# def load_folder(folder, suffix):
#     imgs = []
#     for f in sorted(os.listdir(folder)):
#         if f.endswith(suffix):
#             imgs.append(os.path.join(folder, f))
#     return imgs
#
#
# def load_imageid(folder):
#     images = load_folder(folder, "jpg")
#     img_ids = set()
#     for img in images:
#         img_id = int(img.split("/")[-1].split(".")[0].split("_")[-1])
#         img_ids.add(img_id)
#     return img_ids
#
#
# def weights_init(m):
#     """custom weights initialization."""
#     cname = m.__class__
#     if cname == nn.Linear or cname == nn.Conv2d or cname == nn.ConvTranspose2d:
#         m.weight.data.normal_(0.0, 0.02)
#     elif cname == nn.BatchNorm2d:
#         m.weight.data.normal_(1.0, 0.02)
#         m.bias.data.fill_(0)
#     else:
#         print("%s is not initialized." % cname)
#
#
# def init_net(net, net_file):
#     if net_file:
#         net.load_state_dict(torch.load(net_file))
#     else:
#         net.apply(weights_init)
#
#
# def print_model(model, logger):
#     print(model)
#     nParams = 0
#     for w in model.parameters():
#         nParams += functools.reduce(operator.mul, w.size(), 1)
#     if logger:
#         logger.write("nParams=\t" + str(nParams))
#
#
# def save_model(path, model, epoch, optimizer=None):
#     model_dict = {"epoch": epoch, "model_state": model.state_dict()}
#     if optimizer is not None:
#         model_dict["optimizer_state"] = optimizer.state_dict()
#
#     torch.save(model_dict, path)
#
#
# # Remove Flickr30K Entity annotations in a string
# def remove_annotations(s):
#     return re.sub(r"\[[^ ]+ ", "", s).replace("]", "")
#
#
# def calculate_iou(obj1, obj2):
#     EPS = 1e-6
#     area1 = calculate_area(obj1)
#     area2 = calculate_area(obj2)
#     intersection = get_intersection(obj1, obj2)
#     area_int = calculate_area(intersection)
#     return area_int / ((area1 + area2 - area_int) + EPS)
#
#
# def calculate_area(obj):
#     return (obj[2] - obj[0]) * (obj[3] - obj[1])
#
#
# def get_intersection(obj1, obj2):
#     left = obj1[0] if obj1[0] > obj2[0] else obj2[0]
#     top = obj1[1] if obj1[1] > obj2[1] else obj2[1]
#     right = obj1[2] if obj1[2] < obj2[2] else obj2[2]
#     bottom = obj1[3] if obj1[3] < obj2[3] else obj2[3]
#     if left > right or top > bottom:
#         return [0, 0, 0, 0]
#     return [left, top, right, bottom]
#
#
# def get_match_index(src_bboxes, dst_bboxes):
#     indices = set()
#     for src_bbox in src_bboxes:
#         for i, dst_bbox in enumerate(dst_bboxes):
#             iou = calculate_iou(src_bbox, dst_bbox)
#             if iou >= 0.5:
#                 indices.add(i)  # match iou>0.5!!
#     return list(indices)
#
#
# def get_grounding_alignment(src_bboxes, dst_bboxes):
#     alignment = torch.zeros((len(src_bboxes), len(dst_bboxes)))
#     for i, src_bbox in enumerate(src_bboxes):
#         for j, dst_bbox in enumerate(dst_bboxes):
#             iou = calculate_iou(src_bbox, dst_bbox)
#             # if iou >= 0.9:
#             alignment[i][j] = iou  # match iou>0.5!!
#
#     maxval, _ = alignment.max(dim=-1, keepdim=True)
#     predictions = alignment == maxval  # [B, querys, K]
#     return predictions
#
#
# def bbox_is_match(src_bbox, dst_bboxes):
#     for i, dst_bbox in enumerate(dst_bboxes):
#         iou = calculate_iou(src_bbox, dst_bbox)
#         if iou >= 0.5:
#             return True
#     return False
#
#
# def unsupervised_get_match_index(src_bboxes, dst_bboxes):
#     """
#     src_bboxes: dict (for all entities)
#     """
#     indices = set()
#     for entity, src_bboxes_list in src_bboxes.items():
#         for src_bbox in src_bboxes_list:
#             for i, dst_bbox in enumerate(dst_bboxes):
#                 iou = calculate_iou(src_bbox, dst_bbox)
#                 if iou >= 0.5:
#                     indices.add(i)
#     return list(indices)
#
#
# # code for Parallel Processing
#
#
# def setup_for_distributed(is_master):
#     """
#     This function disables printing when not in master process
#     """
#     import builtins as __builtin__
#
#     builtin_print = __builtin__.print
#
#     def print(*args, **kwargs):
#         force = kwargs.pop("force", False)
#         if is_master or force:
#             builtin_print(*args, **kwargs)
#
#     __builtin__.print = print
#
#
# def is_dist_avail_and_initialized():
#     if not dist.is_available():
#         return False
#     if not dist.is_initialized():
#         return False
#     return True
#
#
# def get_world_size():
#     if not is_dist_avail_and_initialized():
#         return 1
#     return dist.get_world_size()
#
#
# def get_rank():
#     if not is_dist_avail_and_initialized():
#         return 0
#     return dist.get_rank()
#
#
# def is_main_process():
#     return get_rank() == 0
#
#
# def save_on_master(*args, **kwargs):
#     if is_main_process():
#         torch.save(*args, **kwargs)
#
#
# class AttrDict(dict):
#     def __init__(self, *args, **kwargs):
#         super(AttrDict, self).__init__(*args, **kwargs)
#         self.__dict__ = self
#
#
# def init_distributed_mode(args):
#     if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
#         args.rank = int(os.environ["RANK"])
#         args.world_size = int(os.environ["WORLD_SIZE"])
#         args.gpu = int(os.environ["LOCAL_RANK"])
#     elif "SLURM_PROCID" in os.environ:
#         args.rank = int(os.environ["SLURM_PROCID"])
#         args.gpu = args.rank % torch.cuda.device_count()
#     else:
#         print("Not using distributed mode")
#         args.distributed = False
#         return
#
#     args.distributed = True
#
#     torch.cuda.set_device(args.gpu)
#     args.dist_backend = "nccl"
#     print(
#         "| distributed init (rank {}): {}".format(args.rank, args.dist_url),
#         flush=True,
#     )
#     torch.distributed.init_process_group(
#         backend=args.dist_backend,
#         init_method=args.dist_url,
#         world_size=args.world_size,
#         rank=args.rank,
#     )
#     torch.distributed.barrier()
#     setup_for_distributed(args.rank == 0)
#
#
# def parse_with_config(parser):
#     args = parser.parse_args()
#     if args.config is not None:
#         config_args = json.load(open(args.config))
#         override_keys = {
#             arg[2:].split("=")[0] for arg in sys.argv[1:] if arg.startswith("--")
#         }
#         for k, v in config_args.items():
#             if k not in override_keys:
#                 setattr(args, k, v)
#     del args.config
#     return args
#
#
# def get_sentence_offsets(paragraph, sentences):
#     """
#     输入：原始段落和已经分割好的句子列表
#     输出：[(句子, 起始索引, 结束索引)]
#     """
#     offsets = []
#     search_start = 0  # 从段落的哪一位开始查找
#     for sent in sentences:
#         # 在剩余段落中查找句子
#         sent = sent.replace(' .', '.')
#         sent = sent.replace(' ,', ',')
#         sent = sent.replace(' \'s', '\'s')
#
#         idx = paragraph.find(sent, search_start)
#         if idx == -1:
#             raise ValueError(f"句子未在段落中找到: {sent}")
#         offsets.append((sent, (idx, idx + len(sent))))
#         search_start = idx + len(sent)  # 更新查找起点，避免匹配到前面重复的句子
#     return offsets
#
# def map_spans_to_sentences(sentences, spans):
#     """
#     sentences: [(sentence_text, (start, end)), ...]
#     spans: [[start, end], ...]
#
#     返回: {句子索引: [((相对start, 相对end), span_text), ...]}
#     """
#     result = {i: [] for i in range(len(sentences))}
#     unmapped = []  # 存储未能分配的span
#
#     for j, span in enumerate(spans):
#         s_start, s_end = span
#         mapped = False
#         for i, (sent_text, (sent_start, sent_end)) in enumerate(sentences):
#             if sent_start <= s_start < sent_end:
#                 # 转换为句子内部索引
#                 rel_start = s_start - sent_start
#                 rel_end = s_end - sent_start
#                 result[i].append((j, (rel_start, rel_end), sent_text[rel_start:rel_end]))
#                 mapped = True
#                 break
#         if not mapped:
#             unmapped.append(span)
#
#     # 检查机制
#     if unmapped:
#         raise ValueError(f"Some spans were not mapped to any sentence: {unmapped}")
#
#     return result
#
#
#
# def custom_sentence_split(paragraph,nlp):
#     # 基于连词 (and, but, etc.) 来分割段落
#     sentence_endings = ['and', 'but', 'so', 'because', 'then', 'while', 'after', 'before', 'or']
#     sentences = []
#     temp_sentence = []
#
#     # 对段落进行句法分析
#     doc = nlp(paragraph)
#
#     for token in doc:
#         temp_sentence.append(token.text)
#         # 如果当前token是句子连接词，则把当前分句保存
#         if token.text.lower() in sentence_endings :
#             sentences.append(" ".join(temp_sentence))
#             temp_sentence = []
#
#     # 如果最后剩余句子，则也需要保存
#     if temp_sentence:
#         sentences.append(" ".join(temp_sentence))
#
#     ## 先把特别短的句子 拼接到上一个句子中，因为大多是上一句的遗漏
#     merged_short = []
#     for sent in sentences:
#         if merged_short and len(sent.split()) < 6:
#             merged_short[-1] = merged_short[-1] + " " + sent
#         else:
#             merged_short.append(sent)
#
#     # 判断相邻句子的主语是否相同，若相同则合并
#     merged_sentences = []
#     for sent in merged_short:
#         if merged_sentences and (sent.split()[0] in ["he", "she", "it", "they"] or 'ing' in sent.split()[0]):
#             merged_sentences[-1] = merged_sentences[-1] + " " + sent
#         else:
#             merged_sentences.append(sent)
#
#     return merged_sentences
#
#
#
# def box_iou(boxes1, boxes2):
#     """
#     计算两个 box 集合的 IOU
#     boxes1: (m, 4)  [x1, y1, x2, y2]
#     boxes2: (n, 4)  [x1, y1, x2, y2]
#     return: (m, n) 的 IOU 矩阵
#     """
#     area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])  # (m,)
#     area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])  # (n,)
#
#     lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # 左上角 (m, n, 2)
#     rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # 右下角 (m, n, 2)
#
#     wh = (rb - lt).clamp(min=0)  # (m, n, 2)
#     inter = wh[:, :, 0] * wh[:, :, 1]  # (m, n)
#
#     iou = inter / (area1[:, None] + area2 - inter + 1e-6)
#     return iou
#
#
# def update_boxes(a, b, c, d, iou_thresh=0.8):
#     """
#     a: (m, 4) boxes
#     b: (m,) scores
#     c: (n, 4) boxes
#     d: (n,) scores
#     当 IOU > 阈值时，将 a 的行更新为得分更高的 box，同时分数取平均
#     """
#     ious = box_iou(a, c)  # (m, n)
#
#     for i in range(a.size(0)):
#         for j in range(c.size(0)):
#             if ious[i, j] > iou_thresh:
#                 if d[j] > b[i]:
#                     # 更新为 c[j]，分数取平均
#                     a[i] = c[j]
#                     b[i] = (b[i] + d[j]) / 2
#                 else:
#                     # 保持 a[i]，分数取平均
#                     b[i] = (b[i] + d[j]) / 2
#
#     return a, b
#
# # def select_top_boxes(orig_boxes, orig_scores, cur_box, cur_score, topk=3):
# #     # 拼接所有候选框和分数
# #     all_boxes = torch.cat(orig_boxes, dim=0)     # shape [M, 4]
# #     all_scores = torch.cat(orig_scores, dim=0)   # shape [M]
# #
# #     # 先把当前重点框加进去
# #     final_boxes = [cur_box]
# #     final_scores = [cur_score]
# #
# #     # 把当前重点框去掉，避免重复（通过比较是否完全相同）
# #     mask = ~torch.all(all_boxes == cur_box, dim=1)
# #     all_boxes = all_boxes[mask]
# #     all_scores = all_scores[mask]
# #
# #     # 按分数排序，选出 top(k-1)
# #     topk_scores, idx = torch.topk(all_scores, k=topk-cur_box.shape[0])
# #     topk_boxes = all_boxes[idx]
# #
# #     # 拼接
# #     final_boxes.append(topk_boxes)
# #     final_scores.append(topk_scores)
# #
# #     return [torch.cat(final_boxes, dim=0), torch.cat(final_scores, dim=0)]
# def select_top_boxes(orig_boxes, orig_scores, cur_box, cur_score, topk=3):
#     """
#     orig_boxes: list of [Mi, 4] tensors
#     orig_scores: list of [Mi] tensors
#     cur_box: [N, 4] tensor
#     cur_score: [N] tensor
#     topk: 总共要选的框数量
#     """
#     # 拼接所有候选框和分数
#     all_boxes = torch.cat(orig_boxes, dim=0)   # [M,4]
#     all_scores = torch.cat(orig_scores, dim=0) # [M]
#
#     # 先把当前框加入 final
#     final_boxes = [cur_box]
#     final_scores = [cur_score]
#
#     # 避免重复：去掉与 cur_box 中任意框重复的 all_boxes
#     mask = torch.ones(all_boxes.shape[0], dtype=torch.bool)
#     for b in cur_box:
#         mask &= ~(torch.all(all_boxes == b, dim=1))
#     all_boxes = all_boxes[mask]
#     all_scores = all_scores[mask]
#
#     # 计算还需要多少个 top 框
#     remaining = max(topk - cur_box.shape[0], 0)
#     if remaining > 0 and all_boxes.shape[0] > 0:
#         topk_scores, idx = torch.topk(all_scores, k=min(remaining, all_boxes.shape[0]))
#         topk_boxes = all_boxes[idx]
#         final_boxes.append(topk_boxes)
#         final_scores.append(topk_scores)
#
#     # 合并返回
#     final_boxes = torch.cat(final_boxes, dim=0)
#     final_scores = torch.cat(final_scores, dim=0)
#     return final_boxes, final_scores
#
#
#
#
#
# def iou(box1, box2):
#     """计算IoU
#        box1: [N, 4], box2: [M, 4]
#     """
#     N, M = box1.size(0), box2.size(0)
#     box1 = box1.unsqueeze(1).expand(N, M, 4)
#     box2 = box2.unsqueeze(0).expand(N, M, 4)
#
#     x1 = torch.max(box1[..., 0], box2[..., 0])
#     y1 = torch.max(box1[..., 1], box2[..., 1])
#     x2 = torch.min(box1[..., 2], box2[..., 2])
#     y2 = torch.min(box1[..., 3], box2[..., 3])
#
#     inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
#     area1 = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
#     area2 = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])
#     return inter / (area1 + area2 - inter + 1e-6)   # [N, M]
#
#
#
#
# def cluster_mentions(clusters, mentions, scores, mention_ids, threshold=0.50):
#     """
#     clusters: tensor[K, 4] 公共框
#     mentions: list[tensor[3, 4]] 每个mention候选框
#     scores: list[tensor[3]] 每个mention候选分数
#     mention_ids: list[int] mention编号
#     threshold: 聚类阈值
#     """
#     # 初始化cluster组
#     cluster_groups = [[] for _ in range(len(clusters))]
#     new_clusters = []  # 新建cluster的列表
#
#     for mid, M, S in zip(mention_ids, mentions, scores):
#         best_score, best_cluster_idx = -1, None
#         for box, sc in zip(M, S):
#             ious = iou(clusters, box.unsqueeze(0)).squeeze(1)  # [K]
#             max_iou, max_idx = ious.max(0)
#             final_score = sc.item() * max_iou.item()
#             if final_score > best_score:
#                 best_score = final_score
#                 best_cluster_idx = max_idx.item()
#         # 判定
#         if best_score < threshold:
#             new_clusters.append([mid])   # 新建 cluster
#         else:
#             cluster_groups[best_cluster_idx].append(mid)
#
#     # 去掉空的 cluster
#     cluster_groups = [c for c in cluster_groups if len(c) > 0]
#     return cluster_groups + new_clusters
#
#
# def recoref(mention_ids, boxes_list,scores_list):
#     cluster_ids = []
#     cluster_boxes = []
#     cluster_scores = []
#
#     visited = set()
#
#     for i in range(len(boxes_list)):
#         if i in visited:
#             continue
#         # 新建一类
#         cur_ids = [mention_ids[i]]
#         cur_boxes = [boxes_list[i]]
#         cur_scores = [scores_list[i]]
#         visited.add(i)
#
#         # 取前两个边界框（顺序不敏感）
#         box_set_i = {tuple(boxes_list[i][0].tolist()), tuple(boxes_list[i][1].tolist())}
#
#         for j in range(i + 1, len(boxes_list)):
#             if j in visited:
#                 continue
#             box_set_j = {tuple(boxes_list[j][0].tolist()), tuple(boxes_list[j][1].tolist())}
#
#             if box_set_i == box_set_j:
#                 # 同一类
#                 cur_ids.append(mention_ids[j])
#                 cur_boxes.append(boxes_list[j])
#                 cur_scores.append(scores_list[j])
#                 visited.add(j)
#
#         cluster_ids.append(cur_ids)
#         cluster_boxes.append(cur_boxes)
#         cluster_scores.append(cur_scores)
#     return  cluster_ids, cluster_boxes,cluster_scores
#
#
# def select_boxes(boxes_list, score_list):
#     b = []
#     s = []
#     for i, box in enumerate(boxes_list):
#         bbox= torch.cat(box,dim=0)
#         sscore = torch.cat(score_list[i])
#         topk_scores, idx = torch.topk(sscore, k=3)
#         topk_boxes = bbox[idx]
#         b.append(topk_boxes)
#         s.append(topk_scores)
#     return b, s
#
# def merge_boxes(box_groups, score_groups, iou_thresh=0.8, score_thresh=0.4):
#     # 展开所有候选
#     all_boxes = torch.cat(box_groups, dim=0)   # (N, 4)
#     all_scores = torch.cat(score_groups, dim=0)  # (N,)
#
#     # 先去掉低于阈值的框
#     keep = all_scores >= score_thresh
#     all_boxes = all_boxes[keep]
#     all_scores = all_scores[keep]
#
#     N = all_boxes.size(0)
#     visited = torch.zeros(N, dtype=torch.bool)
#     merged_boxes, merged_scores, merged_counts = [], [], []
#
#     for i in range(N):
#         if visited[i]:
#             continue
#         # 计算 i 与所有未访问的框的 IoU
#         ious = box_iou(all_boxes[i:i+1], all_boxes)[0]
#         group_idx = (ious > iou_thresh) & (~visited)
#
#         # 跳过没有形成合并的（只自己一个）
#         if group_idx.sum() < 2:
#             continue
#
#         visited[group_idx] = True
#
#         # 分数合并 = 平均
#         merged_score = all_scores[group_idx].mean()
#
#         # 框合并 = 选择该组里得分最高的框
#         indices = group_idx.nonzero(as_tuple=True)[0]
#         max_idx = indices[all_scores[indices].argmax()]
#         merged_box = all_boxes[max_idx]
#
#         merged_boxes.append(merged_box)
#         merged_scores.append(merged_score)
#         merged_counts.append(group_idx.sum().item())  # 记录合并个数
#
#     if len(merged_boxes) == 0:
#         return torch.empty((0,4)), torch.empty((0,)), torch.empty((0,), dtype=torch.long)
#
#     return torch.stack(merged_boxes), torch.stack(merged_scores), torch.tensor(merged_counts, dtype=torch.long)
#
#
# def count_occurrences(c, boxes_list, tol=1e-3):
#     # 转为tensor方便处理
#     box_a = c[0]
#     box_b = c[1]
#
#     count_a = 0
#     count_b = 0
#     count_co = 0
#
#     for t in boxes_list:
#         # 判断是否包含 box_a
#         match_a = (torch.abs(t - box_a) < tol).all(dim=1).any().item()
#         # 判断是否包含 box_b
#         match_b = (torch.abs(t - box_b) < tol).all(dim=1).any().item()
#
#         if match_a:
#             count_a += 1
#         if match_b:
#             count_b += 1
#         if match_a and match_b:
#             count_co += 1
#
#     return  count_co/ (count_a+count_b -count_co)












from __future__ import print_function
from cProfile import label
import torch.distributed as dist

import functools
import operator
import os
import sys
import re
import json
import numpy as np
import torch
import torch.nn as nn
import string


def largest(bboxes):
    maxS = 0
    use_gpu = torch.cuda.is_available()
    maxBox = torch.tensor([0, 0, 0, 0])
    if use_gpu:
        maxBox = maxBox.cuda()
    for box in bboxes:
        left, top, right, bottom = box[0], box[1], box[2], box[3]
        s = (right - left) * (bottom - top)
        if s > maxS:
            maxS = s
            maxBox = box
    return maxBox


def confidence(score, bboxes):
    maxIdx = np.argmax(score)
    return bboxes[maxIdx]


def union(bboxes):
    leftmin, topmin, rightmax, bottommax = 999, 999, 0, 0
    for box in bboxes:
        left, top, right, bottom = box
        if left == 0 and top == 0:
            continue
        leftmin, topmin, rightmax, bottommax = (
            min(left, leftmin),
            min(top, topmin),
            max(right, rightmax),
            max(bottom, bottommax),
        )

    return [leftmin, topmin, rightmax, bottommax]


def union_target(bboxes_list):
    target_box_list = []
    for boxes in bboxes_list:
        # boxes: [12, 5]
        target_box = union(boxes)  # target_box: [4]
        target_box_list.append(target_box)
    return target_box_list  # [query, 4]


def load_folder(folder, suffix):
    imgs = []
    for f in sorted(os.listdir(folder)):
        if f.endswith(suffix):
            imgs.append(os.path.join(folder, f))
    return imgs


def load_imageid(folder):
    images = load_folder(folder, "jpg")
    img_ids = set()
    for img in images:
        img_id = int(img.split("/")[-1].split(".")[0].split("_")[-1])
        img_ids.add(img_id)
    return img_ids


def weights_init(m):
    """custom weights initialization."""
    cname = m.__class__
    if cname == nn.Linear or cname == nn.Conv2d or cname == nn.ConvTranspose2d:
        m.weight.data.normal_(0.0, 0.02)
    elif cname == nn.BatchNorm2d:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    else:
        print("%s is not initialized." % cname)


def init_net(net, net_file):
    if net_file:
        net.load_state_dict(torch.load(net_file))
    else:
        net.apply(weights_init)


def print_model(model, logger):
    print(model)
    nParams = 0
    for w in model.parameters():
        nParams += functools.reduce(operator.mul, w.size(), 1)
    if logger:
        logger.write("nParams=\t" + str(nParams))


def save_model(path, model, epoch, optimizer=None):
    model_dict = {"epoch": epoch, "model_state": model.state_dict()}
    if optimizer is not None:
        model_dict["optimizer_state"] = optimizer.state_dict()

    torch.save(model_dict, path)


# Remove Flickr30K Entity annotations in a string
def remove_annotations(s):
    return re.sub(r"\[[^ ]+ ", "", s).replace("]", "")


def calculate_iou(obj1, obj2):
    EPS = 1e-6
    area1 = calculate_area(obj1)
    area2 = calculate_area(obj2)
    intersection = get_intersection(obj1, obj2)
    area_int = calculate_area(intersection)
    return area_int / ((area1 + area2 - area_int) + EPS)


def calculate_area(obj):
    return (obj[2] - obj[0]) * (obj[3] - obj[1])


def get_intersection(obj1, obj2):
    left = obj1[0] if obj1[0] > obj2[0] else obj2[0]
    top = obj1[1] if obj1[1] > obj2[1] else obj2[1]
    right = obj1[2] if obj1[2] < obj2[2] else obj2[2]
    bottom = obj1[3] if obj1[3] < obj2[3] else obj2[3]
    if left > right or top > bottom:
        return [0, 0, 0, 0]
    return [left, top, right, bottom]


def get_match_index(src_bboxes, dst_bboxes):
    indices = set()
    for src_bbox in src_bboxes:
        for i, dst_bbox in enumerate(dst_bboxes):
            iou = calculate_iou(src_bbox, dst_bbox)
            if iou >= 0.5:
                indices.add(i)  # match iou>0.5!!
    return list(indices)


def get_grounding_alignment(src_bboxes, dst_bboxes):
    alignment = torch.zeros((len(src_bboxes), len(dst_bboxes)))
    for i, src_bbox in enumerate(src_bboxes):
        for j, dst_bbox in enumerate(dst_bboxes):
            iou = calculate_iou(src_bbox, dst_bbox)
            # if iou >= 0.9:
            alignment[i][j] = iou  # match iou>0.5!!

    maxval, _ = alignment.max(dim=-1, keepdim=True)
    predictions = alignment == maxval  # [B, querys, K]
    return predictions


def bbox_is_match(src_bbox, dst_bboxes):
    for i, dst_bbox in enumerate(dst_bboxes):
        iou = calculate_iou(src_bbox, dst_bbox)
        if iou >= 0.5:
            return True
    return False


def unsupervised_get_match_index(src_bboxes, dst_bboxes):
    """
    src_bboxes: dict (for all entities)
    """
    indices = set()
    for entity, src_bboxes_list in src_bboxes.items():
        for src_bbox in src_bboxes_list:
            for i, dst_bbox in enumerate(dst_bboxes):
                iou = calculate_iou(src_bbox, dst_bbox)
                if iou >= 0.5:
                    indices.add(i)
    return list(indices)


# code for Parallel Processing


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(
        "| distributed init (rank {}): {}".format(args.rank, args.dist_url),
        flush=True,
    )
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def parse_with_config(parser):
    args = parser.parse_args()
    if args.config is not None:
        config_args = json.load(open(args.config))
        override_keys = {
            arg[2:].split("=")[0] for arg in sys.argv[1:] if arg.startswith("--")
        }
        for k, v in config_args.items():
            if k not in override_keys:
                setattr(args, k, v)
    del args.config
    return args


def get_sentence_offsets(paragraph, sentences):
    """
    输入：原始段落和已经分割好的句子列表
    输出：[(句子, 起始索引, 结束索引)]
    """
    offsets = []
    search_start = 0  # 从段落的哪一位开始查找
    for sent in sentences:
        # 在剩余段落中查找句子
        sent = sent.replace(' .', '.')
        sent = sent.replace(' ,', ',')
        sent = sent.replace(' \'s', '\'s')

        idx = paragraph.find(sent, search_start)
        if idx == -1:
            raise ValueError(f"句子未在段落中找到: {sent}")
        offsets.append((sent, (idx, idx + len(sent))))
        search_start = idx + len(sent)  # 更新查找起点，避免匹配到前面重复的句子
    return offsets

def map_spans_to_sentences(sentences, spans):
    """
    sentences: [(sentence_text, (start, end)), ...]
    spans: [[start, end], ...]

    返回: {句子索引: [((相对start, 相对end), span_text), ...]}
    """
    result = {i: [] for i in range(len(sentences))}
    unmapped = []  # 存储未能分配的span

    for j, span in enumerate(spans):
        s_start, s_end = span
        mapped = False
        for i, (sent_text, (sent_start, sent_end)) in enumerate(sentences):
            if sent_start <= s_start < sent_end:
                # 转换为句子内部索引
                rel_start = s_start - sent_start
                rel_end = s_end - sent_start
                result[i].append((j, (rel_start, rel_end), sent_text[rel_start:rel_end]))
                mapped = True
                break
        if not mapped:
            unmapped.append(span)

    # 检查机制
    if unmapped:
        raise ValueError(f"Some spans were not mapped to any sentence: {unmapped}")

    return result



def custom_sentence_split(paragraph,nlp):
    # 基于连词 (and, but, etc.) 来分割段落
    sentence_endings = ['and', 'but', 'so', 'because', 'then', 'while', 'after', 'before', 'or']
    sentences = []
    temp_sentence = []

    # 对段落进行句法分析
    doc = nlp(paragraph)

    for token in doc:
        temp_sentence.append(token.text)
        # 如果当前token是句子连接词，则把当前分句保存
        if token.text.lower() in sentence_endings :
            sentences.append(" ".join(temp_sentence))
            temp_sentence = []

    # 如果最后剩余句子，则也需要保存
    if temp_sentence:
        sentences.append(" ".join(temp_sentence))

    ## 先把特别短的句子 拼接到上一个句子中，因为大多是上一句的遗漏
    merged_short = []
    for sent in sentences:
        if merged_short and len(sent.split()) < 6:
            merged_short[-1] = merged_short[-1] + " " + sent
        else:
            merged_short.append(sent)

    # 判断相邻句子的主语是否相同，若相同则合并
    merged_sentences = []
    for sent in merged_short:
        if merged_sentences and (sent.split()[0] in ["he", "she", "it", "they"] or 'ing' in sent.split()[0]):
            merged_sentences[-1] = merged_sentences[-1] + " " + sent
        else:
            merged_sentences.append(sent)

    return merged_sentences



def box_iou(boxes1, boxes2):
    """
    计算两个 box 集合的 IOU
    boxes1: (m, 4)  [x1, y1, x2, y2]
    boxes2: (n, 4)  [x1, y1, x2, y2]
    return: (m, n) 的 IOU 矩阵
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])  # (m,)
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])  # (n,)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # 左上角 (m, n, 2)
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # 右下角 (m, n, 2)

    wh = (rb - lt).clamp(min=0)  # (m, n, 2)
    inter = wh[:, :, 0] * wh[:, :, 1]  # (m, n)

    iou = inter / (area1[:, None] + area2 - inter + 1e-6)
    return iou


def update_boxes(a, b, c, d, iou_thresh=0.8):
    """
    a: (m, 4) boxes
    b: (m,) scores
    c: (n, 4) boxes
    d: (n,) scores
    当 IOU > 阈值时，将 a 的行更新为得分更高的 box，同时分数取平均
    """
    ious = box_iou(a, c)  # (m, n)

    for i in range(a.size(0)):
        for j in range(c.size(0)):
            if ious[i, j] > iou_thresh:
                if d[j] > b[i]:
                    # 更新为 c[j]，分数取平均
                    a[i] = c[j]
                    b[i] = (b[i] + d[j]) / 2
                else:
                    # 保持 a[i]，分数取平均
                    b[i] = (b[i] + d[j]) / 2

    return a, b

# def select_top_boxes(orig_boxes, orig_scores, cur_box, cur_score, topk=3):
#     # 拼接所有候选框和分数
#     all_boxes = torch.cat(orig_boxes, dim=0)     # shape [M, 4]
#     all_scores = torch.cat(orig_scores, dim=0)   # shape [M]
#
#     # 先把当前重点框加进去
#     final_boxes = [cur_box]
#     final_scores = [cur_score]
#
#     # 把当前重点框去掉，避免重复（通过比较是否完全相同）
#     mask = ~torch.all(all_boxes == cur_box, dim=1)
#     all_boxes = all_boxes[mask]
#     all_scores = all_scores[mask]
#
#     # 按分数排序，选出 top(k-1)
#     topk_scores, idx = torch.topk(all_scores, k=topk-cur_box.shape[0])
#     topk_boxes = all_boxes[idx]
#
#     # 拼接
#     final_boxes.append(topk_boxes)
#     final_scores.append(topk_scores)
#
#     return [torch.cat(final_boxes, dim=0), torch.cat(final_scores, dim=0)]
def select_top_boxes(orig_boxes, orig_scores, cur_box, cur_score, topk=2):
    """
    orig_boxes: list of [Mi, 4] tensors
    orig_scores: list of [Mi] tensors
    cur_box: [N, 4] tensor
    cur_score: [N] tensor
    topk: 总共要选的框数量
    """
    # 拼接所有候选框和分数
    all_boxes = torch.cat(orig_boxes, dim=0)   # [M,4]
    all_scores = torch.cat(orig_scores, dim=0) # [M]

    # 先把当前框加入 final
    final_boxes = [cur_box]
    final_scores = [cur_score]

    # 避免重复：去掉与 cur_box 中任意框重复的 all_boxes
    mask = torch.ones(all_boxes.shape[0], dtype=torch.bool)
    for b in cur_box:
        mask &= ~(torch.all(all_boxes == b, dim=1))
    all_boxes = all_boxes[mask]
    all_scores = all_scores[mask]

    # 计算还需要多少个 top 框
    remaining = max(topk - cur_box.shape[0], 0)
    if remaining > 0 and all_boxes.shape[0] > 0:
        topk_scores, idx = torch.topk(all_scores, k=min(remaining, all_boxes.shape[0]))
        topk_boxes = all_boxes[idx]
        final_boxes.append(topk_boxes)
        final_scores.append(topk_scores)

    # 合并返回
    final_boxes = torch.cat(final_boxes, dim=0)
    final_scores = torch.cat(final_scores, dim=0)
    return final_boxes, final_scores





def iou(box1, box2):
    """计算IoU
       box1: [N, 4], box2: [M, 4]
    """
    N, M = box1.size(0), box2.size(0)
    box1 = box1.unsqueeze(1).expand(N, M, 4)
    box2 = box2.unsqueeze(0).expand(N, M, 4)

    x1 = torch.max(box1[..., 0], box2[..., 0])
    y1 = torch.max(box1[..., 1], box2[..., 1])
    x2 = torch.min(box1[..., 2], box2[..., 2])
    y2 = torch.min(box1[..., 3], box2[..., 3])

    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area1 = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
    area2 = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])
    return inter / (area1 + area2 - inter + 1e-6)   # [N, M]




def cluster_mentions(clusters, mentions, scores, mention_ids, threshold=0.50):
    """
    clusters: tensor[K, 4] 公共框
    mentions: list[tensor[3, 4]] 每个mention候选框
    scores: list[tensor[3]] 每个mention候选分数
    mention_ids: list[int] mention编号
    threshold: 聚类阈值
    """
    # 初始化cluster组
    cluster_groups = [[] for _ in range(len(clusters))]
    new_clusters = []  # 新建cluster的列表

    for mid, M, S in zip(mention_ids, mentions, scores):
        best_score, best_cluster_idx = -1, None
        for box, sc in zip(M, S):
            ious = iou(clusters, box.unsqueeze(0)).squeeze(1)  # [K]
            max_iou, max_idx = ious.max(0)
            final_score = sc.item() * max_iou.item()
            if final_score > best_score:
                best_score = final_score
                best_cluster_idx = max_idx.item()
        # 判定
        if best_score < threshold:
            new_clusters.append([mid])   # 新建 cluster
        else:
            cluster_groups[best_cluster_idx].append(mid)

    # 去掉空的 cluster
    cluster_groups = [c for c in cluster_groups if len(c) > 0]
    return cluster_groups + new_clusters



import spacy
nlp = spacy.load("en_core_web_sm")

def get_number(phrase: str):
    """判断短语的单复数"""
    doc = nlp(phrase)
    for token in doc:
        if token.pos_ in ["NOUN", "PROPN", "PRON"]:
            if "Number=Sing" in token.morph:
                return "Singular"
            elif "Number=Plur" in token.morph:
                return "Plural"
    return "Unknown"

import spacy
from nltk.corpus import wordnet as wn
import nltk

# 如果没下载过 WordNet，需要先运行一次
# nltk.download('wordnet')
# nltk.download('omw-1.4')

# 加载英文模型
nlp = spacy.load("en_core_web_sm")


def get_hypernym_chain(word: str, depth: int = 5):
    """
    给定一个名词，返回它的上位词链 (hypernyms)，从具体到一般。
    """
    synsets = wn.synsets(word, pos=wn.NOUN)
    if not synsets:
        return []

    syn = synsets[0]  # 默认取第一个义项（最常用的含义）
    chain = [syn]

    for _ in range(depth):
        hypers = chain[-1].hypernyms()
        if not hypers:
            break
        chain.append(hypers[0])  # 取第一个上位义项

    return [c.lemma_names()[0] for c in chain]


def phrase_to_category(phrase: str, depth: int = 5):
    """
    给定一个短语，先用 spaCy 找到 head noun，再用 WordNet 提取类别层级。
    """
    doc = nlp(phrase)

    nouns = [t for t in doc if t.pos_ == "NOUN"]
    if not nouns:
        return None
    token = nouns[-1]  # 取最后一个名词
    # 如果是复数名词，直接用原词，而不是 lemma_
    if token.tag_ == "NNS":
        nouns = token.text.lower()
    else:
        nouns = token.lemma_.lower()


    hypernym_chain = get_hypernym_chain(nouns, depth=depth)
    if len(hypernym_chain) == 0:
        return None
    minx = min(len(hypernym_chain)-1, 2)
    return hypernym_chain[minx]


def cluster_mentions_batch(mentions, scores, mention_ids, query, threshold=0.7):
    """
    mentions: list[tensor[n, 4]] 每个 mention 候选框
    scores: list[tensor[n]] 每个 mention 候选分数
    mention_ids: list[int] mention编号
    query: list[str] 每个 mention 对应的文本
    threshold: 聚类阈值
    """

    clusters = []
    cluster_boxes = []
    cluster_categories = []
    cluster_numbers = []

    for mid, M, S, q in zip(mention_ids, mentions, scores, query):
        q_number = get_number(q)           # "Singular" / "Plural"
        q_category = phrase_to_category(q) # str 或 list

        # 如果是单数，只考虑第一个候选框
        if q_number == "Singular":
            M, S = M[:1], S[:1]

        # 如果还没有 cluster，新建一个
        if len(clusters) == 0:
            clusters.append([mid])
            cluster_boxes.append(M)

            if isinstance(q_category, list):
                cluster_categories.append(set(q_category))
            else:
                cluster_categories.append(set([q_category]))

            cluster_numbers.append(set([q_number]))
            continue

        # 批量计算每个 cluster 的最大 IoU
        best_score, best_idx = -1, None
        for idx, cb in enumerate(cluster_boxes):
            ious = iou(M, cb)  # [n, m]
            max_iou = ious.max().item()

            # 类别一致性
            if isinstance(q_category, list):
                cat_score = 1.0 if any(c in cluster_categories[idx] for c in q_category) else 0.0
            else:
                cat_score = 1.0 if q_category in cluster_categories[idx] else 0.0

            # 单复数一致性
            num_score = 1.0 if q_number in cluster_numbers[idx] else 0.0

            # 结合候选框置信度（取该 mention 的最大分数）
            conf_score = S.max().item() if len(S) > 0 else 1.0

            final_score = conf_score * max_iou * cat_score * num_score

            if final_score > best_score:
                best_score, best_idx = final_score, idx

        # 判断是否加入已有 cluster
        if best_score >= threshold:
            clusters[best_idx].append(mid)
            cluster_boxes[best_idx] = torch.cat([cluster_boxes[best_idx], M], dim=0)

            if isinstance(q_category, list):
                cluster_categories[best_idx].update(q_category)
            else:
                cluster_categories[best_idx].add(q_category)

            cluster_numbers[best_idx].add(q_number)
        else:
            clusters.append([mid])
            cluster_boxes.append(M)

            if isinstance(q_category, list):
                cluster_categories.append(set(q_category))
            else:
                cluster_categories.append(set([q_category]))

            cluster_numbers.append(set([q_number]))

    return clusters









def recoref(mention_ids, boxes_list,scores_list):
    cluster_ids = []
    cluster_boxes = []
    cluster_scores = []

    visited = set()

    for i in range(len(boxes_list)):
        if i in visited:
            continue
        # 新建一类
        cur_ids = [mention_ids[i]]
        cur_boxes = [boxes_list[i]]
        cur_scores = [scores_list[i]]
        visited.add(i)

        # 取前两个边界框（顺序不敏感）
        box_set_i = {tuple(boxes_list[i][0].tolist()), tuple(boxes_list[i][1].tolist())}

        for j in range(i + 1, len(boxes_list)):
            if j in visited:
                continue
            box_set_j = {tuple(boxes_list[j][0].tolist()), tuple(boxes_list[j][1].tolist())}

            if box_set_i == box_set_j:
                # 同一类
                cur_ids.append(mention_ids[j])
                cur_boxes.append(boxes_list[j])
                cur_scores.append(scores_list[j])
                visited.add(j)

        cluster_ids.append(cur_ids)
        cluster_boxes.append(cur_boxes)
        cluster_scores.append(cur_scores)
    return  cluster_ids, cluster_boxes,cluster_scores


def select_boxes(boxes_list, score_list):
    b = []
    s = []
    for i, box in enumerate(boxes_list):
        bbox= torch.cat(box,dim=0)
        sscore = torch.cat(score_list[i])
        topk_scores, idx = torch.topk(sscore, k=2)
        topk_boxes = bbox[idx]
        b.append(topk_boxes)
        s.append(topk_scores)
    return b, s

def merge_boxes(box_groups, score_groups, iou_thresh=0.8, score_thresh=0.4):
    # 展开所有候选
    all_boxes = torch.cat(box_groups, dim=0)   # (N, 4)
    all_scores = torch.cat(score_groups, dim=0)  # (N,)

    # 先去掉低于阈值的框
    keep = all_scores >= score_thresh
    all_boxes = all_boxes[keep]
    all_scores = all_scores[keep]

    N = all_boxes.size(0)
    visited = torch.zeros(N, dtype=torch.bool)
    merged_boxes, merged_scores, merged_counts = [], [], []

    for i in range(N):
        if visited[i]:
            continue
        # 计算 i 与所有未访问的框的 IoU
        ious = box_iou(all_boxes[i:i+1], all_boxes)[0]
        group_idx = (ious > iou_thresh) & (~visited)

        # 跳过没有形成合并的（只自己一个）
        if group_idx.sum() < 2:
            continue

        visited[group_idx] = True

        # 分数合并 = 平均
        merged_score = all_scores[group_idx].mean()

        # 框合并 = 选择该组里得分最高的框
        indices = group_idx.nonzero(as_tuple=True)[0]
        max_idx = indices[all_scores[indices].argmax()]
        merged_box = all_boxes[max_idx]

        merged_boxes.append(merged_box)
        merged_scores.append(merged_score)
        merged_counts.append(group_idx.sum().item())  # 记录合并个数

    if len(merged_boxes) == 0:
        return torch.empty((0,4)), torch.empty((0,)), torch.empty((0,), dtype=torch.long)

    return torch.stack(merged_boxes), torch.stack(merged_scores), torch.tensor(merged_counts, dtype=torch.long)


def count_occurrences(c, boxes_list, tol=1e-3):
    # 转为tensor方便处理
    box_a = c[0]
    box_b = c[1]

    count_a = 0
    count_b = 0
    count_co = 0

    for t in boxes_list:
        # 判断是否包含 box_a
        match_a = (torch.abs(t - box_a) < tol).all(dim=1).any().item()
        # 判断是否包含 box_b
        match_b = (torch.abs(t - box_b) < tol).all(dim=1).any().item()

        if match_a:
            count_a += 1
        if match_b:
            count_b += 1
        if match_a and match_b:
            count_co += 1

    return  count_co/ (count_a+count_b -count_co)


def box_ious(box1, box2):
    x1 = torch.max(box1[0], box2[0])
    y1 = torch.max(box1[1], box2[1])
    x2 = torch.min(box1[2], box2[2])
    y2 = torch.min(box1[3], box2[3])

    inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else torch.tensor(0.0)


# 判断一个集群是否满足得分条件
def valid_cluster(score):
    return (score[0] > 0.8) and (score[0] > score[1] * 2)


def merge_clusters(clusters, bboxes, scores):
    merged_clusters, merged_bboxes, merged_scores = [], [], []
    used = set()

    for i in range(len(clusters)):
        if i in used:
            continue

        cur_cluster = clusters[i].copy()
        cur_bbox = bboxes[i]
        cur_score = scores[i]

        for j in range(i + 1, len(clusters)):
            if j in used:
                continue

            if valid_cluster(scores[i]) and valid_cluster(scores[j]) and \
                    box_ious(bboxes[i][0], bboxes[j][0]) > 0.9:
                # 合并 ID
                cur_cluster += clusters[j]
                used.add(j)

                # 选择得分最大的 bbox
                all_scores = torch.cat([cur_score, scores[j]])
                all_bboxes = torch.cat([cur_bbox, bboxes[j]], dim=0)
                max_idx = torch.argmax(all_scores)

                cur_score = all_scores[max_idx:max_idx + 2]  # 保留为 tensor([x])
                cur_bbox = all_bboxes[max_idx:max_idx + 2]  # 保留为 (1,4)

        merged_clusters.append(cur_cluster)
        merged_bboxes.append(cur_bbox)
        merged_scores.append(cur_score)
    return merged_clusters, merged_bboxes, merged_scores
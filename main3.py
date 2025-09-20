import os
import json
import tqdm
import spacy
import torch
import numpy as np
from PIL import Image
from utils.utils import union
from maverick import Maverick
from spacy.symbols import ORTH
from utils.evaluator import Evaluator
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling.vgmodel import GLIPDemo
from caption_format import get_ontonotes ,find_token_span
from utils.utils import (get_sentence_offsets, map_spans_to_sentences,recoref,select_boxes,count_occurrences,cluster_mentions_batch,
                         custom_sentence_split, cluster_mentions,box_iou,select_top_boxes,merge_boxes)


list_of_pronouns = [    "them",    "they",    "their",    "this",    "that",    "which",    "those",    "it",    "who",    "he",    "she",    "her",    "him",    "its",    "his",]

## 加载指代消解模型
coref_model = Maverick(hf_name_or_path="/media/team/data/CODE/vg/GLIP-main/maverick-coref/weights",
                       device="cuda:0", )

## 加载视觉定位模型
config_file = "/media/team/data/CODE/vg/GLIP-main/configs/pretrain/glip_Swin_T_O365_GoldG.yaml"
weight_file = "/media/team/data/CODE/vg/GLIP-main/MODEL/glip_tiny_model_o365_goldg_cc_sbu.pth"
cfg.local_rank = 0
cfg.num_gpus = 1
cfg.merge_from_file(config_file)
cfg.merge_from_list(["MODEL.WEIGHT", weight_file])
cfg.merge_from_list(["MODEL.DEVICE", "cuda"])
glip_demo = GLIPDemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,
    show_mask_heatmaps=False
)

# 加载 SpaCy 英文模型
nlp = spacy.load("en_core_web_sm")
nlp.tokenizer.add_special_case("t-shirt", [{ORTH: "t-shirt"}])
nlp.tokenizer.add_special_case("t-shirts", [{ORTH: "t-shirts"}])


target_bboxlist = []
predict_bboxlist = []
pronoun_target_bboxlist = []
pronoun_predict_bboxlist = []
noun_target_bboxlist = []
noun_predict_bboxlist = []
evaluator = Evaluator()

## 加载数据
datas = json.load(open('/media/team/data/Dataset/CIN/new/testval_annotations.json'))

for i , data in enumerate(tqdm.tqdm(datas)):
    if   data['split'] =='val' :
    # if i != '104285082':
    # if i <907 :
    # if        data['image'] != '104285082':
        pass
    else:
        ## 获取 当前样本数据
        split = data['split']
        image_id = data['image']
        print(i)
        print(image_id)

        img_width = data['img_width']
        img_height = data['img_height']
        caption = data['captions'].strip('\u00a0')  # 字符串

        query = data['query'] # 包含 num_query 个 字符串短语 的  列表
        query_start_end = data['query_start_end'] # 包含 num_query 个 字符串短语 在captain中 char起始位置的 列表

        query_target_clusters = data['cluster'] #包含  num_query 个  代表对应 字符串短语  所属集群label 的列表
        query_target_bboxes = data['target_bboxes'] #包含  num_query 个  对应字符串短语 边界框 的列表


        ## 构建 代表 每个查询 是否为 代词的 列表
        pronoun_label = {}
        for j, q in enumerate(query):
            if q.strip(" ,.")  in list_of_pronouns:
                pronoun_label[j] = 1
            else:
                pronoun_label[j] = 0


        ## for text_coreference
        ontonotes_tokens, token_char_spans = get_ontonotes(caption)
        nested_token_char_spans = []
        for i in range(len(token_char_spans)):
            nested_token_char_spans.extend(token_char_spans[i])
        _, ontonotes_mentions_index = find_token_span(query_start_end,nested_token_char_spans)
        coref_result = coref_model.predict(ontonotes_tokens, predefined_mentions=ontonotes_mentions_index)
        clusters_token_offsets = coref_result['clusters_token_offsets']


        existing_tuples = set(t for group in clusters_token_offsets for t in group)


        ## 获取 由 每个query 在 query列表 中顺序代表 query 的 消解结果
        idx_map = {v: ii for ii, v in enumerate(ontonotes_mentions_index)}
        clusters_token_offsets_idx = [[idx_map[t] for t in group if t in idx_map] for group in clusters_token_offsets]

        pron2noun = {}
        sorted_ontonotes_mentions_index = sorted(ontonotes_mentions_index, key=lambda x: (x[0], x[1]))
        for cluster_token_offsets_idx in clusters_token_offsets_idx:
            for cur_mention_token_offsets_idx in cluster_token_offsets_idx:
                if pronoun_label[cur_mention_token_offsets_idx] ==1:
                    soted_index = sorted_ontonotes_mentions_index.index(ontonotes_mentions_index[cur_mention_token_offsets_idx])
                    cur_pron_logits = coref_result['logits'][0,soted_index]
                    kkk = torch.argmax(cur_pron_logits)
                    really_noun = ontonotes_mentions_index.index(sorted_ontonotes_mentions_index[kkk.item()])
                    pron2noun[cur_mention_token_offsets_idx] = really_noun



        ## for visual grounding
        cur_grounding_boxes = {}
        cur_grounding_scores = {}

        ## 句子分割
        sentences = []
        if caption.count(".") >2:
            sentences_temp = [s.strip() for s in caption.split(".") if s.strip()]
            for  sentence in sentences_temp:
                if len(sentence.split()) >20:
                    merged_sentence = custom_sentence_split(sentence,nlp)
                    sentences.extend(merged_sentence)
                else:
                    sentences.append(sentence)
        else:
            sentences = custom_sentence_split(caption,nlp)
        ## 下标索引
        offsets = get_sentence_offsets(caption, sentences)  # 获取每个子句的起始位置
        results = map_spans_to_sentences(offsets, query_start_end)  #

        ## 图像编码
        image = os.path.join('/media/team/data/Dataset/flickr30k/flickr30k-images/', f'{image_id}.jpg')
        pil_image = Image.open(image).convert("RGB")
        image = np.array(pil_image)[:, :, [2, 1, 0]]

        for key, value in results.items():
            short_sentence = offsets[key][0]
            span_index = []
            span = []
            order_number = []

            for x in value:
                span_index.append([list(x[1])])
                span.append(x[2])
                order_number.append(x[0])

            if len(span_index) == 0:
                continue
            else:
                bboxs, p2b_scores, result = glip_demo.inference(image, short_sentence, span_index)
                top_scores, top_indices = torch.topk(p2b_scores, k=min(2,p2b_scores.shape[1]), dim=1)
                top_bboxes = torch.cat( [bboxs[top_indices], torch.zeros(top_indices.shape[0],(2-min(2,p2b_scores.shape[1])),4)],dim=1).to(top_indices.device)
                top_scores = torch.cat([top_scores,torch.zeros(top_scores.shape[0],(2-min(2,p2b_scores.shape[1])) )],dim=1).to(top_indices.device)
                for iii ,spanindex in enumerate(span_index):
                    cur_grounding_boxes[order_number[iii]] = top_bboxes[iii]
                    cur_grounding_scores[order_number[iii]] = top_scores[iii]

        ## merge
        new_clusters_token_offsets_idx = []
        new_clusters_bboxes = []
        new_grounding_scores = []


        split_cur_cluster = []
        split_cur_bboxs = []
        split_cur_scores = []



        for i10,   cluster in enumerate(clusters_token_offsets_idx):


            cluster = [noun_mention for noun_mention in cluster if pronoun_label[noun_mention]==0  ]
            cur_clusters_bboxes = [cur_grounding_boxes[m] for m in cluster if pronoun_label[m] == 0 ]
            cur_clusters_scores = [cur_grounding_scores[m] for m in cluster  if pronoun_label[m] == 0]
            if len(cur_clusters_bboxes) ==0:
                cluster = clusters_token_offsets_idx[i10]
                cur_clusters_bboxes = [cur_grounding_boxes[m] for m in cluster]
                cur_clusters_scores = [cur_grounding_scores[m] for m in cluster]

            c,d,e = merge_boxes(cur_clusters_bboxes,score_groups=cur_clusters_scores,iou_thresh=0.6,score_thresh=0.5)






            if (torch.max(e).item() /len(cur_clusters_bboxes))  > 0.7 :
                x = torch.max(e).item()
                y = torch.argmax(d).item()
                cccc = c[y:y+1]
                temp_really_cur_cluster  = [ ]
                temp_really_cur_bboxs = []
                temp_really_cur_scores = []
                # torch.ma
                for cc, mention_index in enumerate(cluster):

                    if torch.any(box_iou(cur_grounding_boxes[mention_index][0:2],cccc) >0.6):
                        temp_really_cur_cluster.append(mention_index)
                        temp_really_cur_bboxs.append(cur_grounding_boxes[mention_index])
                        temp_really_cur_scores.append(cur_grounding_scores[mention_index])
                    else:
                        split_cur_cluster.append(mention_index)
                        split_cur_bboxs.append(cur_grounding_boxes[mention_index])
                        split_cur_scores.append(cur_grounding_scores[mention_index])

                new_clusters_token_offsets_idx.append(temp_really_cur_cluster)
                new_clusters_bboxes.append(select_top_boxes(cur_clusters_bboxes,cur_clusters_scores,c,d)[0])

                new_grounding_scores.append(select_top_boxes(cur_clusters_bboxes,cur_clusters_scores,c,d)[1])



            else:
                if len(c) == 0:
                    mask = [torch.any(row > 0.5) for row in cur_clusters_scores]
                    # 统计 True 的数量
                    num_confident_grounding = sum(mask).item()

                    if num_confident_grounding == 0:
                        new_clusters_token_offsets_idx.append(cluster)
                        mat = torch.stack(cur_clusters_scores)
                        row_max, _ = torch.max(mat, dim=1)
                        row_idx = torch.argmax(row_max)
                        new_clusters_bboxes.append(cur_clusters_bboxes[row_idx.item()])
                        new_grounding_scores.append(cur_clusters_scores[row_idx.item()])
                    else:

                        for cc, mention_index in enumerate(cluster):
                            split_cur_cluster.append(mention_index)
                            split_cur_bboxs.append(cur_grounding_boxes[cc])
                            split_cur_scores.append(cur_grounding_scores[cc])
                else:
                    split_cur_bboxs.extend(cur_clusters_bboxes)
                    split_cur_scores.extend(cur_clusters_scores)
                    split_cur_cluster.extend(cluster)

        if not len(split_cur_cluster) == 0:

            new_results = cluster_mentions_batch(torch.stack(split_cur_bboxs, dim=0),
                                                     torch.stack(split_cur_scores, dim=0), split_cur_cluster, query,threshold=0.7)
            new_clusters_token_offsets_idx.extend(new_results)
            for n_cluster in new_results:
                if len(n_cluster) == 1:
                    new_clusters_bboxes.append(cur_grounding_boxes[n_cluster[0]])
                    new_grounding_scores.append(cur_grounding_scores[n_cluster[0]])
                else:
                    cur_scores = torch.stack([cur_grounding_scores[c_m_i] for c_m_i in n_cluster], dim=0)
                    cur_boxes = torch.stack([cur_grounding_boxes[c_m_i] for c_m_i in n_cluster], dim=0)
                    row_idx = torch.argmax(cur_scores) // cur_scores.size(1)
                    new_clusters_bboxes.append(cur_grounding_boxes[row_idx.item()])
                    new_grounding_scores.append(cur_grounding_scores[row_idx.item()])



        for key, val in pron2noun.items():
            for clusters_token_offsets_idx in new_clusters_token_offsets_idx:
                if val in clusters_token_offsets_idx:
                    clusters_token_offsets_idx.append(key)
                    clusters_token_offsets_idx.sort()
                        # cur_grounding_boxes
        flat_new_clusters_token_offsets_idx = [item for sublist in new_clusters_token_offsets_idx for item in sublist]
        for i4 in range(len(query)):
            if i4 not  in flat_new_clusters_token_offsets_idx:
                new_clusters_token_offsets_idx.append([i4])
                new_grounding_scores.append(cur_grounding_scores[i4])
                new_clusters_bboxes.append((cur_grounding_boxes[i4]))


        indices = list(range(len(new_clusters_token_offsets_idx)))
        sorted_indices = sorted(indices, key=lambda i: min(new_clusters_token_offsets_idx[i]))

        # 按排序后的索引重新排列两个列表
        sorted_clusters_token_offsets_idx = [new_clusters_token_offsets_idx[i5] for i5 in sorted_indices]
        sorted_grounding_scores = [new_grounding_scores[i6] for i6 in sorted_indices]
        sorted_clusters_bboxes = [new_clusters_bboxes[i7] for i7 in sorted_indices]

        clusters = {}
        query_nums = []
        for i8, q in enumerate(query):
            query_nums.append(f"{q}{i8}")
        for i9, clusters_token_offset in enumerate(sorted_clusters_token_offsets_idx):
            clusters[str(i9)] = []
            for token_offset in clusters_token_offset:
                phrase = query_nums[token_offset]
                clusters[str(i9)].append(phrase)

        cur_sample = {}
        cur_sample['name'] = image_id
        cur_sample['type'] = "clusters"
        cur_sample['clusters'] = clusters
        if split == 'test':
            json.dump(cur_sample, open(
                os.path.join('/media/team/data/CODE/vg/GLIP-main/zero_shot/data/predict_result_5/test', f'{image_id}.json'),'w'))

        else:
            json.dump(cur_sample, open(
                os.path.join('/media/team/data/CODE/vg/GLIP-main/zero_shot/data/predict_result_5/val', f'{image_id}.json'),
                'w'))



        for j, custer_index in enumerate(sorted_clusters_token_offsets_idx):
            for mention_indexx in custer_index:
                cur_target_bbox = query_target_bboxes[mention_indexx]
                if cur_target_bbox == []:
                    continue
                n_cur_target_bbox = union(cur_target_bbox)
                n_cur_target_bbox[0] = n_cur_target_bbox[0] * img_width / 100
                n_cur_target_bbox[1] = n_cur_target_bbox[1] * img_height / 100
                n_cur_target_bbox[2] = n_cur_target_bbox[0] + n_cur_target_bbox[2] * img_width / 100
                n_cur_target_bbox[3] = n_cur_target_bbox[1] + n_cur_target_bbox[3] * img_height / 100
                target_bboxlist.append(n_cur_target_bbox)


                if pronoun_label[mention_indexx] ==1:
                    temp_bboxs = []
                    temp_score = sorted_grounding_scores[j]
                    temp_bbox = sorted_clusters_bboxes[j]

                    values, indices = torch.topk(temp_score, k=min(2, len(cur_target_bbox)))

                    for index in indices:
                        temp_bboxs.append(temp_bbox[index].tolist())
                    cur_bbox = union(temp_bboxs)
                    pronoun_target_bboxlist.append(n_cur_target_bbox)
                    pronoun_predict_bboxlist.append(cur_bbox)
                    predict_bboxlist.append(cur_bbox)
                else:
                    temp_bboxs = []
                    temp_score = cur_grounding_scores[mention_indexx]
                    temp_bbox =  cur_grounding_boxes [mention_indexx]

                    values, indices = torch.topk(temp_score, k=min(2, len(cur_target_bbox)))

                    for index in indices:
                        temp_bboxs.append(temp_bbox[index].tolist())
                    cur_bbox = union(temp_bboxs)
                    noun_target_bboxlist.append(n_cur_target_bbox)
                    noun_predict_bboxlist.append(cur_bbox)
                    predict_bboxlist.append(cur_bbox)


accuracy, _ = evaluator.evaluate(predict_bboxlist, target_bboxlist)  # [query, 4]
pronoun_accuracy, _ = evaluator.evaluate(
    pronoun_predict_bboxlist, pronoun_target_bboxlist
)  # [query, 4]
noun_accuracy, _ = evaluator.evaluate(
    noun_predict_bboxlist, noun_target_bboxlist
)
#
print(accuracy)
print(pronoun_accuracy)
print(noun_accuracy)
#
#
#
#
#


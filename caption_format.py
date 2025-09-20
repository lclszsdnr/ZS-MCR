import nltk

# nltk.download('punkt')

from nltk.tokenize import PunktSentenceTokenizer, TreebankWordTokenizer

# 初始化分词器
sent_tokenizer = PunktSentenceTokenizer()
word_tokenizer = TreebankWordTokenizer()

## 将一段文本转换 为ontonotes 格式的分词，即分句并分词，同时记录每个词的字符级位置（char span）映射，
def get_ontonotes(captions):

    # 句子切分（带字符位置信息）
    sent_spans = list(sent_tokenizer.span_tokenize(captions))
    # 构造 token 到原文 char span 的映射
    ontonotes_tokens = []
    token_char_spans = []  # 与 ontonotes_tokens 同结构，记录每个 token 的 char span

    for sent_start, sent_end in sent_spans:
        sent_text = captions[sent_start:sent_end]
        token_spans = list(word_tokenizer.span_tokenize(sent_text))

        sent_tokens = []
        sent_token_spans = []

        for start, end in token_spans:
            token = sent_text[start:end]
            sent_tokens.append(token)
            sent_token_spans.append((sent_start + start, sent_start + end))  # 转为全文级 char span

        ontonotes_tokens.append(sent_tokens)
        token_char_spans.append(sent_token_spans)

    return ontonotes_tokens, token_char_spans


# 查找目标 span 对应的 token 索引
def find_token_span(span_chars, token_char_spans):
    matched_segments = []
    matched_index_ranges = []
    for start_a, end_a in span_chars:
        segments = []
        indices = []
        for i, (start_b, end_b) in enumerate(token_char_spans):
            if not (end_b < start_a or start_b > end_a):
                segments.append((start_b, end_b))
                indices.append(i)
        matched_segments.append(segments)
        if indices:
            matched_index_ranges.append((indices[0], indices[-1]))
        # else:
        #     matched_index_ranges.append(None)  # 若没有匹配区间
    return matched_segments, matched_index_ranges
    # new_span_chars = []
    # for  span_char in span_chars:
    #     token_char_spans = [x for sub in token_char_spans for x in sub]
    #
    #     token_start = token_end = None
    #     for j, (start, end) in enumerate(token_char_spans):
    #         # 找起始 token
    #         if token_start is None and span_char[0] >= start and  end >= span_char[0] :
    #             token_start = j
    #         # 找结束 token
    #         if token_end is None and span_char[1] <= end and start <= span_char[1]:
    #             token_end = j
    #     if token_start is not None and token_end is not None:
    #         break
    #     assert token_start is not None or token_end is not None
    #     new_span_chars.append([token_start, token_end])
    # return new_span_chars



# def find_token_span(span_chars, token_char_spans):
#     new_span_token_idxs = []
#     global_token_offset = 0
#
#     for span_char in span_chars:
#         start_char, end_char = span_char
#         token_start = token_end = None
#
#         for sent_spans in token_char_spans:
#             for j, (tok_start, tok_end) in enumerate(sent_spans):
#                 g_idx = global_token_offset + j
#
#                 # 与 span 开始位置有重叠
#                 if token_start is None and (tok_start <= start_char < tok_end or start_char <= tok_start < end_char):
#                     token_start = g_idx
#
#                 # 与 span 结束位置有重叠
#                 if token_end is None and (tok_start < end_char <= tok_end or tok_start < end_char - 1 < tok_end or (end_char - 1 <= tok_start < end_char)):
#                     token_end = g_idx
#
#                 if token_start is not None and token_end is not None:
#                     break
#             if token_start is not None and token_end is not None:
#                 break
#             global_token_offset += len(sent_spans)
#
#         if token_start is not None and token_end is not None:
#             new_span_token_idxs.append([token_start, token_end])
#         else:
#             raise AssertionError(f"Span {span_char} 没有找到匹配的 token")
#
#     return new_span_token_idxs






# # 应用
# result = find_token_span(span_char, token_char_spans)
# print(result)  # 应该输出: (0, 0, 1)

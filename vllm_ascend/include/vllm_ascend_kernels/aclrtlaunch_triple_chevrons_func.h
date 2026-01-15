
#ifndef HEADER_ACLRTLAUNCH_BGMV_EXPAND_BFLOAT16_T_HKERNEL_H_
#define HEADER_ACLRTLAUNCH_BGMV_EXPAND_BFLOAT16_T_HKERNEL_H_



extern "C" uint32_t aclrtlaunch_bgmv_expand_bfloat16_t(uint32_t blockDim, void* stream, void* x, void* weight, void* indices, uint32_t indicesSize, void* yIn, void* yOut, uint32_t batchSize, uint32_t numTokensPerCore, uint32_t maxLoRARank, uint32_t outputHiddenDim, uint32_t sliceOffset, uint32_t outputFullDim);

inline uint32_t bgmv_expand_bfloat16_t(uint32_t blockDim, void* hold, void* stream, void* x, void* weight, void* indices, uint32_t indicesSize, void* yIn, void* yOut, uint32_t batchSize, uint32_t numTokensPerCore, uint32_t maxLoRARank, uint32_t outputHiddenDim, uint32_t sliceOffset, uint32_t outputFullDim)
{
    (void)hold;
    return aclrtlaunch_bgmv_expand_bfloat16_t(blockDim, stream, x, weight, indices, indicesSize, yIn, yOut, batchSize, numTokensPerCore, maxLoRARank, outputHiddenDim, sliceOffset, outputFullDim);
}

#endif

#ifndef HEADER_ACLRTLAUNCH_BGMV_EXPAND_HALF_HKERNEL_H_
#define HEADER_ACLRTLAUNCH_BGMV_EXPAND_HALF_HKERNEL_H_



extern "C" uint32_t aclrtlaunch_bgmv_expand_half(uint32_t blockDim, void* stream, void* x, void* weight, void* indices, uint32_t indicesSize, void* yIn, void* yOut, uint32_t batchSize, uint32_t numTokensPerCore, uint32_t maxLoRARank, uint32_t outputHiddenDim, uint32_t sliceOffset, uint32_t outputFullDim);

inline uint32_t bgmv_expand_half(uint32_t blockDim, void* hold, void* stream, void* x, void* weight, void* indices, uint32_t indicesSize, void* yIn, void* yOut, uint32_t batchSize, uint32_t numTokensPerCore, uint32_t maxLoRARank, uint32_t outputHiddenDim, uint32_t sliceOffset, uint32_t outputFullDim)
{
    (void)hold;
    return aclrtlaunch_bgmv_expand_half(blockDim, stream, x, weight, indices, indicesSize, yIn, yOut, batchSize, numTokensPerCore, maxLoRARank, outputHiddenDim, sliceOffset, outputFullDim);
}

#endif

#ifndef HEADER_ACLRTLAUNCH_BGMV_SHRINK_BFLOAT16_T_HKERNEL_H_
#define HEADER_ACLRTLAUNCH_BGMV_SHRINK_BFLOAT16_T_HKERNEL_H_



extern "C" uint32_t aclrtlaunch_bgmv_shrink_bfloat16_t(uint32_t blockDim, void* stream, void* x, void* weight, void* indices, uint32_t indicesSize, void* y, uint32_t batchSize, uint32_t numTokensPerCore, uint32_t inputHiddenDim, uint32_t maxLoRARank, float scale);

inline uint32_t bgmv_shrink_bfloat16_t(uint32_t blockDim, void* hold, void* stream, void* x, void* weight, void* indices, uint32_t indicesSize, void* y, uint32_t batchSize, uint32_t numTokensPerCore, uint32_t inputHiddenDim, uint32_t maxLoRARank, float scale)
{
    (void)hold;
    return aclrtlaunch_bgmv_shrink_bfloat16_t(blockDim, stream, x, weight, indices, indicesSize, y, batchSize, numTokensPerCore, inputHiddenDim, maxLoRARank, scale);
}

#endif

#ifndef HEADER_ACLRTLAUNCH_BGMV_SHRINK_HALF_HKERNEL_H_
#define HEADER_ACLRTLAUNCH_BGMV_SHRINK_HALF_HKERNEL_H_



extern "C" uint32_t aclrtlaunch_bgmv_shrink_half(uint32_t blockDim, void* stream, void* x, void* weight, void* indices, uint32_t indicesSize, void* y, uint32_t batchSize, uint32_t numTokensPerCore, uint32_t inputHiddenDim, uint32_t maxLoRARank, float scale);

inline uint32_t bgmv_shrink_half(uint32_t blockDim, void* hold, void* stream, void* x, void* weight, void* indices, uint32_t indicesSize, void* y, uint32_t batchSize, uint32_t numTokensPerCore, uint32_t inputHiddenDim, uint32_t maxLoRARank, float scale)
{
    (void)hold;
    return aclrtlaunch_bgmv_shrink_half(blockDim, stream, x, weight, indices, indicesSize, y, batchSize, numTokensPerCore, inputHiddenDim, maxLoRARank, scale);
}

#endif

#ifndef HEADER_ACLRTLAUNCH_GET_MASKED_INPUT_AND_MASK_KERNEL_HKERNEL_H_
#define HEADER_ACLRTLAUNCH_GET_MASKED_INPUT_AND_MASK_KERNEL_HKERNEL_H_



extern "C" uint32_t aclrtlaunch_get_masked_input_and_mask_kernel(uint32_t blockDim, void* stream, void* input, void* masked_input, void* mask_out, const int64_t org_vocab_start_index, const int64_t org_vocab_end_index, const int64_t num_org_vocab_padding, const int64_t added_vocab_start_index, const int64_t added_vocab_end_index, const int64_t size, const uint32_t loop_cnt, const uint32_t aiv_num);

inline uint32_t get_masked_input_and_mask_kernel(uint32_t blockDim, void* hold, void* stream, void* input, void* masked_input, void* mask_out, const int64_t org_vocab_start_index, const int64_t org_vocab_end_index, const int64_t num_org_vocab_padding, const int64_t added_vocab_start_index, const int64_t added_vocab_end_index, const int64_t size, const uint32_t loop_cnt, const uint32_t aiv_num)
{
    (void)hold;
    return aclrtlaunch_get_masked_input_and_mask_kernel(blockDim, stream, input, masked_input, mask_out, org_vocab_start_index, org_vocab_end_index, num_org_vocab_padding, added_vocab_start_index, added_vocab_end_index, size, loop_cnt, aiv_num);
}

#endif

#ifndef HEADER_ACLRTLAUNCH_ROPE_CUSTOM_FALSE_BFLOAT16_T_HKERNEL_H_
#define HEADER_ACLRTLAUNCH_ROPE_CUSTOM_FALSE_BFLOAT16_T_HKERNEL_H_



extern "C" uint32_t aclrtlaunch_rope_custom_false_bfloat16_t(uint32_t blockDim, void* stream, void* positions, void* queryDst, void* keyDst, void* query, void* key, void* cosSinCache, const int rotDim, const int64_t queryStride, const int64_t keyStride, const int64_t dstQueryStride, const int64_t dstKeyStride, const int numHeads, const int numKvHeads, const int headSize, const int64_t numTokens, const int loopNum, const int coreNum);

inline uint32_t rope_custom_false_bfloat16_t(uint32_t blockDim, void* hold, void* stream, void* positions, void* queryDst, void* keyDst, void* query, void* key, void* cosSinCache, const int rotDim, const int64_t queryStride, const int64_t keyStride, const int64_t dstQueryStride, const int64_t dstKeyStride, const int numHeads, const int numKvHeads, const int headSize, const int64_t numTokens, const int loopNum, const int coreNum)
{
    (void)hold;
    return aclrtlaunch_rope_custom_false_bfloat16_t(blockDim, stream, positions, queryDst, keyDst, query, key, cosSinCache, rotDim, queryStride, keyStride, dstQueryStride, dstKeyStride, numHeads, numKvHeads, headSize, numTokens, loopNum, coreNum);
}

#endif

#ifndef HEADER_ACLRTLAUNCH_ROPE_CUSTOM_FALSE_HALF_HKERNEL_H_
#define HEADER_ACLRTLAUNCH_ROPE_CUSTOM_FALSE_HALF_HKERNEL_H_



extern "C" uint32_t aclrtlaunch_rope_custom_false_half(uint32_t blockDim, void* stream, void* positions, void* queryDst, void* keyDst, void* query, void* key, void* cosSinCache, const int rotDim, const int64_t queryStride, const int64_t keyStride, const int64_t dstQueryStride, const int64_t dstKeyStride, const int numHeads, const int numKvHeads, const int headSize, const int64_t numTokens, const int loopNum, const int coreNum);

inline uint32_t rope_custom_false_half(uint32_t blockDim, void* hold, void* stream, void* positions, void* queryDst, void* keyDst, void* query, void* key, void* cosSinCache, const int rotDim, const int64_t queryStride, const int64_t keyStride, const int64_t dstQueryStride, const int64_t dstKeyStride, const int numHeads, const int numKvHeads, const int headSize, const int64_t numTokens, const int loopNum, const int coreNum)
{
    (void)hold;
    return aclrtlaunch_rope_custom_false_half(blockDim, stream, positions, queryDst, keyDst, query, key, cosSinCache, rotDim, queryStride, keyStride, dstQueryStride, dstKeyStride, numHeads, numKvHeads, headSize, numTokens, loopNum, coreNum);
}

#endif

#ifndef HEADER_ACLRTLAUNCH_ROPE_CUSTOM_TRUE_BFLOAT16_T_HKERNEL_H_
#define HEADER_ACLRTLAUNCH_ROPE_CUSTOM_TRUE_BFLOAT16_T_HKERNEL_H_



extern "C" uint32_t aclrtlaunch_rope_custom_true_bfloat16_t(uint32_t blockDim, void* stream, void* positions, void* queryDst, void* keyDst, void* query, void* key, void* cosSinCache, const int rotDim, const int64_t queryStride, const int64_t keyStride, const int64_t dstQueryStride, const int64_t dstKeyStride, const int numHeads, const int numKvHeads, const int headSize, const int64_t numTokens, const int loopNum, const int coreNum);

inline uint32_t rope_custom_true_bfloat16_t(uint32_t blockDim, void* hold, void* stream, void* positions, void* queryDst, void* keyDst, void* query, void* key, void* cosSinCache, const int rotDim, const int64_t queryStride, const int64_t keyStride, const int64_t dstQueryStride, const int64_t dstKeyStride, const int numHeads, const int numKvHeads, const int headSize, const int64_t numTokens, const int loopNum, const int coreNum)
{
    (void)hold;
    return aclrtlaunch_rope_custom_true_bfloat16_t(blockDim, stream, positions, queryDst, keyDst, query, key, cosSinCache, rotDim, queryStride, keyStride, dstQueryStride, dstKeyStride, numHeads, numKvHeads, headSize, numTokens, loopNum, coreNum);
}

#endif

#ifndef HEADER_ACLRTLAUNCH_ROPE_CUSTOM_TRUE_HALF_HKERNEL_H_
#define HEADER_ACLRTLAUNCH_ROPE_CUSTOM_TRUE_HALF_HKERNEL_H_



extern "C" uint32_t aclrtlaunch_rope_custom_true_half(uint32_t blockDim, void* stream, void* positions, void* queryDst, void* keyDst, void* query, void* key, void* cosSinCache, const int rotDim, const int64_t queryStride, const int64_t keyStride, const int64_t dstQueryStride, const int64_t dstKeyStride, const int numHeads, const int numKvHeads, const int headSize, const int64_t numTokens, const int loopNum, const int coreNum);

inline uint32_t rope_custom_true_half(uint32_t blockDim, void* hold, void* stream, void* positions, void* queryDst, void* keyDst, void* query, void* key, void* cosSinCache, const int rotDim, const int64_t queryStride, const int64_t keyStride, const int64_t dstQueryStride, const int64_t dstKeyStride, const int numHeads, const int numKvHeads, const int headSize, const int64_t numTokens, const int loopNum, const int coreNum)
{
    (void)hold;
    return aclrtlaunch_rope_custom_true_half(blockDim, stream, positions, queryDst, keyDst, query, key, cosSinCache, rotDim, queryStride, keyStride, dstQueryStride, dstKeyStride, numHeads, numKvHeads, headSize, numTokens, loopNum, coreNum);
}

#endif

#ifndef HEADER_ACLRTLAUNCH_SGMV_EXPAND_BFLOAT16_T_HKERNEL_H_
#define HEADER_ACLRTLAUNCH_SGMV_EXPAND_BFLOAT16_T_HKERNEL_H_



extern "C" uint32_t aclrtlaunch_sgmv_expand_bfloat16_t(uint32_t blockDim, void* stream, void* x, void* weight, void* loraIndices, uint32_t loraIndicesSize, void* seqLen, uint32_t seqLenSize, void* yIn, void* yOut, uint32_t batchSize, uint32_t numTokensPerCore, uint32_t maxLoRARank, uint32_t outputHiddenDim, uint32_t sliceOffset, uint32_t outputFullDim);

inline uint32_t sgmv_expand_bfloat16_t(uint32_t blockDim, void* hold, void* stream, void* x, void* weight, void* loraIndices, uint32_t loraIndicesSize, void* seqLen, uint32_t seqLenSize, void* yIn, void* yOut, uint32_t batchSize, uint32_t numTokensPerCore, uint32_t maxLoRARank, uint32_t outputHiddenDim, uint32_t sliceOffset, uint32_t outputFullDim)
{
    (void)hold;
    return aclrtlaunch_sgmv_expand_bfloat16_t(blockDim, stream, x, weight, loraIndices, loraIndicesSize, seqLen, seqLenSize, yIn, yOut, batchSize, numTokensPerCore, maxLoRARank, outputHiddenDim, sliceOffset, outputFullDim);
}

#endif

#ifndef HEADER_ACLRTLAUNCH_SGMV_EXPAND_HALF_HKERNEL_H_
#define HEADER_ACLRTLAUNCH_SGMV_EXPAND_HALF_HKERNEL_H_



extern "C" uint32_t aclrtlaunch_sgmv_expand_half(uint32_t blockDim, void* stream, void* x, void* weight, void* loraIndices, uint32_t loraIndicesSize, void* seqLen, uint32_t seqLenSize, void* yIn, void* yOut, uint32_t batchSize, uint32_t numTokensPerCore, uint32_t maxLoRARank, uint32_t outputHiddenDim, uint32_t sliceOffset, uint32_t outputFullDim);

inline uint32_t sgmv_expand_half(uint32_t blockDim, void* hold, void* stream, void* x, void* weight, void* loraIndices, uint32_t loraIndicesSize, void* seqLen, uint32_t seqLenSize, void* yIn, void* yOut, uint32_t batchSize, uint32_t numTokensPerCore, uint32_t maxLoRARank, uint32_t outputHiddenDim, uint32_t sliceOffset, uint32_t outputFullDim)
{
    (void)hold;
    return aclrtlaunch_sgmv_expand_half(blockDim, stream, x, weight, loraIndices, loraIndicesSize, seqLen, seqLenSize, yIn, yOut, batchSize, numTokensPerCore, maxLoRARank, outputHiddenDim, sliceOffset, outputFullDim);
}

#endif

#ifndef HEADER_ACLRTLAUNCH_SGMV_SHRINK_BFLOAT16_T_HKERNEL_H_
#define HEADER_ACLRTLAUNCH_SGMV_SHRINK_BFLOAT16_T_HKERNEL_H_



extern "C" uint32_t aclrtlaunch_sgmv_shrink_bfloat16_t(uint32_t blockDim, void* stream, void* x, void* weight, void* loraIndices, uint32_t loraIndicesSize, void* seqLen, uint32_t seqLenSize, void* y, uint32_t batchSize, uint32_t numTokensPerCore, uint32_t inputHiddenDim, uint32_t maxLoRARank, float scale);

inline uint32_t sgmv_shrink_bfloat16_t(uint32_t blockDim, void* hold, void* stream, void* x, void* weight, void* loraIndices, uint32_t loraIndicesSize, void* seqLen, uint32_t seqLenSize, void* y, uint32_t batchSize, uint32_t numTokensPerCore, uint32_t inputHiddenDim, uint32_t maxLoRARank, float scale)
{
    (void)hold;
    return aclrtlaunch_sgmv_shrink_bfloat16_t(blockDim, stream, x, weight, loraIndices, loraIndicesSize, seqLen, seqLenSize, y, batchSize, numTokensPerCore, inputHiddenDim, maxLoRARank, scale);
}

#endif

#ifndef HEADER_ACLRTLAUNCH_SGMV_SHRINK_HALF_HKERNEL_H_
#define HEADER_ACLRTLAUNCH_SGMV_SHRINK_HALF_HKERNEL_H_



extern "C" uint32_t aclrtlaunch_sgmv_shrink_half(uint32_t blockDim, void* stream, void* x, void* weight, void* loraIndices, uint32_t loraIndicesSize, void* seqLen, uint32_t seqLenSize, void* y, uint32_t batchSize, uint32_t numTokensPerCore, uint32_t inputHiddenDim, uint32_t maxLoRARank, float scale);

inline uint32_t sgmv_shrink_half(uint32_t blockDim, void* hold, void* stream, void* x, void* weight, void* loraIndices, uint32_t loraIndicesSize, void* seqLen, uint32_t seqLenSize, void* y, uint32_t batchSize, uint32_t numTokensPerCore, uint32_t inputHiddenDim, uint32_t maxLoRARank, float scale)
{
    (void)hold;
    return aclrtlaunch_sgmv_shrink_half(blockDim, stream, x, weight, loraIndices, loraIndicesSize, seqLen, seqLenSize, y, batchSize, numTokensPerCore, inputHiddenDim, maxLoRARank, scale);
}

#endif

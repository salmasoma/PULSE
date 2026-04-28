#import "PULSEMoondreamBridge.h"

#import <Foundation/Foundation.h>

#import <llama/llama.h>
#import <llama/mtmd.h>
#import <llama/mtmd-helper.h>

#include <string>
#include <vector>

namespace {

static NSString * const PULSEMoondreamErrorDomain = @"PULSEMoondreamBridge";

NSError * pulseMoondreamError(NSInteger code, NSString *description) {
    return [NSError errorWithDomain:PULSEMoondreamErrorDomain
                               code:code
                           userInfo:@{NSLocalizedDescriptionKey: description}];
}

std::string tokenToString(const llama_vocab * vocab, llama_token token) {
    std::string piece(256, '\0');
    int32_t n = llama_token_to_piece(vocab, token, &piece[0], (int32_t) piece.size(), 0, false);
    if (n < 0) {
        piece.resize((size_t) -n);
        n = llama_token_to_piece(vocab, token, &piece[0], (int32_t) piece.size(), 0, false);
    }
    if (n < 0) {
        return {};
    }
    piece.resize((size_t) n);
    return piece;
}

std::string applyChatTemplate(const llama_model * model, const std::string & userPrompt) {
    const char * tmpl = llama_model_chat_template(model, nullptr);
    if (tmpl == nullptr) {
        return userPrompt;
    }

    llama_chat_message message;
    message.role = "user";
    message.content = userPrompt.c_str();

    int32_t required = llama_chat_apply_template(tmpl, &message, 1, true, nullptr, 0);
    if (required <= 0) {
        return userPrompt;
    }

    std::vector<char> buffer((size_t) required + 1, '\0');
    int32_t written = llama_chat_apply_template(tmpl, &message, 1, true, buffer.data(), (int32_t) buffer.size());
    if (written < 0) {
        buffer.assign((size_t) (-written) + 1, '\0');
        written = llama_chat_apply_template(tmpl, &message, 1, true, buffer.data(), (int32_t) buffer.size());
    }
    if (written <= 0) {
        return userPrompt;
    }

    return std::string(buffer.data(), (size_t) written);
}

llama_sampler * makeSampler(int32_t topK, float topP, float temperature) {
    llama_sampler_chain_params params = llama_sampler_chain_default_params();
    llama_sampler * sampler = llama_sampler_chain_init(params);
    if (topK > 0) {
        llama_sampler_chain_add(sampler, llama_sampler_init_top_k(topK));
    }
    if (topP > 0.0f && topP < 1.0f) {
        llama_sampler_chain_add(sampler, llama_sampler_init_top_p(topP, 1));
    }
    if (temperature > 0.0f) {
        llama_sampler_chain_add(sampler, llama_sampler_init_temp(temperature));
    }
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(0));
    return sampler;
}

} // namespace

@interface PULSEMoondreamBridge () {
    struct llama_model * _model;
    struct llama_context * _context;
    struct mtmd_context * _mtmd;
    const struct llama_vocab * _vocab;
    NSInteger _batchSize;
    NSString * _cachedImagePath;
    std::vector<float> _cachedImageEmbeddings;
}
@end

@implementation PULSEMoondreamBridge

+ (void)initializeBackendIfNeeded {
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        llama_backend_init();
    });
}

- (nullable instancetype)initWithModelPath:(NSString *)modelPath
                                mmprojPath:(NSString *)mmprojPath
                             contextLength:(NSInteger)contextLength
                                 batchSize:(NSInteger)batchSize
                                   threads:(NSInteger)threads
                                  useMetal:(BOOL)useMetal
                                     error:(NSError **)error {
    self = [super init];
    if (!self) {
        return nil;
    }

    [[self class] initializeBackendIfNeeded];

    _batchSize = MAX((NSInteger) 8, batchSize);

    llama_model_params modelParams = llama_model_default_params();
    modelParams.n_gpu_layers = useMetal ? 999 : 0;
    modelParams.use_mmap = true;
    modelParams.use_mlock = false;
    modelParams.check_tensors = false;

    _model = llama_model_load_from_file(modelPath.fileSystemRepresentation, modelParams);
    if (_model == nullptr) {
        if (error) {
            *error = pulseMoondreamError(3001, @"Failed to load the local VQA GGUF text model.");
        }
        return nil;
    }

    llama_context_params contextParams = llama_context_default_params();
    contextParams.n_ctx = (uint32_t) MAX(contextLength, 1024);
    contextParams.n_batch = (uint32_t) _batchSize;
    contextParams.n_ubatch = (uint32_t) _batchSize;
    contextParams.n_seq_max = 1;
    contextParams.n_threads = (int32_t) MAX(threads, 1);
    contextParams.n_threads_batch = (int32_t) MAX(threads, 1);
    contextParams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;

    _context = llama_init_from_model(_model, contextParams);
    if (_context == nullptr) {
        if (error) {
            *error = pulseMoondreamError(3002, @"Failed to initialize the local VQA text context.");
        }
        return nil;
    }

    llama_set_n_threads(_context, (int32_t) MAX(threads, 1), (int32_t) MAX(threads, 1));
    _vocab = llama_model_get_vocab(_model);

    mtmd_context_params mtmdParams = mtmd_context_params_default();
    mtmdParams.use_gpu = useMetal;
    mtmdParams.print_timings = false;
    mtmdParams.n_threads = (int) MAX(threads, 1);
    mtmdParams.warmup = false;
    mtmdParams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;

    _mtmd = mtmd_init_from_file(mmprojPath.fileSystemRepresentation, _model, mtmdParams);
    if (_mtmd == nullptr) {
        if (error) {
            *error = pulseMoondreamError(3003, @"Failed to initialize the local VQA multimodal projector.");
        }
        return nil;
    }

    if (!mtmd_support_vision(_mtmd)) {
        if (error) {
            *error = pulseMoondreamError(3004, @"The staged local VQA projector does not report vision support.");
        }
        return nil;
    }

    return self;
}

- (void)dealloc {
    if (_mtmd != nullptr) {
        mtmd_free(_mtmd);
        _mtmd = nullptr;
    }
    if (_context != nullptr) {
        llama_free(_context);
        _context = nullptr;
    }
    if (_model != nullptr) {
        llama_model_free(_model);
        _model = nullptr;
    }
}

- (nullable NSString *)generateWithPrompt:(NSString *)prompt
                                imagePath:(NSString *)imagePath
                                maxTokens:(NSInteger)maxTokens
                              temperature:(float)temperature
                                     topK:(NSInteger)topK
                                     topP:(float)topP
                                    error:(NSError **)error {
    if (_model == nullptr || _context == nullptr || _mtmd == nullptr || _vocab == nullptr) {
        if (error) {
            *error = pulseMoondreamError(3010, @"The local VQA runtime was not initialized.");
        }
        return nil;
    }

    llama_memory_clear(llama_get_memory(_context), true);

    std::string userPrompt = std::string(prompt.UTF8String ?: "");
    if (userPrompt.find(mtmd_default_marker()) == std::string::npos) {
        if (!userPrompt.empty() && userPrompt.back() != '\n') {
            userPrompt += "\n";
        }
        userPrompt += mtmd_default_marker();
    }
    std::string formattedPrompt = applyChatTemplate(_model, userPrompt);

    mtmd_input_text inputText;
    inputText.text = formattedPrompt.c_str();
    inputText.add_special = true;
    inputText.parse_special = true;

    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    if (chunks == nullptr) {
        if (error) {
            *error = pulseMoondreamError(3012, @"Failed to allocate multimodal input chunks.");
        }
        return nil;
    }

    mtmd_bitmap * bitmap = mtmd_helper_bitmap_init_from_file(_mtmd, imagePath.fileSystemRepresentation);
    if (bitmap == nullptr) {
        mtmd_input_chunks_free(chunks);
        if (error) {
            *error = pulseMoondreamError(3011, @"Failed to decode the ultrasound image into a multimodal bitmap.");
        }
        return nil;
    }

    const mtmd_bitmap * bitmaps[] = { bitmap };
    const int32_t tokenizeResult = mtmd_tokenize(_mtmd, chunks, &inputText, bitmaps, 1);
    mtmd_bitmap_free(bitmap);
    if (tokenizeResult != 0) {
        mtmd_input_chunks_free(chunks);
        if (error) {
            *error = pulseMoondreamError(3013, [NSString stringWithFormat:@"mtmd_tokenize failed with code %d.", tokenizeResult]);
        }
        return nil;
    }

    llama_pos newNPast = 0;
    const int32_t evalResult = mtmd_helper_eval_chunks(
        _mtmd,
        _context,
        chunks,
        0,
        0,
        (int32_t) _batchSize,
        true,
        &newNPast
    );
    mtmd_input_chunks_free(chunks);
    if (evalResult != 0) {
        if (error) {
            *error = pulseMoondreamError(3014, [NSString stringWithFormat:@"mtmd prompt evaluation failed with code %d.", evalResult]);
        }
        return nil;
    }

    llama_sampler * sampler = makeSampler((int32_t) topK, topP, temperature);
    std::string output;
    output.reserve((size_t) MAX(maxTokens, 128) * 4);

    for (NSInteger step = 0; step < maxTokens; step++) {
        const llama_token token = llama_sampler_sample(sampler, _context, -1);
        if (llama_vocab_is_eog(_vocab, token)) {
            break;
        }

        output += tokenToString(_vocab, token);
        llama_sampler_accept(sampler, token);

        llama_token decodeToken = token;
        llama_batch batch = llama_batch_get_one(&decodeToken, 1);
        const int32_t decodeResult = llama_decode(_context, batch);
        if (decodeResult != 0) {
            llama_sampler_free(sampler);
            if (error) {
                *error = pulseMoondreamError(3015, [NSString stringWithFormat:@"llama_decode failed with code %d.", decodeResult]);
            }
            return nil;
        }
    }

    llama_sampler_free(sampler);

    NSString * result = [[NSString alloc] initWithBytes:output.data()
                                                 length:output.size()
                                               encoding:NSUTF8StringEncoding];
    if (result == nil) {
        if (error) {
            *error = pulseMoondreamError(3016, @"The generated local VQA output was not valid UTF-8.");
        }
        return nil;
    }

    return [result stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
}

- (BOOL)prepareImageAtPath:(NSString *)imagePath
                     error:(NSError **)error {
    if (_model == nullptr || _context == nullptr || _mtmd == nullptr || _vocab == nullptr) {
        if (error) {
            *error = pulseMoondreamError(3020, @"The local VQA runtime was not initialized.");
        }
        return NO;
    }

    mtmd_input_text inputText;
    inputText.text = mtmd_default_marker();
    inputText.add_special = true;
    inputText.parse_special = true;

    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    if (chunks == nullptr) {
        if (error) {
            *error = pulseMoondreamError(3021, @"Failed to allocate multimodal chunks for image prewarm.");
        }
        return NO;
    }

    mtmd_bitmap * bitmap = mtmd_helper_bitmap_init_from_file(_mtmd, imagePath.fileSystemRepresentation);
    if (bitmap == nullptr) {
        mtmd_input_chunks_free(chunks);
        if (error) {
            *error = pulseMoondreamError(3022, @"Failed to decode the ultrasound image for local VQA prewarm.");
        }
        return NO;
    }

    const mtmd_bitmap * bitmaps[] = { bitmap };
    const int32_t tokenizeResult = mtmd_tokenize(_mtmd, chunks, &inputText, bitmaps, 1);
    mtmd_bitmap_free(bitmap);
    if (tokenizeResult != 0) {
        mtmd_input_chunks_free(chunks);
        if (error) {
            *error = pulseMoondreamError(3023, [NSString stringWithFormat:@"mtmd_tokenize failed during image prewarm with code %d.", tokenizeResult]);
        }
        return NO;
    }

    BOOL success = NO;
    const size_t nChunks = mtmd_input_chunks_size(chunks);
    for (size_t idx = 0; idx < nChunks; ++idx) {
        const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks, idx);
        if (chunk != nullptr && mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            success = [self ensureImageEmbeddingsForChunk:chunk imagePath:imagePath error:error];
            break;
        }
    }

    mtmd_input_chunks_free(chunks);

    if (!success && error && *error == nil) {
        *error = pulseMoondreamError(3024, @"Image prewarm did not yield a valid image chunk.");
    }

    return success;
}

- (BOOL)ensureImageEmbeddingsForChunk:(const mtmd_input_chunk *)chunk
                            imagePath:(NSString *)imagePath
                                error:(NSError **)error {
    const size_t nImageTokens = mtmd_input_chunk_get_n_tokens(chunk);
    const int32_t nEmbd = llama_model_n_embd(_model);
    const size_t expectedFloats = nImageTokens * (size_t) MAX(nEmbd, 0);

    if (_cachedImagePath != nil &&
        [_cachedImagePath isEqualToString:imagePath] &&
        expectedFloats > 0 &&
        _cachedImageEmbeddings.size() == expectedFloats) {
        return YES;
    }

    const int32_t encodeResult = mtmd_encode_chunk(_mtmd, chunk);
    if (encodeResult != 0) {
        if (error) {
            *error = pulseMoondreamError(3025, [NSString stringWithFormat:@"mtmd_encode_chunk failed with code %d.", encodeResult]);
        }
        return NO;
    }

    float * outputEmbd = mtmd_get_output_embd(_mtmd);
    if (outputEmbd == nullptr || expectedFloats == 0) {
        if (error) {
            *error = pulseMoondreamError(3026, @"The multimodal runtime did not return image embeddings.");
        }
        return NO;
    }

    _cachedImageEmbeddings.assign(outputEmbd, outputEmbd + expectedFloats);
    _cachedImagePath = [imagePath copy];
    return YES;
}

@end

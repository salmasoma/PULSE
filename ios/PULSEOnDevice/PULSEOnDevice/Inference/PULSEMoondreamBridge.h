#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface PULSEMoondreamBridge : NSObject

- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;

- (nullable instancetype)initWithModelPath:(NSString *)modelPath
                                mmprojPath:(NSString *)mmprojPath
                             contextLength:(NSInteger)contextLength
                                 batchSize:(NSInteger)batchSize
                                   threads:(NSInteger)threads
                                  useMetal:(BOOL)useMetal
                                     error:(NSError **)error NS_DESIGNATED_INITIALIZER;

- (nullable NSString *)generateWithPrompt:(NSString *)prompt
                                imagePath:(NSString *)imagePath
                                maxTokens:(NSInteger)maxTokens
                              temperature:(float)temperature
                                     topK:(NSInteger)topK
                                     topP:(float)topP
                                    error:(NSError **)error;

- (BOOL)prepareImageAtPath:(NSString *)imagePath
                     error:(NSError **)error;

@end

NS_ASSUME_NONNULL_END

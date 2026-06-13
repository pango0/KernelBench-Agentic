# Qwen2.5-Coder-7B-Instruct x − Code Analyzer

- Run dir: `agentic_no_analyzer_qwen`  |  Hardware: V100_SXM2_32GB  |  Tasks: 30

## Overall

| metric | value |
|---|---|
| Compilation rate | 63.3% (19/30) |
| Correctness rate | 20.0% (6/30) |
| Faster than baseline (fast_1) | 10.0% (3/30) |
| Geo-mean speedup (correct only) | 0.987x |

## Per level

| level | correct | compile% | correct% | geo-mean speedup | fast_1.0 |
|---|---|---|---|---|---|
| 1 | 3/10 | 90% | 30% | 0.970x | 0.1 |
| 2 | 3/10 | 80% | 30% | 1.000x | 0.1 |
| 3 | 0/10 | 20% | 0% | 0.000x | 0.0 |

## Error taxonomy

| category | count |
|---|---|
| compilation | 11 |
| hallucinated_api | 7 |
| shape_mismatch | 2 |
| wrong_output | 4 |
| slow | 3 |

## Per-problem

| L | P | name | compiled | correct | speedup | category |
|---|---|---|---|---|---|---|
| 1 | 1 | 1_Square_matrix_multiplication_.py | Y | N | - | hallucinated_api |
| 1 | 2 | 2_Standard_matrix_multiplication_. | Y | N | - | wrong_output |
| 1 | 3 | 3_Batched_matrix_multiplication.py | N | N | - | compilation |
| 1 | 4 | 4_Matrix_vector_multiplication_.py | Y | N | - | hallucinated_api |
| 1 | 5 | 5_Matrix_scalar_multiplication.py | Y | Y | 0.97x | slow |
| 1 | 6 | 6_Matmul_with_large_K_dimension_.p | Y | Y | 1.00x | correct |
| 1 | 7 | 7_Matmul_with_small_K_dimension_.p | Y | N | - | wrong_output |
| 1 | 8 | 8_Matmul_with_irregular_shapes_.py | Y | Y | 0.95x | slow |
| 1 | 9 | 9_Tall_skinny_matrix_multiplicatio | Y | N | - | hallucinated_api |
| 1 | 10 | 10_3D_tensor_matrix_multiplication | Y | N | - | hallucinated_api |
| 2 | 1 | 1_Conv2D_ReLU_BiasAdd.py | Y | N | - | hallucinated_api |
| 2 | 2 | 2_ConvTranspose2d_BiasAdd_Clamp_Sc | Y | Y | 1.00x | correct |
| 2 | 3 | 3_ConvTranspose3d_Sum_LayerNorm_Av | N | N | - | compilation |
| 2 | 4 | 4_Conv2d_Mish_Mish.py | Y | N | - | shape_mismatch |
| 2 | 5 | 5_ConvTranspose2d_Subtract_Tanh.py | Y | Y | 1.00x | slow |
| 2 | 6 | 6_Conv3d_Softmax_MaxPool_MaxPool.p | Y | N | - | wrong_output |
| 2 | 7 | 7_Conv3d_ReLU_LeakyReLU_GELU_Sigmo | Y | Y | 1.00x | correct |
| 2 | 8 | 8_Conv3d_Divide_Max_GlobalAvgPool_ | Y | N | - | wrong_output |
| 2 | 9 | 9_Matmul_Subtract_Multiply_ReLU.py | Y | N | - | hallucinated_api |
| 2 | 10 | 10_ConvTranspose2d_MaxPool_Hardtan | N | N | - | compilation |
| 3 | 1 | 1_MLP.py | N | N | - | compilation |
| 3 | 2 | 2_ShallowWideMLP.py | N | N | - | compilation |
| 3 | 3 | 3_DeepNarrowMLP.py | N | N | - | compilation |
| 3 | 4 | 4_LeNet5.py | Y | N | - | hallucinated_api |
| 3 | 5 | 5_AlexNet.py | N | N | - | compilation |
| 3 | 6 | 6_GoogleNetInceptionModule.py | N | N | - | compilation |
| 3 | 7 | 7_GoogleNetInceptionV1.py | N | N | - | compilation |
| 3 | 8 | 8_ResNetBasicBlock.py | Y | N | - | shape_mismatch |
| 3 | 9 | 9_ResNet18.py | N | N | - | compilation |
| 3 | 10 | 10_ResNet101.py | N | N | - | compilation |

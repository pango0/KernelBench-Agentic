# Qwen2.5-Coder-7B-Instruct x − Refinement loop (1 turn)

- Run dir: `agentic_single_turn_qwen`  |  Hardware: V100_SXM2_32GB  |  Tasks: 30

## Overall

| metric | value |
|---|---|
| Compilation rate | 70.0% (21/30) |
| Correctness rate | 20.0% (6/30) |
| Faster than baseline (fast_1) | 10.0% (3/30) |
| Geo-mean speedup (correct only) | 0.976x |

## Per level

| level | correct | compile% | correct% | geo-mean speedup | fast_1.0 |
|---|---|---|---|---|---|
| 1 | 3/10 | 90% | 30% | 0.968x | 0.0 |
| 2 | 3/10 | 90% | 30% | 0.977x | 0.1 |
| 3 | 0/10 | 30% | 0% | 0.000x | 0.0 |

## Error taxonomy

| category | count |
|---|---|
| compilation | 9 |
| hallucinated_api | 6 |
| shape_mismatch | 2 |
| wrong_output | 7 |
| slow | 3 |

## Per-problem

| L | P | name | compiled | correct | speedup | category |
|---|---|---|---|---|---|---|
| 1 | 1 | 1_Square_matrix_multiplication_.py | Y | N | - | hallucinated_api |
| 1 | 2 | 2_Standard_matrix_multiplication_. | Y | N | - | shape_mismatch |
| 1 | 3 | 3_Batched_matrix_multiplication.py | Y | Y | 1.00x | correct |
| 1 | 4 | 4_Matrix_vector_multiplication_.py | Y | N | - | hallucinated_api |
| 1 | 5 | 5_Matrix_scalar_multiplication.py | Y | N | - | hallucinated_api |
| 1 | 6 | 6_Matmul_with_large_K_dimension_.p | N | N | - | compilation |
| 1 | 7 | 7_Matmul_with_small_K_dimension_.p | Y | Y | 1.01x | correct |
| 1 | 8 | 8_Matmul_with_irregular_shapes_.py | Y | N | - | wrong_output |
| 1 | 9 | 9_Tall_skinny_matrix_multiplicatio | Y | N | - | wrong_output |
| 1 | 10 | 10_3D_tensor_matrix_multiplication | Y | Y | 0.93x | slow |
| 2 | 1 | 1_Conv2D_ReLU_BiasAdd.py | N | N | - | compilation |
| 2 | 2 | 2_ConvTranspose2d_BiasAdd_Clamp_Sc | Y | Y | 0.93x | slow |
| 2 | 3 | 3_ConvTranspose3d_Sum_LayerNorm_Av | Y | N | - | wrong_output |
| 2 | 4 | 4_Conv2d_Mish_Mish.py | Y | N | - | hallucinated_api |
| 2 | 5 | 5_ConvTranspose2d_Subtract_Tanh.py | Y | N | - | wrong_output |
| 2 | 6 | 6_Conv3d_Softmax_MaxPool_MaxPool.p | Y | N | - | hallucinated_api |
| 2 | 7 |  | Y | N | - | shape_mismatch |
| 2 | 8 | 8_Conv3d_Divide_Max_GlobalAvgPool_ | Y | Y | 0.99x | slow |
| 2 | 9 | 9_Matmul_Subtract_Multiply_ReLU.py | Y | N | - | hallucinated_api |
| 2 | 10 | 10_ConvTranspose2d_MaxPool_Hardtan | Y | Y | 1.00x | correct |
| 3 | 1 |  | N | N | - | compilation |
| 3 | 2 | 2_ShallowWideMLP.py | N | N | - | compilation |
| 3 | 3 | 3_DeepNarrowMLP.py | Y | N | - | wrong_output |
| 3 | 4 | 4_LeNet5.py | Y | N | - | wrong_output |
| 3 | 5 |  | N | N | - | compilation |
| 3 | 6 | 6_GoogleNetInceptionModule.py | N | N | - | compilation |
| 3 | 7 | 7_GoogleNetInceptionV1.py | Y | N | - | wrong_output |
| 3 | 8 | 8_ResNetBasicBlock.py | N | N | - | compilation |
| 3 | 9 |  | N | N | - | compilation |
| 3 | 10 | 10_ResNet101.py | N | N | - | compilation |

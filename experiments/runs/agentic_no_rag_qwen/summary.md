# Qwen2.5-Coder-7B-Instruct x − RAG Researcher

- Run dir: `agentic_no_rag_qwen`  |  Hardware: V100_SXM2_32GB  |  Tasks: 30

## Overall

| metric | value |
|---|---|
| Compilation rate | 76.7% (23/30) |
| Correctness rate | 20.0% (6/30) |
| Faster than baseline (fast_1) | 6.7% (2/30) |
| Geo-mean speedup (correct only) | 0.973x |

## Per level

| level | correct | compile% | correct% | geo-mean speedup | fast_1.0 |
|---|---|---|---|---|---|
| 1 | 3/10 | 70% | 30% | 0.969x | 0.1 |
| 2 | 1/10 | 80% | 10% | 0.970x | 0.0 |
| 3 | 2/10 | 80% | 20% | 0.970x | 0.0 |

## Error taxonomy

| category | count |
|---|---|
| compilation | 6 |
| hallucinated_api | 8 |
| shape_mismatch | 3 |
| wrong_output | 7 |
| slow | 4 |

## Per-problem

| L | P | name | compiled | correct | speedup | category |
|---|---|---|---|---|---|---|
| 1 | 1 | 1_Square_matrix_multiplication_.py | Y | Y | 0.98x | slow |
| 1 | 2 | 2_Standard_matrix_multiplication_. | N | N | - | compilation |
| 1 | 3 | 3_Batched_matrix_multiplication.py | Y | N | - | hallucinated_api |
| 1 | 4 | 4_Matrix_vector_multiplication_.py | N | N | - | hallucinated_api |
| 1 | 5 | 5_Matrix_scalar_multiplication.py | Y | Y | 0.97x | slow |
| 1 | 6 | 6_Matmul_with_large_K_dimension_.p | Y | Y | 1.00x | correct |
| 1 | 7 | 7_Matmul_with_small_K_dimension_.p | Y | N | - | hallucinated_api |
| 1 | 8 | 8_Matmul_with_irregular_shapes_.py | N | N | - | compilation |
| 1 | 9 | 9_Tall_skinny_matrix_multiplicatio | Y | N | - | hallucinated_api |
| 1 | 10 | 10_3D_tensor_matrix_multiplication | Y | N | - | wrong_output |
| 2 | 1 | 1_Conv2D_ReLU_BiasAdd.py | Y | N | - | hallucinated_api |
| 2 | 2 | 2_ConvTranspose2d_BiasAdd_Clamp_Sc | Y | N | - | shape_mismatch |
| 2 | 3 | 3_ConvTranspose3d_Sum_LayerNorm_Av | Y | N | - | wrong_output |
| 2 | 4 | 4_Conv2d_Mish_Mish.py | N | N | - | compilation |
| 2 | 5 | 5_ConvTranspose2d_Subtract_Tanh.py | Y | N | - | wrong_output |
| 2 | 6 | 6_Conv3d_Softmax_MaxPool_MaxPool.p | Y | Y | 0.95x | slow |
| 2 | 7 | 7_Conv3d_ReLU_LeakyReLU_GELU_Sigmo | Y | N | - | hallucinated_api |
| 2 | 8 | 8_Conv3d_Divide_Max_GlobalAvgPool_ | Y | N | - | hallucinated_api |
| 2 | 9 | 9_Matmul_Subtract_Multiply_ReLU.py | Y | N | - | shape_mismatch |
| 2 | 10 | 10_ConvTranspose2d_MaxPool_Hardtan | N | N | - | compilation |
| 3 | 1 | 1_MLP.py | Y | N | - | shape_mismatch |
| 3 | 2 | 2_ShallowWideMLP.py | Y | Y | 1.00x | correct |
| 3 | 3 | 3_DeepNarrowMLP.py | Y | N | - | wrong_output |
| 3 | 4 | 4_LeNet5.py | Y | Y | 0.94x | slow |
| 3 | 5 | 5_AlexNet.py | Y | N | - | wrong_output |
| 3 | 6 | 6_GoogleNetInceptionModule.py | Y | N | - | hallucinated_api |
| 3 | 7 | 7_GoogleNetInceptionV1.py | N | N | - | compilation |
| 3 | 8 | 8_ResNetBasicBlock.py | Y | N | - | wrong_output |
| 3 | 9 | 9_ResNet18.py | Y | N | - | wrong_output |
| 3 | 10 | 10_ResNet101.py | N | N | - | compilation |

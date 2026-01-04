#!/bin/bash

cmake -B build
cmake --build build --config Release -j 4

conda create --name vlut
conda activate vlut
pip install -r requirements.txt

hf download bitLLM/bitnet_b1_58-3B --local-dir /workspace/vlut.cpp/models/

model='bitnet_b1_58-3B';
python3 ./convert_hf_to_gguf_vlut.py ./models/"${model}" --outfile ./models/"${model}"/"${model}".vlut.gguf

./build/bin/llama-quantize ./models/"${model}"/bitnet_b1_58-3B.vlut.gguf I2_V_4

./build/bin/llama-batched \
  -m ./models/bitnet_b1_58-3B/ggml-model-I2_V_4.gguf \
  -p "I believe the meaning of life is" \
  -np 1 -n 16 -t 4 --temp 0.5 --repeat-penalty 1.5
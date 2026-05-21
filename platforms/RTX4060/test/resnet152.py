import time
import torch
import torchvision
import argparse

BATCH_SIZE = 16

def infer(model, input):
    with torch.no_grad():
        return model(input).cpu()

def run(run_cnt):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; RTX4060 test requires an NVIDIA GPU")

    device = torch.device("cuda:0")
    model = torchvision.models.resnet152(weights=torchvision.models.ResNet152_Weights.DEFAULT)
    model.eval().to(device)
    input = torch.ones(BATCH_SIZE, 3, 224, 224, device=device)

    print(infer(model, input))

    while True:
        start = time.time()
        for _ in range(run_cnt):
            infer(model, input)
        end = time.time()
        print(f"thpt: {BATCH_SIZE * run_cnt / (end - start):.2f} img/s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ResNet152 inference on NVIDIA GPU (RTX4060)")
    parser.add_argument("-c", "--run-cnt", type=int, default=10, help="Run count for inference")
    args = parser.parse_args()
    run(args.run_cnt)
